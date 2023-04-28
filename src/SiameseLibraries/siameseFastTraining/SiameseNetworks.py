import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import siameseFastTraining.Metrics
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.subjects = [subj for subj in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subj))]
        self.subject_to_images = {subject: os.listdir(os.path.join(root_dir, subject)) for subject in self.subjects}

    def __len__(self):
        total_images = 0
        for subject_images in self.subject_to_images.values():
            total_images += len(subject_images)
        return total_images

    def __getitem__(self, index):
        anchor_subject, anchor_image_path = self.get_subject_image_path(index)
        positive_image_path = self.get_positive_image(anchor_subject, anchor_image_path)
        negative_image_path = self.get_negative_image(anchor_subject)

        anchor_image = Image.open(anchor_image_path).convert("RGB")
        positive_image = Image.open(positive_image_path).convert("RGB")
        negative_image = Image.open(negative_image_path).convert("RGB")

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image

    def get_subject_image_path(self, index):
        subject_index = 0
        subject = self.subjects[subject_index]
        count = len(self.subject_to_images[subject])

        while index >= count:
            index -= count
            subject_index += 1
            subject = self.subjects[subject_index]
            count = len(self.subject_to_images[subject])

        image_name = self.subject_to_images[subject][index]
        image_path = os.path.join(self.root_dir, subject, image_name)
        return subject, image_path

    def get_positive_image(self, anchor_subject, anchor_image_path):
        subject_images = self.subject_to_images[anchor_subject]
        positive_image_name = random.choice(
            [img for img in subject_images if os.path.join(anchor_subject, img) != anchor_image_path]
        )
        positive_image_path = os.path.join(self.root_dir, anchor_subject, positive_image_name)
        return positive_image_path

    def get_negative_image(self, anchor_subject):
        negative_subject = random.choice([subj for subj in self.subjects if subj != anchor_subject])
        negative_image_name = random.choice(self.subject_to_images[negative_subject])
        negative_image_path = os.path.join(self.root_dir, negative_subject, negative_image_name)
        return negative_image_path


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, model_name="efficientnet-b0", pretrained=True):
        super(EfficientNetFeatureExtractor, self).__init__()

        self.efficient_net = (
            EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
        )

        # Get the number of output features from the EfficientNet model
        num_output_features = self.efficient_net._fc.in_features

        # Remove the classification head to use it as a feature extractor
        self.efficient_net._fc = nn.Identity()

        # Define the fully connected layer with the appropriate input size
        self.fc = nn.Linear(num_output_features, 256)  # 128

        # Add a normalization layer
        self.norm = nn.BatchNorm1d(256)  # 128

        # Freeze the pre-trained EfficientNet model parameters
        for param in self.efficient_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.efficient_net(x)
        x = self.fc(x)
        x = self.norm(x)
        return x


class SiameseNetwork(nn.Module):
    def __init__(
        self,
        feature_extractor,
        model_name,
        init_method,
        batch_norm,
        learning_rate,
        epochs,
        dataset,
        loss_function,
        accuracy_threshold,
        img_size,
    ):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = feature_extractor

        # Define parameters for saving:
        self.name = model_name
        self.init_method = init_method
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset = dataset
        self.loss_function = loss_function
        self.img_size = img_size

    def forward(self, anchor, positive, negative):
        output1 = self.feature_extractor(anchor)
        output2 = self.feature_extractor(positive)
        output3 = self.feature_extractor(negative)
        return output1, output2, output3

    def compare_images(self, preprocessed_image1, preprocessed_image2, threshold):
        self.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            # Extract the feature vectors for the two images
            feature_vector1 = self.feature_extractor(preprocessed_image1)
            feature_vector2 = self.feature_extractor(preprocessed_image2)

            # Calculate the similarity between the feature vectors
            similarity = torch.norm(feature_vector1 - feature_vector2).item()

        # Compare the similarity to the threshold and return the result
        return similarity <= threshold  # <= threshold   # Add for returning a True/False


class Trainer:
    def __init__(
        self,
        siamese_network,
        model_name,
        init_method,
        batch_norm,
        learning_rate,
        epochs,
        dataset,
        img_size,
        optimizer,
        loss_function,
    ):
        self.siamese_network = siamese_network
        self.name = model_name
        self.init_method = init_method
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset = dataset
        self.img_size = img_size
        self.optimizer = optimizer
        self.loss_function = loss_function

    def train(self, siamese_train_dataloader, siamese_val_dataloader, epochs: int or None = None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if epochs is None:
            epochs = self.epochs

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        self.siamese_network = self.siamese_network.to(device)

        # Assuming you have a validation DataLoader named val_dataloader
        for epoch in range(epochs):
            # Training loop
            self.siamese_network.train()
            train_loss = 0.0
            train_accuracy = 0.0
            for batch, (image_batch_anchor, image_batch_positive, image_batch_negative) in tqdm(
                enumerate(siamese_train_dataloader), total=len(siamese_train_dataloader), desc="Training"
            ):

                image_batch_anchor = image_batch_anchor.to(device)
                image_batch_positive = image_batch_positive.to(device)
                image_batch_negative = image_batch_negative.to(device)

                anchor, positive, negative = self.siamese_network(
                    image_batch_anchor, image_batch_positive, image_batch_negative
                )
                loss = self.loss_function(anchor, positive, negative)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_accuracy += siameseFastTraining.Metrics.batch_accuracy(anchor, positive, negative)

            train_loss /= len(siamese_train_dataloader)
            train_accuracy /= len(siamese_train_dataloader)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Validation loop
            self.siamese_network.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            with torch.inference_mode():
                for batch, (image_batch_anchor, image_batch_positive, image_batch_negative) in tqdm(
                    enumerate(siamese_val_dataloader), total=len(siamese_val_dataloader), desc="Validation"
                ):

                    image_batch_anchor = image_batch_anchor.to(device)
                    image_batch_positive = image_batch_positive.to(device)
                    image_batch_negative = image_batch_negative.to(device)

                    anchor, positive, negative = self.siamese_network(
                        image_batch_anchor, image_batch_positive, image_batch_negative
                    )
                    loss = self.loss_function(anchor, positive, negative)

                    val_loss += loss.item()
                    val_accuracy += siameseFastTraining.Metrics.batch_accuracy(anchor, positive, negative)

            val_loss /= len(siamese_val_dataloader)
            val_accuracy /= len(siamese_val_dataloader)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(
                f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )
