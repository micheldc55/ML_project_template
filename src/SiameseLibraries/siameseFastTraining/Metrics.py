import torch


def batch_accuracy(anchor_out, positive_out, negative_out):
    positive_distance = (anchor_out - positive_out).pow(2).sum(1).sqrt()
    negative_distance = (anchor_out - negative_out).pow(2).sum(1).sqrt()

    correct = (positive_distance < negative_distance).sum().item()
    total = anchor_out.size(0)

    accuracy = correct / total
    return accuracy
