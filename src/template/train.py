# train.py

import argparse
import os
import sys

# import joblib
import pandas as pd
import sklearn.metrics

import config

sys.path.append("..path_to_base_folder..")
import models.sklearn_model_dispatcher


def run_sklearn(fold: int, model_name: str):
    # read the training data with folds
    path_to_train_file = os.path.join("./input", f"train_folds.csv")
    df = pd.read_csv(path_to_train_file)

    # training data is where kfold is not equal to provided fold
    # also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column from dataframe and convert it to
    # a numpy array by using .values.
    # target is label column in the dataframe
    X_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values
    # similarly, for validation, we have
    X_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    # initialize simple decision tree classifier from sklearn
    model_dict = models.sklearn_model_dispatcher.define_sklearn_model(model_name)
    clf = model_dict[model_name]["model"]
    # fir the model on training data
    clf.fit(X_train, y_train)
    # create predictions for validation samples
    preds = clf.predict(X_valid)
    # calculate & print accuracy
    accuracy = sklearn.metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")
    # save the model
    # joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    # add the different arguments you need and their type
    # currently, we only need fold
    parser.add_argument("--fold", type=int)
    parser.add_argument("--modelname", type=str)
    # read the arguments from the command line
    args = parser.parse_args()

    # run the fold specified by command line arguments
    run_sklearn(
        fold=args.fold,
        model_name=args.modelname,
        # fold=0,
        # model_name="decision_tree_classifier",
    )

    # NOTE: To run, you need to activate the conda environment first. So, in the terminal, you have to run something like:
    # Main_directory % /Users/.../opt/anaconda3/bin/python /Users/..1../..2../src/train.py --fold 0 --modelname "decision_tree_classifier"
