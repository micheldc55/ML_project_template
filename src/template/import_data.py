import os

import pandas as pd
import sklearn.datasets
import sklearn.model_selection

import config


def read_data():
    """Reads the data and returns a DataFrame.

    :return: A Pandas DataFrame with the Iris Dataset
    :rtype: pd.DataFrame
    """
    data_dict = sklearn.datasets.load_iris()
    data = data_dict["data"]
    labels = data_dict["target"]
    data_col_names = data_dict["feature_names"]

    df = pd.DataFrame(data, columns=data_col_names)
    df["label"] = labels
    return df


def generate_folds(df: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # initiate the kfold class from model_selection module
    kf = sklearn.model_selection.KFold(n_splits=n_folds)
    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, "kfold"] = fold

    # save the new csv with kfold column
    path_to_train_file = os.path.join("./input", f"train_{n_folds}_folds.csv")
    df.to_csv(path_to_train_file, index=False)


if __name__ == "__main__":
    df = read_data()
    generate_folds(df, 3)
