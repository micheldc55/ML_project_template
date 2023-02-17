import typing

import models.sklearn_model_params


class ScikitModel(typing.Protocol):
    """Base class to type hint sklearn models. This class is not used in the actual code."""

    def fit(self, X, y):
        self.fit(X, y)

    def predict(self, X):
        self.predict(X)


def define_sklearn_model(model_name: str) -> dict:
    """Function that receives a model class from sklearn and a model name and creates a model
    dictionary that contains another dictionray with the model name as a key and the model itself
    and the parameters used as the values. This can be used to track model hyperparameters in
    training.

    :param model: A sklearn model. For example sklearn.tree.DecisionTreeClassifer (note that
    there are not brackets at the end, it's the class, not an instance)
    :type model: ScikitModel
    :param model_name: Name of the model, as saved in the sklearn_model_params.yml file.
    :type model_name: str
    :return: Dictionary that contains another dictionary where the key is the model_name argument,
    and the value is another dictionary with the model (and the parameters set as in the
    sklearn_model_params.yml file for that model_name)
    :rtype: _type_
    """
    MODEL_DICT = models.sklearn_model_params.get_model_dict()

    if model_name in MODEL_DICT.keys():
        specific_model_dict = MODEL_DICT[model_name]
        model_params = specific_model_dict["model_params"]
        model = specific_model_dict["model"]
    else:
        raise KeyError(
            f"""No model under the name "{model_name}" in the sklearn_model_params.yml file 
            please check the "model_name" argument of the function and cross-check it with 
            the keys in the .yml file
            """
        )

    model_dict = {model_name: {"model": model(**model_params), "model_params": model_params}}

    return model_dict


if __name__ == "__main__":
    model_name = "decision_tree_classifier"
    my_model_dict = define_sklearn_model(model_name)
    print(my_model_dict[model_name]["model"])
