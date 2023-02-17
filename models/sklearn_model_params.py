import sklearn.ensemble
import sklearn.tree


def get_model_dict():
    """Function that outputs a dictionary with all the models to be tested

    :return: _description_
    :rtype: _type_
    """
    return {
        "decision_tree_classifier": {
            "model": sklearn.tree.DecisionTreeClassifier,
            "model_params": {"criterion": "gini", "max_depth": 7, "max_features": "sqrt", "random_state": 101},
        },
        "random_forest_classifier": {
            "model": sklearn.ensemble.RandomForestClassifier,
            "model_params": {
                "n_estimators": 501,
                "criterion": "gini",
                "max_depth": 7,
                "max_features": "sqrt",
                "random_state": 101,
            },
        },
    }
