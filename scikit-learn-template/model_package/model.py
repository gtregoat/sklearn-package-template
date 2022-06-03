"""
This module contains the model class. It inherits from scikit learn's class Pipeline
so it integrates well in this ecosystem.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn import __version__ as sklearn_version
import joblib
from .version import __version__


class ScikitLearnModel(Pipeline):
    def __init__(self, **model_params):
        """
        Creates a scikit-learn pipeline with two steps: OneHotEncoder (for the categorical variables),
        and a LogisticRegression as the classifier.
        As this inherits from the scikit-learn class Pipeline, it will have all the same methods and attributes,
        with the addition of:
            versions: contains versions at training time. This is used to track if a model can be safely reloaded and
                used.
            save: a method to save the model with joblib.

        :param model_params: parameters for the logistic regression.
        """
        super().__init__(
            steps=[('encoder', OneHotEncoder()), ('clf', LogisticRegression(**model_params))]
        )
        self.versions = {
            "model_package": __version__,
            "scikit-learn": sklearn_version
        }

    def save(self, path: str, compress: int = 3) -> None:
        """
        Saves the model using joblib.

        :param path: path to save the model.
        :param compress: compression ratio to pass to joblib. Between 0 and 9.
        """
        joblib.dump(self, path, compress=compress)
