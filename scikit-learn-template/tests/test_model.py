"""
This module tests that the model can be correctly built, fitted and saved.
"""
import pytest
import model_package
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from numpy.testing import assert_array_equal
from unittest.mock import patch


@pytest.fixture
def training_data():
    return pd.DataFrame(
        {"UserId": [1,
                    2,
                    3,
                    4,
                    5,
                    6, ],
         "Event": ["phone_call",
                   "send_add",
                   "send_sms",
                   "send_email",
                   "send_newsletter",
                   "send_newsletter", ],
         "Category": [
             "Jobs",
             "Phone",
             "Holidays",
             "Motor",
             "Leisure",
             "Holidays",
         ],
         }
    )


@pytest.fixture
def training_labels():
    return pd.Series(
        [0, 0, 0, 0, 0, 1],
        name="label"
    )


def test_model_init():
    model = model_package.ScikitLearnModel()
    assert isinstance(model, Pipeline)
    assert model.steps[0][0] == "encoder"
    assert isinstance(model.steps[0][1], OneHotEncoder)
    assert model.steps[1][0] == "clf"
    assert isinstance(model.steps[1][1], LogisticRegression)
    # Check default parameters
    assert model.steps[-1][1].get_params() == {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                                               'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100,
                                               'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
                                               'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0,
                                               'warm_start': False}
    # Change a parameter
    model = model_package.ScikitLearnModel(penalty="l1")
    assert model.steps[-1][1].get_params() == {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                                               'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100,
                                               'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l1',
                                               'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0,
                                               'warm_start': False}


def test_model_fit(training_data, training_labels):
    model = model_package.ScikitLearnModel()
    model.fit(training_data, training_labels)
    assert model.__sklearn_is_fitted__()


def test_model_predict(training_data, training_labels):
    model = model_package.ScikitLearnModel()
    model.fit(training_data, training_labels)
    y_pred = model.predict(training_data)
    assert_array_equal(y_pred, np.zeros(len(training_labels)))


def test_model_save():
    save_path = "test"
    model = model_package.ScikitLearnModel()
    # Verify joblib is being used to save at the correct path
    with patch("joblib.dump", wraps=lambda self, path, compress: path) as model_save:
        model.save(save_path)
    model_save.assert_called_once_with(model, save_path, compress=3)
