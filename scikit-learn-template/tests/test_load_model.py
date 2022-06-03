"""
This module tests the loading of a model and that all warnings and exceptions are raised.
"""
from unittest.mock import MagicMock, patch
import joblib
import pytest
from model_package.load_model import \
    (
    load_model,
    UnknownModelLoadingError,
    IncompatibleVersionError,
    raise_or_warn
)
from model_package import __version__
from sklearn import __version__ as sklearn_version


class MockedModel:
    def __init__(self, wrong_version):
        if wrong_version == "scikit-learn":
            self.versions = {
                "model_package": __version__,
                "scikit-learn": -1
            }
        elif wrong_version == "model_package":
            self.versions = {
                "model_package": -1,
                "scikit-learn": sklearn_version
            }

        elif wrong_version == "all":
            self.versions = {
                "model_package": -1,
                "scikit-learn": -1
            }
        else:
            self.versions = {
                "model_package": __version__,
                "scikit-learn": sklearn_version
            }


def test_raise_or_warn():
    message = "test"
    with pytest.raises(IncompatibleVersionError) as e:
        raise_or_warn(message=message, on_wrong_version="raise")
    assert format(e.value) == message
    with pytest.warns(match=message):
        raise_or_warn(message=message, on_wrong_version="ignore")


def test_load_model():
    save_path = "loading_is_mocked"

    # Functions that are only needed in this scope
    def raise_warn_test(err_message):
        with pytest.raises(IncompatibleVersionError) as e:
            load_model(path=save_path, on_wrong_version="raise")
        assert format(e.value) == err_message
        with pytest.warns(match=err_message):
            raise_or_warn(message=err_message, on_wrong_version="ignore")

    def raise_path(path):
        raise Exception(path)

    # Check it works
    joblib.load = MagicMock(return_value=MockedModel(None))
    loaded_model = load_model(path=save_path, on_wrong_version="raise")
    assert isinstance(loaded_model, MockedModel)
    # Check failures
    joblib.load = MagicMock(return_value=MockedModel("all"))
    # assert False
    message = "Neither the model_package or sklearn versions are the same as training.\n" \
              f"model_package during training: -1. Now: {__version__}" \
              f"scikit-learn during training: -1. Now: {sklearn_version}"
    raise_warn_test(message)
    joblib.load = MagicMock(return_value=MockedModel("scikit-learn"))
    message = f"Version missmatch detected. \nscikit-learn during training: -1. " \
              f"Now: {sklearn_version}"
    raise_warn_test(message)
    joblib.load = MagicMock(return_value=MockedModel("model_package"))
    message = f"Version missmatch detected. \nmodel_package during training: -1. " \
              f"Now: {__version__}"
    raise_warn_test(message)
    # Check argument assertion error is raised
    with pytest.raises(AssertionError) as e:
        load_model(path=save_path, on_wrong_version="raise2")
    assert format(e.value) == f"on_wrong_version must be either 'raise' or 'ignore', received raise2"

    # Test UnknownModelLoadingError and that the correct path is being fed to joblib
    # at the same time by raising an exception that will trigger UnknownModelLoadingError with
    # the path in the error message
    with patch("joblib.load", raise_path):
        with pytest.raises(UnknownModelLoadingError) as e:
            load_model(path=save_path, on_wrong_version="raise")
    assert format(e.value) == "Unexpected error when loading the model: loading_is_mocked"
