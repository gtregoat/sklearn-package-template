"""
This module provides a load_model function to reload the model and raise a warning or an exception
when the library versions are different (inconsistent versions may lead to an undesired output).

Usage:
load_model(path, on_wrong_version="raise")
"""
import joblib
from .version import __version__
from sklearn import __version__ as sklearn_version
import warnings
from .model import ScikitLearnModel


class UnknownModelLoadingError(Exception):
    pass


class IncompatibleVersionError(Exception):
    pass


def raise_or_warn(message: str, on_wrong_version: str) -> None:
    """
    Either raises a warning or an IncompatibleVersionError with the provided message.

    :param message: message to display.
    :param on_wrong_version: "raise" or "ignore". "raise" will stop the code while "ignore" raises a warning.
    """
    if on_wrong_version == "raise":
        raise IncompatibleVersionError(message)
    else:
        warnings.warn(message)


def load_model(path: str, on_wrong_version: str = "raise") -> ScikitLearnModel:
    """
    Loads a previously trained model. When scikit-learn or the model_package package current versions do not
    match training time versions, the loading will either stop with an exception or continue with a warning
    depending on the "on_wrong_action" parameter.

    :param path: path to the trained model.
    :param on_wrong_version: "raise" or "ignore". "raise" will stop the code while "ignore" raises a warning.
    :return: trained ScikitLearnModel model.
    """
    assert on_wrong_version in (
        "raise", "ignore"), f"on_wrong_version must be either 'raise' or 'ignore', received {on_wrong_version}"
    try:
        model = joblib.load(path)
    except Exception as e:
        raise UnknownModelLoadingError(f"Unexpected error when loading the model: {e}")
    training_detector_version = model.versions["model_package"]
    training_sklearn_version = model.versions["scikit-learn"]
    detector_version_ok = training_detector_version == __version__
    sklearn_version_ok = training_sklearn_version == sklearn_version
    if not detector_version_ok and not sklearn_version_ok:
        message = "Neither the model_package or sklearn versions are the same as training.\n" \
                  f"model_package during training: {training_detector_version}. Now: {__version__}" \
                  f"scikit-learn during training: {training_sklearn_version}. Now: {sklearn_version}"
        raise_or_warn(message, on_wrong_version)
    elif not detector_version_ok:
        message = f"Version missmatch detected. \nmodel_package during training: {training_detector_version}. " \
                  f"Now: {__version__}"
        raise_or_warn(message, on_wrong_version)
    elif not sklearn_version_ok:
        message = f"Version missmatch detected. \nscikit-learn during training: {training_sklearn_version}. " \
                  f"Now: {sklearn_version}"
        raise_or_warn(message, on_wrong_version)
    return model
