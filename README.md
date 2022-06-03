# Scikit-learn package template

This repository aims at providing a package to train a scikit-learn model (pipeline) and exploit it for prediction. For this purpose,
it contains:
- A machine learning package: scikit-learn-template/model_package
- A CLI to train it from the command line
- Example scripts to train the model and generate predictions.

## Installation
This project requires python 3.9.

To install required packages, go to scikit-learn-template and run:
```shell
poetry install
```

## Usage

For all the following, make sure you are in the model_package directory.

### Training

#### Demo script

Run the following in a terminal:

```shell
./demo-train.sh  
```

#### CLI
Input arguments:
```text
--train             Path to the csv file for training
--label_column      Name of the label column. Default is 'label'.
--user_id_column    Name of the user id column. Default is 'UserId'. This column will not be used in training.
--model_path        Path to store the model or to load it from.
--evaluation_folds  How many folds to use to evaluate. If not provided, no evaluation is performed.
```

Example:
```shell
python scikit-learn-template --train your_file.csv --model_path saved_model.joblib --evaluation_folds 4
```
### Predictions

#### Demo script

Run the following in a terminal:

```shell
./demo-predict.sh  
```

#### CLI
Input arguments:
```text
--predict           Path to the csv file for prediction.
--user_id_column    Name of the user id column. Default is 'UserId'. This column will not be used in training.
--model_path        Path to store the model or to load it from.
```
Note that you can, with one single command, train and generate predictions.

Example:
```shell
python scikit-learn-template --train your_file.csv --predict your_file_no_label.csv --predictions_path predictions.csv --evaluation_folds 4
```

## Code structure
```
sklearn-package-template
│   README.md: this file documenting the project
│   demo-train.sh: runable demo shell script showing how to use the cli to train and save a model. 
│   demo-predict.sh: demo shell script to generate predictions 
│
└───scikit-learn-template: contains the model package and the cli
│   │   __main__.py: command line interface (CLI)
│   │   api_exceptions.py: error classes for the CLI
│   │   poetry.lock: poetry configuration for packages
│   │   pyproject.toml: poetry configuration file
│   │
│   └───model_package: model package
│   │   │   __init__.py: exposes main classes, methods and variables
│   │   │   load_model.py: contains a function to reload saved models
│   │   │   model.py: model definition
│   │   │   version.py: contains the __version__ variable.
│   │
│   └───tests: tests for the detector package (mirrors its structure)
│   │   │   test_load_model.py: tests model loading
│   │   │   test_model.py: tests model building
```

## Running the tests
Coverage for the model_package is 100% files and 100% lines. However, the API is untested.

To run the tests, go to the scikit-learn-template directory and run:
```shell
export PYTHONPATH=$(pwd):${PYTHONPATH}  # Make sure it finds the package
pytest tests
```
