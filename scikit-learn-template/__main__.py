"""
CLI module. This will:
- read data
- fit a model if asked
- evaluate the model using cross-validation if asked
- save the model if asked
- generate predictions if asked

See readme for how to use it.
"""
import argparse
import pandas as pd
from model_package import ScikitLearnModel, load_model
import api_exceptions
from sklearn.model_selection import cross_val_score

parser = argparse.ArgumentParser(description="Train or use a scikit-learn model.")
parser.add_argument("--train",
                    help="Path to the csv file for training")
parser.add_argument("--label_column",
                    help="Name of the label column. Default is 'label'.",
                    default="label")
parser.add_argument("--user_id_column",
                    help="Name of the user id column. Default is 'UserId'. This column will not be used in training.",
                    default="UserId")
parser.add_argument("--model_path",
                    help="Path to store the model or to load it from.")
parser.add_argument("--predict",
                    help="Path to the csv file for prediction.")
parser.add_argument("--predictions_path",
                    help="Where to store the predictions once they have been generated.")
parser.add_argument("--evaluation_folds",
                    type=int,
                    help="How many folds to use to evaluate. If not provided, no evaluation is performed.")


def read_data(path: str, index_col: str, phase: str):
    try:
        data = pd.read_csv(path, index_col=index_col)
        print("data read")
    except pd.errors.ParserError as e:
        raise api_exceptions.UnsupportedFileFormat(f"There was an issue when reading {phase} data. Make sure "
                                                   f"the file is a csv. Error: {e}")
    return data


def check_arguments(train: str, predict: str, model_path: str, predictions_path: str):
    if train is None and predict is None:
        raise api_exceptions.MissingArgument("None of --training or --predict has been provided. Provide "
                                             "at least one.")
    if predict is not None and predictions_path is None:
        raise api_exceptions.MissingArgument("To generate predictions, please provide a path for where to write "
                                             "the output csv (usage: --predictions_path mypath.csv).")
    if train is None and predict is not None and model_path is None:
        raise api_exceptions.MissingArgument("To generate predictions without training a model, please provide "
                                             "the path to a trained model (usage: --model_path mypath).")


if __name__ == "__main__":
    args = parser.parse_args()
    check_arguments(train=args.train, model_path=args.model_path, predictions_path=args.predictions_path,
                    predict=args.predict)

    if args.train is not None:
        training_data = read_data(args.train, args.user_id_column, "training")
        if args.label_column not in training_data.columns:
            raise api_exceptions.LabelColumnNotFound(f"Asked to train on label column {args.label_column} "
                                                     f"but it was not found in the training data. Found "
                                                     f"columns {training_data.columns}")
        model = ScikitLearnModel()
        if args.evaluation_folds is not None:
            print(f"Model accuracy on {args.evaluation_folds} folds:",
                  cross_val_score(model,
                                  training_data.drop(columns=[args.label_column]),
                                  training_data.loc[:, args.label_column],
                                  cv=args.evaluation_folds),
                  )
        print(f"Fitting model on {len(training_data)} rows...")
        model.fit(training_data.drop(columns=[args.label_column]), training_data.loc[:, args.label_column])
        print("Model was fitted.")
        if args.model_path is not None:
            print(f"saving model at {args.model_path}...")
            model.save(args.model_path)
            print("model was saved.")

    if args.predict is not None:
        prediction_data = read_data(args.predict, args.user_id_column, "predict")
        if args.train is None:  # Model will not be undefined because of the check_arguments function call
            print(f"Loading model from {args.model_path}...")
            model = load_model(args.model_path)
            print("Loaded.")
        print("Generating predictions...")
        predictions = pd.Series(
            model.predict(prediction_data),
            index=prediction_data.index,
            name=args.label_column
        )
        print(f"Saving_predictions at {args.predictions_path}...")
        predictions.to_csv(args.predictions_path)
        print("Saved.")
