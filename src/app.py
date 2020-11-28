import json

from pyspark.sql import SparkSession

import arg_parser
import dataset
import evaluation
import model_selection
import processing
from utils import OUTPUT_DIR

if __name__ == "__main__":
    """
    App that performs the following tasks:
    - Load a dataset
    - Process the dataset
    - Train a model
    - Evaluate the model
    """
    spark = SparkSession \
        .builder \
        .appName("form_completion_rate") \
        .getOrCreate()

    args = arg_parser.get_args()

    if args.debug:
        print("Using debug mode")

    df = dataset.load_dataset(spark, args.datasetPath, args.debug)

    df_processed = processing.transform(df, args.debug)

    train_df, test_df = df_processed.randomSplit([0.8, 0.2], seed=42)

    cv = model_selection.cross_validation(train_df, args.debug)

    model_path = OUTPUT_DIR.joinpath("best_model")
    print("Saving best model at {}".format(model_path))
    cv.bestModel.save(str(model_path))

    metrics = evaluation.cv_metrics(cv, test_df)

    metrics_path = OUTPUT_DIR.joinpath("metrics.json")
    with open(metrics_path, 'w') as f:
        print("Saving metrics at {}".format(metrics_path))
        json.dump(metrics, f, indent=4, sort_keys=True)
