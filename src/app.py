from pyspark.sql import SparkSession

import arg_parser
import dataset
import model_selection
import processing

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

    model = cv.bestModel

    if model.hasSummary:
        training_summary = model.summary

        print("Total iterations: {}".format(training_summary.totalIterations))
        print("Train RMSE: %f" % training_summary.rootMeanSquaredError)
        print("Train r2: %f" % training_summary.r2)

    test_summary = model.evaluate(test_df)

    print("Test RMSE: %f" % test_summary.rootMeanSquaredError)
    print("Test r2: %f" % test_summary.r2)
