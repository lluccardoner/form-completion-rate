from pyspark.sql import SparkSession

import arg_parser
import dataset
import processing
import training

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

    train_params = {
        "labelCol": "CR",
        "regParam": 0.3,
        "elasticNetParam": 0.1
    }
    model = training.fit(df_processed, train_params, args.debug)

    training_summary = model.summary

    print("Total iterations: {}".format(training_summary.totalIterations))
    print("RMSE: %f" % training_summary.rootMeanSquaredError)
    print("r2: %f" % training_summary.r2)
