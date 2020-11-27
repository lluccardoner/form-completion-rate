from pyspark.sql import SparkSession

import arg_parser
import dataset
import processing

if __name__ == "__main__":
    """
    App that performs the following tasks:
    - Load a dataset
    - Process the dataset
    - Train a model
    - Do predictions
    """
    spark = SparkSession \
        .builder \
        .appName("form_complation_rate") \
        .getOrCreate()

    args = arg_parser.get_args()

    if args.debug:
        print("Using debug mode")

    df = dataset.load_dataset(spark, args.datasetPath, args.debug)

    df_processed = processing.transform(df, args.debug)

    df_processed.show()
