from pyspark.sql import SparkSession

import dataset

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("form_complation_rate") \
        .getOrCreate()

    df = dataset.load_dataset("")
