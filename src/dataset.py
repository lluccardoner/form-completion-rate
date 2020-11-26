from pyspark.sql import DataFrame
from pyspark.sql import SparkSession


def load_dataset(spark: SparkSession, path: str) -> DataFrame:
    """
    Load dataset from the given path
    and add the completion rate column
    :param spark: SparkSession
    :param path: input path
    :return: pyspark DataFrame
    """

    df = spark.read.option("header", True).csv(path)
    df = df.withColumn("CR", df["submissions"] / df["views"])
    return df
