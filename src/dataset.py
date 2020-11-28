from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import *

DATASET_SCHEMA = StructType([
                                StructField("form_id", IntegerType(), True),
                                StructField("views", IntegerType(), True),
                                StructField("submissions", IntegerType(), True)
                            ] +
                            [StructField("feat_" + str(i).zfill(2), DoubleType(), True) for i in range(1, 47 + 1)])


def load_dataset(spark: SparkSession, path: str, debug: bool = False) -> DataFrame:
    """
    Load dataset from the given path
    and add the completion rate column
    :param debug: if true, print debug info
    :param spark: SparkSession
    :param path: input path
    :return: pyspark DataFrame
    """

    if debug:
        print("Loading dataset from {}".format(path))

    df = spark.read \
        .option("header", True) \
        .schema(DATASET_SCHEMA) \
        .csv(path)
    df = df.withColumn("CR", df["submissions"] / df["views"])
    return df
