import unittest

from pyspark.sql import Row
from pyspark.sql import SparkSession

import processing
from utils import are_dfs_equal


class ProcessingTest(unittest.TestCase):
    spark = None

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession \
            .builder \
            .appName("form_complation_rate") \
            .getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_transform_should_return_dataframe(self):
        row = Row("CR", "views", "feat_01", "feat_02")
        data = [
            row(0.5, 100, 0.1, 125.0),
            row(0.5, 100, 0.1, 125.0)
        ]
        in_df = self.spark.createDataFrame(data)

        out_df = processing.transform(in_df)

        self.assertTrue(are_dfs_equal(in_df, out_df))
