import unittest

from pyspark.ml.linalg import Vectors
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
            .appName("form_completion_rate") \
            .getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_transform_should_return_dataframe(self):
        in_data = [
            Row(CR=0.5, views=100, feat_01=0.1, feat_02=125.0),
            Row(CR=0.5, views=100, feat_01=0.1, feat_02=125.0)
        ]
        in_df = self.spark.createDataFrame(in_data)

        expected_data = [
            Row(CR=0.5, views=100, feat_01=0.1, feat_02=125.0, features=Vectors.dense([0.1, 125.0])),
            Row(CR=0.5, views=100, feat_01=0.1, feat_02=125.0, features=Vectors.dense([0.1, 125.0]))
        ]
        expected_df = self.spark.createDataFrame(expected_data)

        out_df = processing.transform(in_df)

        self.assertTrue(are_dfs_equal(expected_df, out_df))
