import unittest

from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.sql import SparkSession

import training


class TrainingTest(unittest.TestCase):
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

    def test_fit(self):
        in_data = [
            Row(CR=1.0, features=Vectors.dense([0.2, 125.0])),
            Row(CR=0.5, features=Vectors.dense([0.1, 150.0])),
            Row(CR=0.0, features=Vectors.dense([0.0, 175.0]))
        ]
        in_df = self.spark.createDataFrame(in_data)

        params = {
            "featuresCol": "features",
            "labelCol": "CR"
        }

        model = training.fit(in_df, params)

        self.assertTrue(model.getParam("featuresCol"), "features")
        self.assertTrue(model.getParam("labelCol"), "CR")
