import unittest

from pyspark.sql import SparkSession

import dataset


class DatasetTest(unittest.TestCase):
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

    def test_load_dataset_should_return_dataframe(self):
        test_path = "resources/test_dataset.csv"
        num_features = 47
        main_columns = ["form_id", "submissions", "views", "CR"]
        feature_columns = ["feat_" + str(i).zfill(2) for i in range(1, num_features + 1)]

        df = dataset.load_dataset(self.spark, test_path)

        self.assertCountEqual(df.columns, main_columns + feature_columns)
