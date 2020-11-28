from pyspark.ml import Model
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame


def fit(df: DataFrame, params: dict, debug: bool = False) -> Model:
    lr = LinearRegression()
    lr.setParams(**params)
    if debug:
        print(lr.explainParams())
    return lr.fit(df)
