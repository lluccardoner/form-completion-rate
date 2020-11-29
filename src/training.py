from pyspark.ml import Model
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame


def fit(df: DataFrame, params: dict, debug: bool = False) -> Model:
    lr = get_stage(params)
    if debug:
        print(lr.explainParams())
    return lr.fit(df)


def get_stage(params):
    lr = LinearRegression()
    lr.setParams(**params)
    return lr
