from pyspark.ml import Pipeline
from pyspark.sql import DataFrame


def transform(df: DataFrame, debug: bool = False) -> DataFrame:
    stages = []
    pipeline = Pipeline()
    pipeline.setStages(stages)
    if debug:
        print(pipeline.explainParams())
    return pipeline.fit(df).transform(df)
