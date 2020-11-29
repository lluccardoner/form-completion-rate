from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame


def transform(df: DataFrame, debug: bool = False) -> DataFrame:
    feature_cols = [c for c in df.columns if "feat_" in c]

    stages = get_stages(feature_cols)
    pipeline = Pipeline()
    pipeline.setStages(stages)
    if debug:
        print(pipeline.explainParams())
    return pipeline.fit(df).transform(df)


def get_stages(feature_cols: list) -> list:
    # TODO log transform
    vec_assembler = get_vector_assembler(feature_cols)
    stages = [
        vec_assembler
    ]
    return stages


def get_vector_assembler(input_cols: list) -> VectorAssembler:
    stage = VectorAssembler(
        inputCols=input_cols,
        outputCol="features")
    return stage
