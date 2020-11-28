from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame


def transform(df: DataFrame, debug: bool = False) -> DataFrame:
    feature_cols = [c for c in df.columns if "feat_" in c]

    vec_assembler = get_vector_assembler(feature_cols)
    stages = [
        vec_assembler
    ]

    pipeline = Pipeline()
    pipeline.setStages(stages)
    if debug:
        print(pipeline.explainParams())
    return pipeline.fit(df).transform(df)


def get_vector_assembler(input_cols: list) -> VectorAssembler:
    stage = VectorAssembler(
        inputCols=input_cols,
        outputCol="features")
    return stage
