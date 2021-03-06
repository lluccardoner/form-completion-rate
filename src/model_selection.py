import time

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql import DataFrame


def cross_validation(df: DataFrame, debug: bool = False) -> CrossValidatorModel:
    cv = get_stage()
    if debug:
        print(cv.explainParams())

    print("Start grid-search cross-validation model selection...")
    start_time = time.time()
    model = cv.fit(df)
    duration = time.time() - start_time
    print("Finish grid-search cross-validation model selection in {:.2f} seconds".format(duration))
    return model


def get_stage() -> CrossValidator:
    lr = LinearRegression()
    # TODO value distributions instead of discrete
    grid = ParamGridBuilder() \
        .addGrid(lr.maxIter, [100, 200]) \
        .addGrid(lr.regParam, [0.1, 0.2]) \
        .addGrid(lr.elasticNetParam, [0.1, 0.2]) \
        .build()
    evaluator = RegressionEvaluator(
        metricName="mae"
    )
    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        seed=42
    )
    return cv
