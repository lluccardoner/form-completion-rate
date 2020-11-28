import time

from pyspark.ml import Model
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame


def cross_validation(df: DataFrame, debug: bool = False) -> Model:
    lr = LinearRegression()
    grid = ParamGridBuilder() \
        .addGrid(lr.maxIter, [100, 200]) \
        .addGrid(lr.regParam, [0.1, 0.2]) \
        .addGrid(lr.elasticNetParam, [0.1, 0.2]) \
        .build()
    evaluator = RegressionEvaluator(
        metricName="rmse"
    )
    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=grid,
        evaluator=evaluator
    )
    if debug:
        print(cv.explainParams())

    print("Start cross-validation model selection...")
    start_time = time.time()
    model = cv.fit(df)
    duration = time.time() - start_time
    print("Finish cross-validation model selection in {:.2f} seconds".format(duration))
    return model
