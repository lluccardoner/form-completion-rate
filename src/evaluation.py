import json

from pyspark.ml import Model
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql import DataFrame


def cv_metrics(model: CrossValidatorModel, output_path=None) -> dict:
    num_folds = model.getNumFolds()
    evaluator_metric = model.getEvaluator().getMetricName()

    param_maps = model.getEstimatorParamMaps()
    cv_avg_metrics = model.avgMetrics

    grid = []
    for p, m in zip(param_maps, cv_avg_metrics):
        grid_item = {
            "params": dict([(str(param), value) for param, value in p.items()]),
            evaluator_metric: m
        }
        grid.append(grid_item)

    metrics = {
        "cross_validation_metrics": {
            "num_folds": num_folds,
            "evaluator_metric": evaluator_metric,
            "grid_search": grid
        }
    }

    if output_path is not None:
        save_metrics(metrics, output_path)

    return metrics


def model_metrics(model: Model, test_df: DataFrame, output_path=None) -> dict:
    metrics = {
        "model_params": dict([(str(param), value) for param, value in model.extractParamMap().items()])
    }

    if model.hasSummary:
        training_summary = model.summary

        metrics["training_summary"] = {
            "totalIterations": training_summary.totalIterations,
            "rmse": training_summary.rootMeanSquaredError,
            "r2": training_summary.r2
        }

    test_summary = model.evaluate(test_df)

    metrics["test_summary"] = {
        "rmse": test_summary.rootMeanSquaredError,
        "r2": test_summary.r2
    }

    if output_path is not None:
        save_metrics(metrics, output_path)

    return metrics


def save_metrics(metrics, path):
    with open(path, 'w') as f:
        print("Saving metrics at {}".format(path))
        json.dump(metrics, f, indent=4, sort_keys=True)
