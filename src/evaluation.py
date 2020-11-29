import json
from pathlib import Path

from pyspark.ml import Model
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql import DataFrame


def cv_metrics(model: CrossValidatorModel, output_dir: Path = None) -> dict:
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

    if output_dir is not None:
        save_metrics(metrics, output_dir / "cv_metrics.json")

    return metrics


def model_metrics(model: Model, test_df: DataFrame, output_dir: Path = None, residuals=False) -> dict:
    metrics = {
        "model_params": dict([(str(param), value) for param, value in model.extractParamMap().items()])
    }

    if model.hasSummary:
        training_summary = model.summary

        metrics["training_summary"] = {
            "totalIterations": training_summary.totalIterations,
            "rmse": training_summary.rootMeanSquaredError,
            "mae": training_summary.meanAbsoluteError,
            "r2": training_summary.r2
        }

    test_summary = model.evaluate(test_df)

    metrics["test_summary"] = {
        "rmse": test_summary.rootMeanSquaredError,
        "mae": test_summary.meanAbsoluteError,
        "r2": test_summary.r2
    }

    if output_dir is not None:
        save_metrics(metrics, output_dir / "model_metrics.json")
        if residuals:
            save_residuals_plot(model, test_df, output_dir)

    return metrics


def save_metrics(metrics, path):
    with open(path, 'w') as f:
        print("Saving metrics at {}".format(path))
        json.dump(metrics, f, indent=4, sort_keys=True)


def save_residuals_plot(model: Model, test_df: DataFrame, output_dir: Path):
    predictions = model.transform(test_df.limit(2000))
    residuals = predictions.withColumn("residual", predictions["label"] - predictions["prediction"])
    data = residuals.select("prediction", "residual").toPandas()
    ax = data.plot.scatter(x="prediction", y="residual")
    ax.axhline(y=0, linestyle='--', color='black')
    fig = ax.get_figure()
    fig.savefig(str(output_dir / "residuals-plot.png"))
