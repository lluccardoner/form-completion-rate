import time

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

import arg_parser
import dataset
import model_selection
import processing
from utils import DEPLOY_DIR

if __name__ == "__main__":
    """
    Train and deploy model
    """

    spark = SparkSession \
        .builder \
        .appName("form_completion_rate") \
        .getOrCreate()

    args = arg_parser.get_args()

    if args.debug:
        print("Using debug mode")

    df = dataset.load_dataset(spark, args.datasetPath, args.debug)

    feature_cols = [c for c in df.columns if "feat_" in c]

    processing_stages = processing.get_stages(feature_cols)
    train_stage = model_selection.get_stage()

    stages = processing_stages + [train_stage]
    pipeline = Pipeline().setStages(stages)

    print("Start training model...")
    start_time = time.time()
    model = pipeline.fit(df)
    duration = time.time() - start_time
    print("Finish training in {:.2f} seconds".format(duration))

    print("Deploying model at {}".format(DEPLOY_DIR))
    model.write().overwrite().save(str(DEPLOY_DIR))
