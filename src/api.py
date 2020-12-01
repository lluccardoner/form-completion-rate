import time

import uvicorn
from fastapi import FastAPI
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType

from utils import DEPLOY_DIR
from utils import Sample

INPUT_SCHEMA = StructType([StructField("feat_" + str(i).zfill(2), DoubleType(), True) for i in range(1, 47 + 1)])

app = FastAPI()

spark = SparkSession \
    .builder \
    .master("local") \
    .appName("form_completion_rate_api") \
    .getOrCreate()

model = PipelineModel.load(str(DEPLOY_DIR / "latest"))


@app.get("/")
def home():
    return "OK"


@app.post("/predict")
def predict(sample: Sample):
    if model is None:
        return {"Error": "No model loaded yet"}
    sample_df = spark.createDataFrame([dict(sample)], schema=INPUT_SCHEMA)
    prediction_df = model.transform(sample_df)
    prediction = prediction_df.collect()[0]["prediction"]
    return {"Prediction": prediction}


def load_model():
    print("Reloading model {}".format(time.ctime()))
    try:
        global model
        model = PipelineModel.load(str(DEPLOY_DIR / "latest"))
    except:
        print("Could not load model")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
