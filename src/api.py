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
    .appName("form_completion_rate_api") \
    .getOrCreate()

model = PipelineModel.load(str(DEPLOY_DIR / "latest"))


@app.get("/")
def home():
    return "OK"


@app.post("/predict")
async def predict(sample: Sample):
    sample_df = spark.createDataFrame([dict(sample)], schema=INPUT_SCHEMA)
    prediction_df = model.transform(sample_df)
    prediction = prediction_df.collect()[0]["prediction"]
    return {"Prediction": prediction}

if __name__ == "__main__":
    uvicorn.run("api:app", reload=True)
