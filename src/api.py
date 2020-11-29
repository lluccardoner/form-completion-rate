import uvicorn
from fastapi import FastAPI
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegressionModel
from pyspark.sql import SparkSession

from utils import MODEL_DIR
from utils import Sample

app = FastAPI()

spark = SparkSession \
    .builder \
    .appName("form_completion_rate_api") \
    .getOrCreate()

model = LinearRegressionModel.load(str(MODEL_DIR))


@app.get("/")
def home():
    return "OK"


@app.post("/predict")
async def predict(sample: Sample):
    print(dict(sample))
    sample = Vectors.dense([i for i in range(47)])

    prediction = model.predict(sample)
    return {"Prediction": prediction}


if __name__ == "__main__":
    uvicorn.run("api:app")
