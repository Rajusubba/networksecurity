
import os
import sys
import pymongo
import certifi
import pandas as pd
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
# ... other imports ...

BASE_DIR = Path(__file__).resolve().parent

# If table.html is in the project root:
templates = Jinja2Templates(directory="templates")

# If you move table.html into a "templates" folder instead, use:
# templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
PREPROCESSOR_PATH = BASE_DIR / "final_models" / "preprocessor.pkl"
MODEL_PATH = BASE_DIR / "final_models" / "model.pkl"
# ---------- MongoDB setup ----------
ca = certifi.where()
load_dotenv()

mongo_db_url = os.getenv("MONGO_URL_KEY")
print(mongo_db_url)

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# ---------- FastAPI app ----------
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return {"message": "Training successful!!"}
    except Exception as e:
        logging.exception("Error while running training pipeline")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Prediction endpoint ----------
# Upload a CSV file, run inference, save output.csv, return a simple JSON summary

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # make sure we use the full paths
        preprocessor = load_object(str(PREPROCESSOR_PATH))
        final_model = load_object(str(MODEL_PATH))

        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)

        df["predicted_label"] = y_pred
        df.to_csv("prediction_output/output.csv", index=False)

        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse("table.html", {"request": request, "table_html": table_html})

    except Exception as e:
        logging.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    from uvicorn import run as app_run

    app_run(app, host="localhost", port=8000)