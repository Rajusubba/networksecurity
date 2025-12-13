import os
from pathlib import Path

import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

PREPROCESSOR_PATH = BASE_DIR / "final_models" / "preprocessor.pkl"
MODEL_PATH = BASE_DIR / "final_models" / "model.pkl"
PREDICTION_OUT_DIR = BASE_DIR / "prediction_output"
PREDICTION_OUT_DIR.mkdir(exist_ok=True)

# ---------------- MongoDB ----------------
ca = certifi.where()
load_dotenv()

mongo_db_url = os.getenv("MONGO_URL_KEY")
if not mongo_db_url:
    logging.warning("MONGO_URL_KEY is not set. Mongo features may fail.")

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca) if mongo_db_url else None
collection = None
if client:
    database = client[DATA_INGESTION_DATABASE_NAME]
    collection = database[DATA_INGESTION_COLLECTION_NAME]

# ---------------- FastAPI ----------------
app = FastAPI(title="Network Security ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/train")
async def train_route():
    """
    IMPORTANT: Import TrainingPipeline *inside* the route so the server can start
    in Docker/EC2 without triggering DagsHub OAuth / training imports on startup.
    """
    try:
        from networksecurity.pipeline.training_pipeline import TrainingPipeline  # lazy import
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return {"message": "Training successful!!"}
    except Exception as e:
        logging.exception("Error while running training pipeline")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Missing preprocessor: {PREPROCESSOR_PATH}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
        if not TEMPLATES_DIR.exists():
            raise FileNotFoundError(f"Missing templates folder: {TEMPLATES_DIR}")
        if not (TEMPLATES_DIR / "table.html").exists():
            raise FileNotFoundError(f"Missing template: {TEMPLATES_DIR / 'table.html'}")

        df = pd.read_csv(file.file)

        preprocessor = load_object(str(PREPROCESSOR_PATH))
        final_model = load_object(str(MODEL_PATH))
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        y_pred = network_model.predict(df)
        df["predicted_label"] = y_pred

        out_path = PREDICTION_OUT_DIR / "output.csv"
        df.to_csv(out_path, index=False)

        table_html = df.to_html(classes="table table-striped", index=False)
        return templates.TemplateResponse(
            "table.html",
            {"request": request, "table_html": table_html, "output_path": str(out_path)},
        )

    except Exception as e:
        logging.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
    