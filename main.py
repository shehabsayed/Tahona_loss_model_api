from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from utils.data_loader import load_today_data
from utils.ingesters.factory import DataIngester
from utils.make_prediction import predict_from_raw

app = FastAPI(title="Chicken Loss Prediction API")


# Generic input schema (flexible for your many columns)
class PredictionRequest(BaseModel):
    data: List[int]


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert JSON → DataFrame
    df = DataIngester.load_today()
    ids = request.data
    df = df[df['placement_id'].isin(ids)].reset_index(drop=True)

    # Run full ML pipeline
    y_pred, info = predict_from_raw(df)

    result = {}
    for i, meta in enumerate(info):
        pid = int(meta["placement_id"])
        day = meta["day"]

        if pid not in result:
            result[pid] = {}

        result[pid][f"day_{day}"] = y_pred[i].tolist()

    # Convert to JSON serializable
    return {
        "n_predictions": len(info),
        "predictions": result
    }

