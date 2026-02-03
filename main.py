import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from utils.ingesters.factory import DataIngester
from services.daily_loss_prediction import predict_from_raw

app = FastAPI(title="Chicken Daily Loss Prediction API")


class PredictionRequest(BaseModel):
    data: List[int]


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    df = DataIngester.load_today()
    ids = request.data
    df = df[df['placement_id'].isin(ids)].reset_index(drop=True)

    y_pred, placement_days = predict_from_raw(df)

    results = []
    for i, (pid, n_days) in enumerate(placement_days.items()):
        losses = y_pred[i][:n_days]               # remove padding
        losses = np.round(losses).astype(int).tolist()

        results.append({
            "placement_id": int(pid),
            "loss": losses
        })

    return results

