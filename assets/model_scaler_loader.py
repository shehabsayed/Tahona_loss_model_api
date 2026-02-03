# assets/loader.py
import joblib
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model
from utils.custom_layers import SqueezeLastDim, VectorQuantizer
from tcn import TCN

BASE = Path(__file__).parent

model = load_model(
    BASE / "model.keras",
    custom_objects={
        "SqueezeLastDim": SqueezeLastDim,
        "VectorQuantizer": VectorQuantizer,
        'TCN':TCN
    }
)

input_scaler = joblib.load(BASE / "input_scaler.joblib")
output_scaler = joblib.load(BASE / "output_scaler.joblib")
