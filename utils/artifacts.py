import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.custom_layers import SqueezeLastDim, VectorQuantizer
from tcn import TCN

# Paths
MODEL_PATH = r"E:\Work\Notebooks\Tahoona\Final_Model\lstm_tcn_vq_model.keras"
INPUT_SCALER_PATH = r"E:\Work\Notebooks\Tahoona\Final_Model\input_scaler.joblib"
OUTPUT_SCALER_PATH = r"E:\Work\Notebooks\Tahoona\Final_Model\output_scaler.joblib"

# Load once (on app startup)
model = load_model(
    MODEL_PATH,
    custom_objects={
        "SqueezeLastDim": SqueezeLastDim,
        "VectorQuantizer": VectorQuantizer,
        'TCN':TCN
    }
)

input_scaler = joblib.load(INPUT_SCALER_PATH)
output_scaler = joblib.load(OUTPUT_SCALER_PATH)
