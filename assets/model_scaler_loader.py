import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '-1'

import random
import tensorflow as tf
import numpy as np
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


import joblib
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
