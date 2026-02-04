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

import pandas as pd
from utils.settings import FEATURES
from utils.feature_selection import build_features
from utils.imputation import impute_weather
from utils.prepare_for_prediction import create_sequences
from assets.model_scaler_loader import model, input_scaler, output_scaler
from utils.settings import numeric_features, categorical_features, MAX_LEN, PAD_VALUE
from utils.prediction_handeling import handle_negative_predictions


def predict_from_raw(df_raw: pd.DataFrame):
    df = build_features(df_raw, FEATURES)
    df = impute_weather(df)

    X, placement_days = create_sequences(
        df,
        input_scaler,
        numeric_features,
        categorical_features,
        max_len=MAX_LEN,
        pad_value=PAD_VALUE
    )

    y_scaled = model.predict(X)
    y = output_scaler.inverse_transform(y_scaled)
    y = handle_negative_predictions(y)

    return y, placement_days

