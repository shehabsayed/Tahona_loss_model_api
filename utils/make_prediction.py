import pandas as pd
from utils.feature_selection import build_features
from utils.imputation import impute_weather
from utils.sequence_creation import create_sequences
from utils.artifacts import model, input_scaler, output_scaler
from utils.settings import numeric_features, categorical_features, MAX_LEN, PAD_VALUE
from utils.prediction_handeling import handle_negative_predictions

def predict_from_raw(df_raw: pd.DataFrame):
    df = build_features(df_raw)
    df = impute_weather(df)
    
    X, masks, info = create_sequences(
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
    return y, info
