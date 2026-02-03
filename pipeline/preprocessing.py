from utils.sequence_creation import create_sequences_for_training, shuffle_sequences
from utils.feature_selection import build_features
from utils.imputation import impute_weather
from utils.settings import numeric_features, categorical_features, TARGET, MAX_LEN, PAD_VALUE

def process(df, input_scaler, output_scaler):
    df = build_features(df)
    df = impute_weather(df)

    X, y, mask = create_sequences_for_training(
    df, input_scaler, output_scaler, numeric_features, categorical_features, TARGET, MAX_LEN, PAD_VALUE
    )

    X, y, mask = shuffle_sequences(X, y, mask, seed=42)

    return X, y, mask