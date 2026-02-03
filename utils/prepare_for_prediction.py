import numpy as np
import pandas as pd


# Create padded time-series sequences 
def create_sequences(
    df: pd.DataFrame,
    input_scaler,
    numeric_features,
    categorical_features,
    max_len=31,
    pad_value=0
):
    all_X = []
    placement_days = {}

    for placement_id, group in df.groupby("placement_id"):
        group = group.sort_values("age").reset_index(drop=True)
        n_days = len(group)
        placement_days[placement_id] = n_days

        numeric_scaled = input_scaler.transform(
            group[numeric_features]
        )

        categorical_vals = group[categorical_features].values
        if categorical_vals.ndim == 1:
            categorical_vals = categorical_vals.reshape(-1, 1)

        X_full = np.concatenate(
            [numeric_scaled, categorical_vals],
            axis=1
        )

        if n_days < max_len:
            pad = np.full(
                (max_len - n_days, X_full.shape[1]),
                pad_value
            )
            X_full = np.vstack([pad, X_full])

        all_X.append(X_full)

    return np.array(all_X), placement_days


