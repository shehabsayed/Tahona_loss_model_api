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

def create_sequences_for_training(df, input_scaler, output_scaler,
                     numeric_features, categorical_features,
                     target, max_len=31, pad_value=0):

    all_X, all_y, all_masks = [], [], []

    for placement_id, group in df.groupby("placement_id"):
        group = group.sort_values('age').reset_index(drop=True)
        n_days = len(group)

        # Scale features
        numeric_scaled = input_scaler.transform(group[numeric_features])
        categorical_vals = group[categorical_features].values
        if categorical_vals.ndim == 1:
            categorical_vals = categorical_vals.reshape(-1, 1)

        X_full = np.concatenate([numeric_scaled, categorical_vals], axis=1)
        y_full = output_scaler.transform(group[[target]]).flatten()

        for i in range(n_days):
            seq_len = i + 1
            X_seq = X_full[:seq_len]

            # Pad input sequence
            if seq_len < max_len:
                pad = np.full((max_len - seq_len, X_full.shape[1]), pad_value)
                X_seq = np.vstack([pad, X_seq])

            # Mask: 1 for valid steps, 0 for padded steps
            mask = np.zeros(max_len)
            mask[max_len - seq_len:] = 1

            all_X.append(X_seq)
            all_y.append(y_full)
            all_masks.append(mask)

    return np.array(all_X), np.array(all_y), np.array(all_masks)

def shuffle_sequences(X, y, masks, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx], masks[idx]