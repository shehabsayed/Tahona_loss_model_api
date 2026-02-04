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

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from utils.sequence_creation import create_sequences_for_training, shuffle_sequences
from utils.settings import numeric_features, categorical_features, TARGET, MAX_LEN, PAD_VALUE
import pandas as pd


def remove_small_placements(df: pd.DataFrame) -> pd.DataFrame:
    placement_counts = df['placement_id'].value_counts()
    valid_placements = placement_counts[placement_counts >= 29].index
    df_filtered = df[df['placement_id'].isin(valid_placements)].copy()
    return df_filtered



def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    stats_df = (
    df
    .groupby("placement_id")
    .apply(lambda x: x.iloc[:25]["daily_loss"].agg(["mean", "std"]))
    .reset_index()
    )
    placements_to_remove = []

    for pid, group in df.groupby("placement_id"):
        # Skip if placement has less than 31 rows (safety check)
        if len(group) < 31:
            continue

        mean_val = stats_df.loc[stats_df["placement_id"] == pid, "mean"].values[0]
        std_val  = stats_df.loc[stats_df["placement_id"] == pid, "std"].values[0]

        last_two = group.iloc[-6:]["daily_loss"]

        if any(abs(last_two - mean_val) > 3 * std_val):
            placements_to_remove.append(pid)

    df = df[
        ~df["placement_id"].isin(placements_to_remove)
    ].reset_index(drop=True)
    
    return df


def remove_nulls_bird_density(df: pd.DataFrame) -> pd.DataFrame:

    null_values = df[df['birds_density'].isnull()]['placement_id'].unique()
    df = df[~df['placement_id'].isin(null_values)].copy()

    return df

def remove_nonpositive_live_birds(df: pd.DataFrame) -> pd.DataFrame:
    nonpostitive_live_birds_placement_ids = df[df['live_birds_start'] <= 0]['placement_id'].unique()
    df = df[~df['placement_id'].isin(nonpostitive_live_birds_placement_ids)].copy()

    return df


def process(df, input_scaler, output_scaler):

    X, y, mask = create_sequences_for_training(
    df, input_scaler, output_scaler, numeric_features, categorical_features, TARGET, MAX_LEN, PAD_VALUE
    )

    X, y, mask = shuffle_sequences(X, y, mask, seed=42)
    
    return X, y, mask