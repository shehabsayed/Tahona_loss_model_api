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


from utils.train_data_loading import load_for_training
from utils.feature_selection import add_season_encoded, add_birds_density, handle_weights, add_total_meds, remove_extra_days
from utils.settings import TRAIN_FEATURES
from utils.imputation import impute_weather
from pipeline.preprocessing import remove_nulls_bird_density, remove_nonpositive_live_birds, remove_outliers, remove_small_placements
import pandas as pd


def split_for_training():

    df = load_for_training()

    df = remove_small_placements(df)
    df = remove_extra_days(df)
    df = remove_outliers(df)
    df = add_total_meds(df)
    df = remove_nonpositive_live_birds(df)
    df = handle_weights(df)
    df = add_birds_density(df)
    df = remove_nulls_bird_density(df)
    df = add_season_encoded(df)
    df = df[TRAIN_FEATURES]
    df = impute_weather(df)

    # Placement start date
    placement_start_dates = (
        df.groupby("placement_id")["the_date"]
        .min()
        .reset_index(name="placement_start_date")
    )

    # Cutoff = max date - 2 months
    cutoff_date = df["the_date"].max() - pd.DateOffset(months=2)


    test_placements = placement_start_dates.loc[
        placement_start_dates["placement_start_date"] >= cutoff_date,
        "placement_id"
    ]

    train_placements = placement_start_dates.loc[
        placement_start_dates["placement_start_date"] < cutoff_date,
        "placement_id"
    ]

    df_train = df[df["placement_id"].isin(train_placements)].copy()
    df_test  = df[df["placement_id"].isin(test_placements)].copy()

    return df_train, df_test

