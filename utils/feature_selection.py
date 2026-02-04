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



# Helper function to determine the season from the month
def get_season(month: int) -> str:
    if 3 <= month <= 5:
        return "Spring"
    elif 6 <= month <= 8:
        return "Summer"
    elif 9 <= month <= 11:
        return "Autumn"
    else:
        return "Winter"

# Encodes the season based on the date and adds it as a new feature
def add_season_encoded(df: pd.DataFrame) -> pd.DataFrame:
    season_mapping = {
        "Spring": 4,
        "Summer": 3,
        "Winter": 2,
        "Autumn": 1
    }

    df = df.copy()
    df["the_date"] = pd.to_datetime(df["the_date"])
    df["season_encoded"] = df["the_date"].dt.month.apply(get_season).map(season_mapping)


    return df

# Sums up various medication-related columns to create a total medication feature
def add_total_meds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    vaccination = [c for c in df.columns if c.startswith("vaccination_daily")]
    medication = [c for c in df.columns if c.startswith("medication_daily")]
    anticoccidials = [c for c in df.columns if c.startswith("othertypes_daily_Anticoccidials")]
    disinfectant = [c for c in df.columns if c.startswith("othertypes_daily_Disinfectant")]
    vitamins = [c for c in df.columns if c.startswith("othertypes_daily_Vitamins")]
    probiotics = [c for c in df.columns if c.startswith("othertypes_daily_Probiotics&herbal products")]
    expectorants = [c for c in df.columns if c.startswith("othertypes_daily_مقشعات")]
    wood = [c for c in df.columns if c.startswith("othertypes_daily_wood shavings")]

    groups = [
        vaccination, medication, anticoccidials,
        disinfectant, vitamins, probiotics,
        expectorants
    ]

    total_cols = []
    for cols in groups:
        if cols:
            name = cols[0].split("_daily")[0] + "_total"
            df[name] = df[cols].sum(axis=1)
            total_cols.extend(cols)

    if wood:
        df["wood_shavings_total"] = df[wood].sum(axis=1)
        total_cols.extend(wood)

    meds_cols = [c for c in df.columns if c.endswith("_total") and c != "wood_shavings_total"]
    df["total_meds"] = df[meds_cols].sum(axis=1)

    df = df.drop(columns=total_cols + meds_cols)

    return df

# Ensures that daily weights are in a consistent format (grams)
def handle_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    fixed_groups = []

    for placement_id, group in df.groupby("placement_id"):
        group = group.sort_values("age").reset_index(drop=True)

        for i in range(len(group) - 1):
            current = group.loc[i, "daily_weight"]
            next_val = group.loc[i + 1, "daily_weight"]

            if current < 1 and current > next_val:
                group.loc[i, "daily_weight"] = current / 10

        group["daily_weight"] = group["daily_weight"].apply(
            lambda x: x * 1000 if x < 1 else x
        )

        fixed_groups.append(group)

    return pd.concat(fixed_groups).reset_index(drop=True)

# Adds a feature representing the density of birds in the area (density = (live_birds_start * daily_weight) / size_area)
def add_birds_density(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["birds_density"] = (
        df["live_birds_start"] * df["daily_weight"]
    ) / df["size_area"]

    df = df.drop(columns=["size_area", "live_birds_start"])
    return df

# Removes any extra days beyond the maximum allowed per placement (31 days)
def remove_extra_days(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(["placement_id", "the_date"]).reset_index(drop=True)
    df = (df.groupby("placement_id", group_keys=False).head(31))
    return df


# Main function to build the feature set
def build_features(df: pd.DataFrame, features) -> pd.DataFrame:
    df = add_season_encoded(df)
    df = add_total_meds(df)
    df = handle_weights(df)
    df = add_birds_density(df)
    df = remove_extra_days(df)
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return df[features]
