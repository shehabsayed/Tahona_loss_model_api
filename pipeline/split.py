from utils.train_data_loading import load_for_training
from turtle import pd
from pathlib import Path

def split_for_training():
    df = load_for_training()
    
    df["the_date"] = pd.to_datetime(df["the_date"])

    df = df.sort_values(
        ["placement_id", "the_date"]
    ).reset_index(drop=True)

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

