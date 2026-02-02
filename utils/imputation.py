import pandas as pd


# Sort data by placement and age to prepare for time-based interpolation
def sort_for_imputation(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["placement_id", "age"]
    ).reset_index(drop=True)


# Linearly interpolate missing weather values inside each placement
def interpolate_weather(df: pd.DataFrame, weather_cols) -> pd.DataFrame:
    df = df.copy()

    for col in weather_cols:
        df[col] = (
            df.groupby("placement_id")[col]
              .transform(lambda x: x.interpolate(
                  method="linear",
                  limit_direction="both"
              ))
        )

    return df


# Fill remaining missing weather values using median of (farm_id, season)
def fill_by_farm_and_season(df: pd.DataFrame, weather_cols) -> pd.DataFrame:
    df = df.copy()

    for col in weather_cols:
        df[col] = df[col].fillna(
            df.groupby(["farm_id", "season_encoded"])[col]
              .transform("median")
        )

    return df


# Fill any remaining missing weather values using median of the season only
def fill_by_season(df: pd.DataFrame, weather_cols) -> pd.DataFrame:
    df = df.copy()

    for col in weather_cols:
        df[col] = df[col].fillna(
            df.groupby("season_encoded")[col]
              .transform("median")
        )

    return df

def fill_remaining_with_global_median(df: pd.DataFrame, weather_cols) -> pd.DataFrame:
    df = df.copy()

    WEATHER_MEDIANS = {
    1 : {'temperature_min': 28.0, 'humidity_morning': 55.0},
    2 : {'temperature_min': 28.0, 'humidity_morning': 52.0},
    3 : {'temperature_min': 29.5, 'humidity_morning': 50.0},
    4 : {'temperature_min': 31.8, 'humidity_morning': 39.0},
    }
    for col in weather_cols:
        df[col] = df[col].fillna(
            df['season_encoded'].map(lambda season: WEATHER_MEDIANS[season][col])
        )

    return df

# Full weather imputation pipeline
def impute_weather(df: pd.DataFrame) -> pd.DataFrame:
    weather_cols = ["temperature_min", "humidity_morning"]

    df = sort_for_imputation(df)
    df = interpolate_weather(df, weather_cols)
    df = fill_by_farm_and_season(df, weather_cols)
    df = fill_by_season(df, weather_cols)
    df = fill_remaining_with_global_median(df, weather_cols)

    return df
