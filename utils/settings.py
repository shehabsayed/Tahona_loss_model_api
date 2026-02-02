FEATURES = [
    "placement_id",
    "age",
    "temperature_min",
    "humidity_morning",
    "daily_weight",
    "daily_first_feed_intake",
    "daily_second_feed_intake",
    "daily_mortality",
    "total_meds",
    "birds_density",
    "season_encoded",
    "house_id",
    "farm_id"
]


MAX_LEN = 31
PAD_VALUE = 0

numeric_features = [
    "age",
    "temperature_min",
    "humidity_morning",
    "daily_weight",
    "daily_first_feed_intake",
    "daily_second_feed_intake",
    "daily_mortality",
    "total_meds",
    "birds_density"
]

categorical_features = [
    "season_encoded",
    "house_id",
    "farm_id"
]
