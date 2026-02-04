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

TRAIN_FEATURES = [
    "the_date",
    "daily_loss",
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

TARGET = "daily_loss"

