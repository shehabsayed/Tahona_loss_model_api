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

import gzip
import pandas as pd
from pathlib import Path


def load_for_training():
        
        prefix = f"custom_broilers_df.ftr.gz"

        base_path = Path(__file__).resolve().parent.parent
        data_dir = base_path / "data"

        matches = data_dir/prefix
        if not matches.exists():
            raise FileNotFoundError(
                f"No data file found for today: {prefix}"
            )

        with gzip.open(matches,'rb') as f:
            df = pd.read_feather(f)
        
        return df
