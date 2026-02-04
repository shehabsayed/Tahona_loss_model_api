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

from abc import ABC, abstractmethod
import pandas as pd

class BaseIngester(ABC):
    
    @staticmethod
    @abstractmethod
    def load(path: str) -> pd.DataFrame:
        pass
