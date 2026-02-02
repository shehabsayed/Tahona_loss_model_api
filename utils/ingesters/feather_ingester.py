import pandas as pd
from .base import BaseIngester

class FeatherIngester(BaseIngester):

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        return pd.read_feather(path)
