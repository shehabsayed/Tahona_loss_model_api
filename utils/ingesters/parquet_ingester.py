import pandas as pd
from .base import BaseIngester

class ParquetIngester(BaseIngester):

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        return pd.read_parquet(path)
