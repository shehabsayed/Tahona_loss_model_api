import pandas as pd
from .base import BaseIngester

class CSVIngester(BaseIngester):

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        return pd.read_csv(path)
