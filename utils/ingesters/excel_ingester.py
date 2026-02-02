import pandas as pd
from .base import BaseIngester

class ExcelIngester(BaseIngester):

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        return pd.read_excel(path)
