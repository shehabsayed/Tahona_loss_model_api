from abc import ABC, abstractmethod
import pandas as pd

class BaseIngester(ABC):
    
    @staticmethod
    @abstractmethod
    def load(path: str) -> pd.DataFrame:
        pass
