import pandas as pd
import gzip
import shutil
import tempfile
from .base import BaseIngester


class FeatherIngester(BaseIngester):

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        if path.endswith(".gz"):
            # Decompress to temp file
            with gzip.open(path, "rb") as f_in:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ftr") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    temp_path = f_out.name

            df = pd.read_feather(temp_path)
        else:
            df = pd.read_feather(path)

        return df
