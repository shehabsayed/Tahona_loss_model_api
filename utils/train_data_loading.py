import gzip
import pandas as pd
from pathlib import Path


def load_for_training():
        
        prefix = f"custom_broilers_df.ftr.gz"

        base_path = Path(__file__).resolve().parent.parent
        data_dir = base_path / "data"

        matches = list(data_dir.glob(f"{prefix}.*"))

        if not matches:
            raise FileNotFoundError(
                f"No data file found for today: {prefix}"
            )

        with gzip.open(matches[0],'rb') as f:
            df = pd.read_feather(f)
        
        return df
