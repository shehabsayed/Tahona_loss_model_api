import pandas as pd
from datetime import datetime
from pathlib import Path


def load_today_data():
    today = datetime.today().strftime("%Y-%m-%d")
    filename = f"df-{today}.csv"

    base_path = Path(__file__).resolve().parent.parent
    data_path = base_path / "data" / filename

    if not data_path.exists():
        raise FileNotFoundError(
            f"Today's data file not found: {data_path}"
        )

    return pd.read_csv(data_path)
