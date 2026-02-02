from pathlib import Path
from datetime import datetime

from .csv_ingester import CSVIngester
from .excel_ingester import ExcelIngester
from .parquet_ingester import ParquetIngester
from .feather_ingester import FeatherIngester


class DataIngester:

    INGESTERS = {
        ".csv": CSVIngester,
        ".xlsx": ExcelIngester,
        ".xls": ExcelIngester,
        ".parquet": ParquetIngester,
        ".ftr": FeatherIngester,
        ".feather": FeatherIngester,
    }

    @staticmethod
    def load_today():
        today = datetime.today().strftime("%Y-%m-%d")
        prefix = f"df-{today}"

        base_path = Path(__file__).resolve().parent.parent.parent
        data_dir = base_path / "data"

        matches = list(data_dir.glob(f"{prefix}.*"))

        if not matches:
            raise FileNotFoundError(
                f"No data file found for today: {prefix}"
            )

        path = matches[0]
        ext = path.suffix.lower()

        if ext not in DataIngester.INGESTERS:
            raise ValueError(f"Unsupported file format: {ext}")

        ingester = DataIngester.INGESTERS[ext]
        return ingester.load(path)
