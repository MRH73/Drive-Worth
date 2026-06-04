from pathlib import Path

import pandas as pd


SOURCE_DATA_URL = (
    "https://gist.githubusercontent.com/harshitmanda35/"
    "888474a03966b678294d2dfb29bff888/raw/car_price_prediction.csv"
)

REQUIRED_COLUMNS = ["year", "mileage", "price"]


FALLBACK_ROWS = [
    (2018, 68000, 11900),
    (2019, 51000, 17800),
    (2020, 42000, 18500),
    (2020, 53000, 21900),
    (2021, 35000, 24500),
    (2021, 26000, 36900),
    (2022, 28000, 31500),
    (2022, 24000, 42900),
]


def _fallback_dataset() -> pd.DataFrame:
    # Small backup dataset so the project still runs without internet.
    return pd.DataFrame(FALLBACK_ROWS, columns=REQUIRED_COLUMNS)


def _clean_source_data(data: pd.DataFrame) -> pd.DataFrame:
    # Support the current GitHub CSV and common Kaggle/OpenDataBay column names.
    renamed = data.rename(
        columns={
            "Year": "year",
            "model_year": "year",
            "Mileage": "mileage",
            "milage": "mileage",
            "Price": "price",
        }
    )

    cleaned = renamed[REQUIRED_COLUMNS].copy()

    # Remove symbols like "$" or "," before converting to numbers.
    cleaned["year"] = pd.to_numeric(cleaned["year"], errors="coerce")
    cleaned["mileage"] = pd.to_numeric(
        cleaned["mileage"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )
    cleaned["price"] = pd.to_numeric(
        cleaned["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )

    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    cleaned["year"] = cleaned["year"].astype(int)
    cleaned["mileage"] = cleaned["mileage"].astype(int)
    cleaned["price"] = cleaned["price"].astype(float)

    return cleaned


def load_source_dataset() -> tuple[pd.DataFrame, str]:
    # First try the real public CSV. Use the fallback only if download fails.
    try:
        source_data = pd.read_csv(SOURCE_DATA_URL)
        return _clean_source_data(source_data), SOURCE_DATA_URL
    except Exception:
        return _fallback_dataset(), "local fallback sample"


def ensure_dataset(path: str | Path = "data/car_prices.csv") -> tuple[Path, str]:
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        existing_data = pd.read_csv(dataset_path)
        cleaned_data = _clean_source_data(existing_data)
        cleaned_data.to_csv(dataset_path, index=False)
        return dataset_path, f"local CSV cleaned from {SOURCE_DATA_URL}"

    data, source_name = load_source_dataset()
    data.to_csv(dataset_path, index=False)
    return dataset_path, source_name
