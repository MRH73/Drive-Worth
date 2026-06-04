from pathlib import Path

import pandas as pd


SOURCE_DATA_URL = (
    "https://raw.githubusercontent.com/stat-lu/dataviz/main/data/us_cars.csv"
)

RAW_DATA_PATH = Path("data/source_car_prices.csv")
TRAINING_DATA_PATH = Path("data/car_prices.csv")
REQUIRED_COLUMNS = ["make", "model", "year", "mileage", "condition", "price"]
EXPECTED_CONDITIONS = {"Clean Vehicle", "Salvage Insurance"}


FALLBACK_ROWS = [
    ("Ford", "Door", 2018, 42000, "Clean Vehicle", 20700),
    ("Chevrolet", "1500", 2018, 35000, "Clean Vehicle", 27000),
    ("Dodge", "Mpv", 2018, 51000, "Clean Vehicle", 9000),
    ("Nissan", "Altima", 2019, 26000, "Clean Vehicle", 18500),
    ("Ford", "Escape", 2017, 61000, "Clean Vehicle", 14500),
    ("Ford", "Door", 2014, 88000, "Salvage Insurance", 8200),
]


def _fallback_dataset() -> pd.DataFrame:
    return pd.DataFrame(FALLBACK_ROWS, columns=REQUIRED_COLUMNS)


def _clean_source_data(data: pd.DataFrame) -> pd.DataFrame:
    # This source CSV has title_status, which is a useful simple condition field:
    # "clean vehicle" or "salvage insurance".
    condition_column = "title_status" if "title_status" in data.columns else "Condition"
    working_data = data.copy()
    if condition_column == "title_status" and "condition" in working_data.columns:
        working_data = working_data.drop(columns=["condition"])

    renamed = working_data.rename(
        columns={
            "brand": "make",
            "Brand": "make",
            "model": "model",
            "Model": "model",
            "year": "year",
            "Year": "year",
            "model_year": "year",
            "mileage": "mileage",
            "Mileage": "mileage",
            "milage": "mileage",
            "price": "price",
            "Price": "price",
            condition_column: "condition",
        }
    )

    cleaned = renamed[REQUIRED_COLUMNS].copy()
    cleaned["make"] = cleaned["make"].astype(str).str.strip().str.title()
    cleaned["model"] = cleaned["model"].astype(str).str.strip().str.title()
    cleaned["condition"] = cleaned["condition"].astype(str).str.strip().str.title()
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

    # Remove rows that are clearly not useful for beginner-friendly training:
    # $0 prices, very old cars, and extreme mileage outliers.
    cleaned = cleaned[
        (cleaned["price"] >= 1000)
        & (cleaned["price"] <= 100000)
        & (cleaned["year"] >= 1990)
        & (cleaned["mileage"] >= 100)
        & (cleaned["mileage"] <= 250000)
    ]

    return cleaned


def _has_current_source_schema(data: pd.DataFrame) -> bool:
    # If the cached raw file is from an older source, download the new source.
    return {"brand", "model", "year", "mileage", "title_status", "price"}.issubset(data.columns)


def _has_current_training_schema(data: pd.DataFrame) -> bool:
    # If the cleaned CSV still has the old New/Used condition values, rebuild it.
    if not set(REQUIRED_COLUMNS).issubset(data.columns):
        return False

    conditions = set(data["condition"].dropna().astype(str).unique().tolist())
    return EXPECTED_CONDITIONS.issubset(conditions)


def load_source_dataset() -> tuple[pd.DataFrame, str]:
    if RAW_DATA_PATH.exists():
        local_data = pd.read_csv(RAW_DATA_PATH)
        if _has_current_source_schema(local_data):
            return _clean_source_data(local_data), f"local raw CSV at {RAW_DATA_PATH}"

    try:
        source_data = pd.read_csv(SOURCE_DATA_URL)
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        source_data.to_csv(RAW_DATA_PATH, index=False)
        return _clean_source_data(source_data), SOURCE_DATA_URL
    except Exception:
        return _fallback_dataset(), "local fallback sample"


def ensure_dataset(path: str | Path = TRAINING_DATA_PATH) -> tuple[Path, str]:
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        existing_data = pd.read_csv(dataset_path)
        if _has_current_training_schema(existing_data):
            return dataset_path, f"cleaned CSV at {dataset_path}"

    data, source_name = load_source_dataset()
    data.to_csv(dataset_path, index=False)
    return dataset_path, source_name


def dataset_options(path: str | Path = TRAINING_DATA_PATH) -> dict[str, object]:
    dataset_path, _source_name = ensure_dataset(path)
    data = pd.read_csv(dataset_path)

    make_model_map = {}
    for make, group in data.groupby("make"):
        make_model_map[make] = sorted(group["model"].unique().tolist())

    return {
        "makes": sorted(data["make"].unique().tolist()),
        "make_model_map": make_model_map,
        "conditions": sorted(data["condition"].unique().tolist()),
        "min_year": int(data["year"].min()),
        "max_year": int(data["year"].max()),
        "min_mileage": int(data["mileage"].min()),
        "max_mileage": int(data["mileage"].max()),
    }
