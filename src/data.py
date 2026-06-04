from pathlib import Path

import pandas as pd


# Public CSV source used by the project.
# It is a GitHub Gist CSV with brand, model, year, mileage, fuel, transmission,
# condition, and price columns.
SOURCE_DATA_URL = (
    "https://gist.githubusercontent.com/harshitmanda35/"
    "888474a03966b678294d2dfb29bff888/raw/car_price_prediction.csv"
)

# These are the columns we keep for the beginner version of the model.
# We skip engine size and horsepower because the user asked to avoid them.
REQUIRED_COLUMNS = [
    "brand",
    "model",
    "year",
    "mileage",
    "fuel_type",
    "transmission",
    "condition",
    "price",
]


# If the online CSV cannot be downloaded, this small local dataset keeps the
# project runnable. It is only a fallback, not the preferred source.
FALLBACK_ROWS = [
    ("Toyota", "Corolla", 2020, 42000, "Gasoline", "Automatic", "Used", 18500),
    ("Toyota", "Camry", 2021, 35000, "Hybrid", "Automatic", "Used", 24500),
    ("Toyota", "RAV4", 2022, 28000, "Gasoline", "Automatic", "Like New", 31500),
    ("Honda", "Civic", 2019, 51000, "Gasoline", "Automatic", "Used", 17800),
    ("Honda", "Accord", 2020, 44000, "Gasoline", "Automatic", "Used", 22600),
    ("Honda", "CR-V", 2021, 39000, "Gasoline", "Automatic", "Like New", 28900),
    ("Ford", "Focus", 2018, 68000, "Gasoline", "Manual", "Used", 11900),
    ("Ford", "Explorer", 2020, 58000, "Gasoline", "Automatic", "Used", 27500),
    ("Ford", "Mustang", 2021, 26000, "Gasoline", "Automatic", "Like New", 36900),
    ("Chevrolet", "Malibu", 2019, 61000, "Gasoline", "Automatic", "Used", 15700),
    ("Chevrolet", "Equinox", 2020, 53000, "Gasoline", "Automatic", "Used", 21900),
    ("Chevrolet", "Silverado", 2021, 47000, "Diesel", "Automatic", "Used", 38200),
    ("BMW", "3 Series", 2020, 42000, "Gasoline", "Automatic", "Used", 30900),
    ("BMW", "5 Series", 2021, 31000, "Hybrid", "Automatic", "Like New", 43800),
    ("BMW", "X5", 2022, 25000, "Gasoline", "Automatic", "Like New", 57900),
    ("Mercedes", "C-Class", 2020, 39000, "Gasoline", "Automatic", "Used", 33500),
    ("Mercedes", "E-Class", 2021, 33000, "Hybrid", "Automatic", "Like New", 48900),
    ("Mercedes", "GLC", 2022, 26000, "Gasoline", "Automatic", "Like New", 54800),
    ("Hyundai", "Elantra", 2020, 48000, "Gasoline", "Automatic", "Used", 16800),
    ("Hyundai", "Tucson", 2021, 36000, "Gasoline", "Automatic", "Used", 24800),
    ("Hyundai", "Santa Fe", 2022, 29000, "Hybrid", "Automatic", "Like New", 33500),
    ("Kia", "Forte", 2020, 46000, "Gasoline", "Automatic", "Used", 15900),
    ("Kia", "Sportage", 2021, 37000, "Gasoline", "Automatic", "Used", 23900),
    ("Kia", "Sorento", 2022, 30000, "Hybrid", "Automatic", "Like New", 34900),
    ("Tesla", "Model 3", 2021, 33000, "Electric", "Automatic", "Used", 32900),
    ("Tesla", "Model Y", 2022, 24000, "Electric", "Automatic", "Like New", 42900),
    ("Tesla", "Model X", 2021, 28000, "Electric", "Automatic", "Used", 64900),
    ("Audi", "A3", 2020, 41000, "Gasoline", "Automatic", "Used", 27900),
    ("Audi", "A4", 2021, 34000, "Gasoline", "Automatic", "Like New", 36500),
    ("Audi", "Q5", 2022, 27000, "Hybrid", "Automatic", "Like New", 47200),
    ("Nissan", "Altima", 2020, 50000, "Gasoline", "Automatic", "Used", 17900),
    ("Nissan", "Rogue", 2021, 39000, "Gasoline", "Automatic", "Used", 24400),
    ("Volkswagen", "Jetta", 2020, 47000, "Gasoline", "Automatic", "Used", 17400),
    ("Volkswagen", "Tiguan", 2021, 36000, "Gasoline", "Automatic", "Used", 26200),
]


def _fallback_dataset() -> pd.DataFrame:
    # Convert the fallback rows into a pandas DataFrame.
    return pd.DataFrame(FALLBACK_ROWS, columns=REQUIRED_COLUMNS)


def _clean_source_data(data: pd.DataFrame) -> pd.DataFrame:
    # The source CSV uses names like "Fuel Type".
    # Rename them to clean snake_case names used by our code.
    renamed = data.rename(
        columns={
            "Brand": "brand",
            "Model": "model",
            "Year": "year",
            "model_year": "year",
            "Mileage": "mileage",
            "milage": "mileage",
            "Fuel Type": "fuel_type",
            "Transmission": "transmission",
            "Condition": "condition",
            "Price": "price",
        }
    )

    # The OpenDataBay/Kaggle-style dataset may not have a condition column.
    # Keep the app simple by filling it with Unknown when it is missing.
    if "condition" not in renamed.columns:
        renamed["condition"] = "Unknown"

    # Keep only the beginner-friendly columns.
    cleaned = renamed[REQUIRED_COLUMNS].copy()

    # Convert numeric values to real numbers and remove broken rows.
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

    # Normalize text so dropdown values are consistent.
    text_columns = ["brand", "model", "fuel_type", "transmission", "condition"]
    for column in text_columns:
        cleaned[column] = cleaned[column].astype(str).str.strip()

    # Cast numeric columns after removing missing values.
    cleaned["year"] = cleaned["year"].astype(int)
    cleaned["mileage"] = cleaned["mileage"].astype(int)
    cleaned["price"] = cleaned["price"].astype(float)

    return cleaned


def load_source_dataset() -> tuple[pd.DataFrame, str]:
    # First try to read the online CSV.
    # If the internet is unavailable, fall back to the small local sample.
    try:
        source_data = pd.read_csv(SOURCE_DATA_URL)
        return _clean_source_data(source_data), SOURCE_DATA_URL
    except Exception:
        return _fallback_dataset(), "local fallback sample"


def ensure_dataset(path: str | Path = "data/car_prices.csv") -> tuple[Path, str]:
    # This function makes sure the training CSV exists and has the right columns.
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        existing = pd.read_csv(dataset_path)
        if set(REQUIRED_COLUMNS).issubset(existing.columns):
            return dataset_path, "existing local CSV"
        try:
            cleaned_existing = _clean_source_data(existing)
            cleaned_existing.to_csv(dataset_path, index=False)
            return dataset_path, "downloaded public CSV"
        except Exception:
            pass

    # Download and clean the public source if possible.
    data, source_name = load_source_dataset()
    data.to_csv(dataset_path, index=False)

    return dataset_path, source_name


def dataset_options(path: str | Path = "data/car_prices.csv") -> dict[str, object]:
    # The UI uses this to build dropdowns.
    dataset_path, _source_name = ensure_dataset(path)
    data = pd.read_csv(dataset_path)

    brand_model_map = {}
    for brand, group in data.groupby("brand"):
        brand_model_map[brand] = sorted(group["model"].unique().tolist())

    return {
        "brands": sorted(data["brand"].unique().tolist()),
        "brand_model_map": brand_model_map,
        "fuel_types": sorted(data["fuel_type"].unique().tolist()),
        "transmissions": sorted(data["transmission"].unique().tolist()),
        "conditions": sorted(data["condition"].unique().tolist()),
        "min_year": int(data["year"].min()),
        "max_year": int(data["year"].max()),
    }
