from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import SOURCE_DATA_URL, dataset_options, ensure_dataset


MODEL_VERSION = 3
RAW_NUMERIC_FEATURES = ["year", "mileage"]
ENGINEERED_NUMERIC_FEATURES = [
    "year",
    "mileage",
    "vehicle_age",
    "vehicle_age_squared",
    "mileage_log",
    "mileage_squared",
]
CATEGORICAL_FEATURES = ["make", "model", "condition"]
TARGET = "price"


def _one_hot(values: pd.Series, categories: list[str]) -> np.ndarray:
    matrix = np.zeros((len(values), len(categories)))
    category_index = {category: index for index, category in enumerate(categories)}

    for row_index, value in enumerate(values.astype(str)):
        column_index = category_index.get(value)
        if column_index is not None:
            matrix[row_index, column_index] = 1

    return matrix


def _engineer_numeric_features(data: pd.DataFrame, reference_year: int) -> np.ndarray:
    year = data["year"].to_numpy(dtype=float)
    mileage = data["mileage"].to_numpy(dtype=float)
    vehicle_age = np.maximum(reference_year - year, 0)

    return np.column_stack(
        [
            year,
            mileage,
            vehicle_age,
            vehicle_age**2,
            np.log1p(mileage),
            (mileage / 100_000) ** 2,
        ]
    )


def _build_design_matrix(
    data: pd.DataFrame,
    numeric_mean: np.ndarray | None = None,
    numeric_std: np.ndarray | None = None,
    make_categories: list[str] | None = None,
    model_categories: list[str] | None = None,
    condition_categories: list[str] | None = None,
    reference_year: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if reference_year is None:
        reference_year = int(data["year"].max())

    numeric = _engineer_numeric_features(data, reference_year)

    if numeric_mean is None:
        numeric_mean = numeric.mean(axis=0)
    if numeric_std is None:
        numeric_std = numeric.std(axis=0)

    numeric_std[numeric_std == 0] = 1
    numeric_scaled = (numeric - numeric_mean) / numeric_std

    if make_categories is None:
        make_categories = sorted(data["make"].astype(str).unique().tolist())
    if model_categories is None:
        model_categories = sorted(data["model"].astype(str).unique().tolist())
    if condition_categories is None:
        condition_categories = sorted(data["condition"].astype(str).unique().tolist())

    make_encoded = _one_hot(data["make"], make_categories)
    model_encoded = _one_hot(data["model"], model_categories)
    condition_encoded = _one_hot(data["condition"], condition_categories)
    design_matrix = np.hstack([numeric_scaled, make_encoded, model_encoded, condition_encoded])

    metadata = {
        "numeric_mean": numeric_mean,
        "numeric_std": numeric_std,
        "make_categories": make_categories,
        "model_categories": model_categories,
        "condition_categories": condition_categories,
        "reference_year": reference_year,
    }
    return design_matrix, metadata


def compute_cost(x_values: np.ndarray, y_values: np.ndarray, weights: np.ndarray, bias: float) -> float:
    example_count = len(y_values)
    predictions = x_values @ weights + bias
    errors = predictions - y_values
    return float((errors @ errors) / (2 * example_count))


def train_linear_regression(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, float, list[float]]:
    starting_weights = np.zeros(x_values.shape[1])
    starting_bias = 0.0
    starting_cost = compute_cost(x_values, y_values, starting_weights, starting_bias)

    x_with_bias = np.column_stack([np.ones(len(x_values)), x_values])
    solution, _residuals, _rank, _singular_values = np.linalg.lstsq(x_with_bias, y_values, rcond=None)
    bias = float(solution[0])
    weights = solution[1:]
    final_cost = compute_cost(x_values, y_values, weights, bias)

    return weights, bias, [round(starting_cost, 2), round(final_cost, 2)]


def _to_price(raw_predictions: np.ndarray) -> np.ndarray:
    # The model now predicts real dollar prices directly.
    # We only stop negative prices, because a car price cannot be below $0.
    return np.maximum(raw_predictions, 0)


def _metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    errors = predictions - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    total_variation = np.sum((y_true - y_true.mean()) ** 2)
    unexplained_variation = np.sum(errors**2)
    r2 = 1 - unexplained_variation / total_variation if total_variation else 0.0

    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
    }


def _feature_impacts(
    data: pd.DataFrame,
    metadata: dict[str, object],
    weights: np.ndarray,
    bias: float,
    baseline_predictions: np.ndarray,
) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(42)
    y_values = data[TARGET].to_numpy(dtype=float)
    baseline_mae = np.mean(np.abs(baseline_predictions - y_values))
    impacts = []

    for feature in ["make", "model", "condition", "year", "mileage"]:
        shuffled = data.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        shuffled_x, _metadata = _build_design_matrix(
            shuffled,
            numeric_mean=metadata["numeric_mean"],
            numeric_std=metadata["numeric_std"],
            make_categories=metadata["make_categories"],
            model_categories=metadata["model_categories"],
            condition_categories=metadata["condition_categories"],
            reference_year=metadata["reference_year"],
        )
        shuffled_predictions = _to_price(shuffled_x @ weights + bias)
        shuffled_mae = np.mean(np.abs(shuffled_predictions - y_values))
        impacts.append(
            {
                "feature": feature,
                "impact": round(float(max(shuffled_mae - baseline_mae, 0)), 2),
            }
        )

    return sorted(impacts, key=lambda item: item["impact"], reverse=True)


def train_and_save(
    dataset_path: str | Path = "data/car_prices.csv",
    model_path: str | Path = "models/linear_model.json",
) -> dict[str, object]:
    dataset, source_name = ensure_dataset(dataset_path)
    data = pd.read_csv(dataset)

    x_values, metadata = _build_design_matrix(data)
    y_values = data[TARGET].to_numpy(dtype=float)
    weights, bias, cost_history = train_linear_regression(x_values, y_values)
    predictions = _to_price(x_values @ weights + bias)
    metrics = _metrics(y_values, predictions)
    feature_impacts = _feature_impacts(data, metadata, weights, bias, predictions)
    options = dataset_options(dataset_path)

    model = {
        "model_name": "NumPy Linear Regression with Nonlinear Features",
        "model_version": MODEL_VERSION,
        "data_source_url": SOURCE_DATA_URL,
        "raw_numeric_features": RAW_NUMERIC_FEATURES,
        "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "weights": weights.round(8).tolist(),
        "bias": round(float(bias), 8),
        "numeric_mean": metadata["numeric_mean"].round(8).tolist(),
        "numeric_std": metadata["numeric_std"].round(8).tolist(),
        "make_categories": metadata["make_categories"],
        "model_categories": metadata["model_categories"],
        "condition_categories": metadata["condition_categories"],
        "make_model_map": options["make_model_map"],
        "conditions": options["conditions"],
        "mileage_range": [options["min_mileage"], options["max_mileage"]],
        "year_range": [options["min_year"], options["max_year"]],
        "reference_year": metadata["reference_year"],
        "final_cost": cost_history[-1],
        "cost_history": cost_history,
        "feature_impacts": feature_impacts,
        "metrics": metrics,
        "data_source": source_name,
        "row_count": int(len(data)),
        "cost_function": "J(w, b) = (1 / 2m) * sum((predicted_price - actual_price)^2)",
        "feature_note": "The model uses nonlinear feature engineering: log(mileage), mileage squared, vehicle age, and vehicle age squared.",
    }

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_text(json.dumps(model, indent=2), encoding="utf-8")

    return model


def load_or_train_model(model_path: str | Path = "models/linear_model.json") -> dict[str, object]:
    model_file = Path(model_path)
    if not model_file.exists():
        return train_and_save(model_path=model_file)

    model = json.loads(model_file.read_text(encoding="utf-8"))
    if model.get("model_version") != MODEL_VERSION:
        return train_and_save(model_path=model_file)

    if "condition_categories" not in model or "feature_note" not in model:
        return train_and_save(model_path=model_file)

    return model


def _input_frame(make: str, model_name: str, condition: str, year: float, mileage: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "make": make,
                "model": model_name,
                "condition": condition,
                "year": year,
                "mileage": mileage,
            }
        ]
    )


def _predict_from_frame(model: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    x_values, _metadata = _build_design_matrix(
        frame,
        numeric_mean=np.array(model["numeric_mean"], dtype=float),
        numeric_std=np.array(model["numeric_std"], dtype=float),
        make_categories=model["make_categories"],
        model_categories=model["model_categories"],
        condition_categories=model["condition_categories"],
        reference_year=model["reference_year"],
    )
    weights = np.array(model["weights"], dtype=float)
    bias = float(model["bias"])
    return _to_price(x_values @ weights + bias)


def predict_price(model: dict[str, object], make: str, model_name: str, condition: str, year: float, mileage: float) -> float:
    frame = _input_frame(make, model_name, condition, year, mileage)
    prediction = _predict_from_frame(model, frame)[0]
    return round(float(prediction), 2)


def prediction_line(model: dict[str, object], make: str, model_name: str, condition: str, year: float) -> list[dict[str, float]]:
    min_mileage, max_mileage = model["mileage_range"]
    mileage_values = np.linspace(min_mileage, max_mileage, 40)
    rows = [
        {
            "make": make,
            "model": model_name,
            "condition": condition,
            "year": year,
            "mileage": mileage,
        }
        for mileage in mileage_values
    ]
    predictions = _predict_from_frame(model, pd.DataFrame(rows))

    return [
        {
            "mileage": round(float(mileage), 2),
            "predicted_price": round(float(price), 2),
        }
        for mileage, price in zip(mileage_values, predictions)
    ]
