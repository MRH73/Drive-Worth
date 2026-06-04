from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ensure_dataset


# This beginner version uses only numeric inputs.
# That lets us train Linear Regression directly with NumPy.
FEATURES = ["year", "mileage"]
TARGET = "price"


def _prepare_arrays(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # X is the input matrix: each row is one car example.
    # y is the output vector: each value is the real car price.
    x_values = data[FEATURES].to_numpy(dtype=float)
    y_values = data[TARGET].to_numpy(dtype=float)
    return x_values, y_values


def _scale_features(x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Scaling keeps year and mileage on a similar range.
    # This makes gradient descent easier to train.
    mean = x_values.mean(axis=0)
    std = x_values.std(axis=0)
    std[std == 0] = 1
    x_scaled = (x_values - mean) / std
    return x_scaled, mean, std


def compute_cost(x_values: np.ndarray, y_values: np.ndarray, weights: np.ndarray, bias: float) -> float:
    # Cost function for Linear Regression:
    # J(w, b) = (1 / 2m) * sum((prediction - real_price)^2)
    example_count = len(y_values)
    predictions = x_values @ weights + bias
    errors = predictions - y_values
    return float((errors @ errors) / (2 * example_count))


def train_linear_regression(
    x_values: np.ndarray,
    y_values: np.ndarray,
    learning_rate: float = 0.03,
    iterations: int = 1200,
) -> tuple[np.ndarray, float, list[float]]:
    # Start with weights and bias equal to zero.
    weights = np.zeros(x_values.shape[1])
    bias = 0.0
    cost_history = []
    example_count = len(y_values)

    for iteration in range(iterations):
        # Make predictions for every training example.
        predictions = x_values @ weights + bias
        errors = predictions - y_values

        # Gradients say how to move weights and bias to reduce cost.
        weight_gradient = (x_values.T @ errors) / example_count
        bias_gradient = errors.mean()

        # Update the parameters.
        weights = weights - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient

        # Save some cost values so the UI can show training progress.
        if iteration % 25 == 0 or iteration == iterations - 1:
            cost_history.append(round(compute_cost(x_values, y_values, weights, bias), 2))

    return weights, bias, cost_history


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


def train_and_save(
    dataset_path: str | Path = "data/car_prices.csv",
    model_path: str | Path = "models/linear_model.json",
) -> dict[str, object]:
    dataset, source_name = ensure_dataset(dataset_path)
    data = pd.read_csv(dataset)

    x_values, y_values = _prepare_arrays(data)
    x_scaled, feature_mean, feature_std = _scale_features(x_values)
    weights, bias, cost_history = train_linear_regression(x_scaled, y_values)
    predictions = x_scaled @ weights + bias
    metrics = _metrics(y_values, predictions)

    model = {
        "model_name": "NumPy Linear Regression",
        "features": FEATURES,
        "weights": weights.round(6).tolist(),
        "bias": round(float(bias), 6),
        "feature_mean": feature_mean.round(6).tolist(),
        "feature_std": feature_std.round(6).tolist(),
        "final_cost": cost_history[-1],
        "cost_history": cost_history,
        "metrics": metrics,
        "data_source": source_name,
        "row_count": int(len(data)),
        "cost_function": "J(w, b) = (1 / 2m) * sum((prediction - actual_price)^2)",
    }

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_text(json.dumps(model, indent=2), encoding="utf-8")

    return model


def load_or_train_model(model_path: str | Path = "models/linear_model.json") -> dict[str, object]:
    model_file = Path(model_path)
    if not model_file.exists():
        return train_and_save(model_path=model_file)

    return json.loads(model_file.read_text(encoding="utf-8"))


def predict_price(model: dict[str, object], year: float, mileage: float) -> float:
    # Apply the same scaling used during training, then use w*x + b.
    x_values = np.array([year, mileage], dtype=float)
    mean = np.array(model["feature_mean"], dtype=float)
    std = np.array(model["feature_std"], dtype=float)
    weights = np.array(model["weights"], dtype=float)
    bias = float(model["bias"])

    x_scaled = (x_values - mean) / std
    prediction = x_scaled @ weights + bias
    return round(float(max(prediction, 0)), 2)
