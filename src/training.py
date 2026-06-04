from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import ensure_dataset


# Numeric columns are already numbers.
# We removed engine_size and horsepower because the user asked to skip them.
NUMERIC_FEATURES = ["year", "mileage"]

# Categorical columns are text values.
# OneHotEncoder will convert them into model-friendly 0/1 columns.
CATEGORICAL_FEATURES = ["brand", "model", "fuel_type", "transmission", "condition"]

# This is the number we want to predict.
TARGET = "price"


def _preprocessor() -> ColumnTransformer:
    # A ColumnTransformer lets us process numeric and categorical columns differently.
    return ColumnTransformer(
        transformers=[
            # Scaling helps Linear Regression compare numeric features fairly.
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            # One-hot encoding turns category values into binary columns.
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def _model() -> Pipeline:
    # The whole ML pipeline has two parts:
    # 1. preprocess the data
    # 2. train a Linear Regression model
    return Pipeline(
        steps=[
            ("preprocessor", _preprocessor()),
            ("model", LinearRegression()),
        ]
    )


def _metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    # MAE: average dollar error.
    # RMSE: dollar error that punishes very large mistakes more.
    # R2: how much variation in price the model explains.
    return {
        "mae": round(mean_absolute_error(y_true, predictions), 2),
        "rmse": round(root_mean_squared_error(y_true, predictions), 2),
        "r2": round(r2_score(y_true, predictions), 4),
    }


def _feature_impact(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> list[dict[str, float | str]]:
    # Permutation importance explains which inputs matter most.
    # It shuffles one feature at a time and measures how much accuracy drops.
    importance = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=8,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )

    rows = []
    for feature, value in zip(x_test.columns, importance.importances_mean):
        rows.append({"feature": feature, "impact": round(float(value), 2)})

    return sorted(rows, key=lambda item: item["impact"], reverse=True)


def train_and_save(
    dataset_path: str | Path = "data/car_prices.csv",
    model_path: str | Path = "models/best_model.joblib",
) -> dict[str, object]:
    # Get the dataset path. This may download the public CSV if needed.
    dataset, source_name = ensure_dataset(dataset_path)
    data = pd.read_csv(dataset)

    # X contains the information we know about the car.
    # y contains the price we want the model to learn.
    x = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]

    # Keep 20% of the rows for testing.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Train only Linear Regression.
    model = _model()
    model.fit(x_train, y_train)

    # Evaluate the model on rows it did not train on.
    predictions = model.predict(x_test)
    metrics = _metrics(y_test, predictions)
    impact = _feature_impact(model, x_test, y_test)

    # Store everything Flask needs for predictions and dashboard display.
    bundle = {
        "model": model,
        "best_model_name": "Linear Regression",
        "metrics": [{"name": "Linear Regression", **metrics}],
        "feature_impact": impact,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "input_features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "data_source": source_name,
        "row_count": int(len(data)),
    }

    # Save the trained model bundle so the app can load it quickly.
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_file)

    return bundle


def load_or_train_model(model_path: str | Path = "models/best_model.joblib") -> dict[str, object]:
    # If the saved model exists, use it.
    model_file = Path(model_path)
    if not model_file.exists():
        return train_and_save(model_path=model_file)

    bundle = joblib.load(model_file)
    expected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    # If an older model was saved with different inputs, retrain it.
    if bundle.get("input_features") != expected_features:
        return train_and_save(model_path=model_file)

    return bundle
