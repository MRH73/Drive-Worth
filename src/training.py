from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import ensure_dataset


# These are the numeric columns used by the model.
NUMERIC_FEATURES = [
    "year",
    "mileage",
    "engine_size",
    "horsepower",
    "accident_history",
    "owners",
]

# These are text/category columns.
# They need encoding before a model can use them.
CATEGORICAL_FEATURES = ["brand", "fuel_type", "transmission", "body_type"]

# This is the value we want to predict.
TARGET = "price"


def _preprocessor() -> ColumnTransformer:
    # A ColumnTransformer lets us process different columns in different ways.
    return ColumnTransformer(
        transformers=[
            # StandardScaler puts numeric values on a similar scale.
            # This helps linear models work better.
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            # OneHotEncoder turns categories into 0/1 columns.
            # Example: brand=Toyota becomes a Toyota column with value 1.
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def _models() -> dict[str, object]:
    # Every model here is a regression model because it predicts a number.
    return {
        # Simple baseline model.
        "Linear Regression": LinearRegression(),
        # Linear model with regularization to reduce overfitting.
        "Ridge Regression": Ridge(alpha=1.0),
        # Non-linear model that averages many decision trees.
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=12,
            n_jobs=1,
        ),
        # Non-linear model that improves mistakes step by step.
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.07,
            max_depth=3,
            random_state=42,
        ),
    }


def _metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    # MAE: average absolute error in dollars.
    # RMSE: similar to MAE, but punishes large errors more.
    # R2: how much of the price variation the model explains.
    return {
        "mae": round(mean_absolute_error(y_true, predictions), 2),
        "rmse": round(root_mean_squared_error(y_true, predictions), 2),
        "r2": round(r2_score(y_true, predictions), 4),
    }


def _feature_impact(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> list[dict[str, float | str]]:
    # Permutation importance explains which features matter most.
    # It shuffles one column at a time and checks how much the model gets worse.
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
        # Higher impact means that feature helped the model more.
        rows.append({"feature": feature, "impact": round(float(value), 2)})

    # Sort from most important to least important.
    return sorted(rows, key=lambda item: item["impact"], reverse=True)


def train_and_save(
    dataset_path: str | Path = "data/car_prices.csv",
    model_path: str | Path = "models/best_model.joblib",
) -> dict[str, object]:
    # Make sure the CSV exists, then load it with pandas.
    dataset = ensure_dataset(dataset_path)
    data = pd.read_csv(dataset)

    # X contains the input features.
    # y contains the target value: car price.
    x = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]

    # Split the data into training and testing sets.
    # The model learns from training data and is evaluated on test data.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    results = []
    fitted_models = {}

    # Train and evaluate every model.
    for name, estimator in _models().items():
        # A Pipeline connects preprocessing and model training together.
        # This prevents mistakes where training and prediction use different steps.
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _preprocessor()),
                ("model", estimator),
            ]
        )
        # Learn patterns from the training data.
        pipeline.fit(x_train, y_train)

        # Predict prices for cars the model did not train on.
        predictions = pipeline.predict(x_test)

        # Calculate model quality.
        model_metrics = _metrics(y_test, predictions)
        results.append({"name": name, **model_metrics})

        # Save the trained model in memory so we can choose the best one later.
        fitted_models[name] = pipeline

    # Pick the model with the lowest RMSE.
    results = sorted(results, key=lambda row: row["rmse"])
    best_name = results[0]["name"]
    best_model = fitted_models[best_name]

    # Calculate which features had the biggest impact for the best model.
    impact = _feature_impact(best_model, x_test, y_test)

    # Store everything the Flask app needs.
    bundle = {
        "model": best_model,
        "best_model_name": best_name,
        "metrics": results,
        "feature_impact": impact,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "input_features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    }

    # Save the model bundle to disk.
    # This avoids retraining the model every time the app starts.
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_file)

    return bundle


def load_or_train_model(model_path: str | Path = "models/best_model.joblib") -> dict[str, object]:
    # If a trained model already exists, load it.
    model_file = Path(model_path)
    if not model_file.exists():
        # If not, train one first.
        return train_and_save(model_path=model_file)

    return joblib.load(model_file)
