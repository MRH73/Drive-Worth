from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.training import load_or_train_model, predict_price, prediction_line


TRAINING_DATA_PATH = Path("data/car_prices.csv")


# Flask creates the web application object.
app = Flask(__name__)

# Load the trained model when the server starts.
# If the model file does not exist yet, the app will train one automatically.
model_bundle = load_or_train_model()


def positive_int_arg(name: str, default: int, minimum: int, maximum: int) -> int:
    # Read a query parameter like ?page=2 and keep it inside a safe range.
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        value = default

    return min(max(value, minimum), maximum)


@app.route("/")
def index():
    # Render the main HTML page.
    # These lists fill the dropdown menus in the form.
    return render_template(
        "index.html",
        model_name=model_bundle["model_name"],
        makes=model_bundle["make_categories"],
        make_model_map=model_bundle["make_model_map"],
        conditions=model_bundle["conditions"],
        min_year=model_bundle["year_range"][0],
        max_year=model_bundle["year_range"][1],
        min_mileage=model_bundle["mileage_range"][0],
        max_mileage=model_bundle["mileage_range"][1],
        data_source=model_bundle.get("data_source", "unknown"),
        row_count=model_bundle.get("row_count", 0),
        make_count=len(model_bundle["make_categories"]),
        model_count=len(model_bundle["model_categories"]),
        final_cost=model_bundle["final_cost"],
        cost_function=model_bundle["cost_function"],
        metrics=model_bundle["metrics"],
        feature_note=model_bundle["feature_note"],
        feature_impacts=model_bundle.get("feature_impacts", []),
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        # Read the JSON body sent by JavaScript.
        payload = request.get_json(force=True)
        make = payload["make"]
        model_name = payload["model"]
        condition = payload["condition"]
        year = float(payload["year"])
        mileage = float(payload["mileage"])
        prediction = predict_price(
            model_bundle,
            make=make,
            model_name=model_name,
            condition=condition,
            year=year,
            mileage=mileage,
        )
        line_points = prediction_line(
            model_bundle,
            make=make,
            model_name=model_name,
            condition=condition,
            year=year,
        )

        # Return JSON so the frontend can update the page without reloading.
        return jsonify(
            {
                "predicted_price": prediction,
                "model_name": model_bundle["model_name"],
                "final_cost": model_bundle["final_cost"],
                "cost_function": model_bundle["cost_function"],
                "metrics": model_bundle["metrics"],
                "cost_history": model_bundle["cost_history"],
                "feature_impacts": model_bundle.get("feature_impacts", []),
                "line_points": line_points,
                "prediction_point": {
                    "mileage": mileage,
                    "predicted_price": prediction,
                },
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        # If a required field is missing or has the wrong type, return a bad request.
        return jsonify({"error": f"Invalid input: {error}"}), 400


@app.route("/api/metrics")
def metrics():
    # This endpoint gives the frontend model training details.
    return jsonify(
        {
            "model_name": model_bundle["model_name"],
            "metrics": model_bundle["metrics"],
            "final_cost": model_bundle["final_cost"],
            "cost_function": model_bundle["cost_function"],
            "cost_history": model_bundle["cost_history"],
            "feature_impacts": model_bundle.get("feature_impacts", []),
            "data_source": model_bundle.get("data_source", "unknown"),
            "row_count": model_bundle.get("row_count", 0),
        }
    )


@app.route("/api/training-data")
def training_data():
    # This endpoint lets the UI show the same cleaned CSV rows used for training.
    page = positive_int_arg("page", default=1, minimum=1, maximum=10_000)
    per_page = positive_int_arg("per_page", default=10, minimum=5, maximum=50)
    search = request.args.get("search", "").strip().lower()

    data = pd.read_csv(TRAINING_DATA_PATH)

    if search:
        # Simple search across the text columns that users recognize in the UI.
        search_mask = (
            data["make"].astype(str).str.lower().str.contains(search, na=False)
            | data["model"].astype(str).str.lower().str.contains(search, na=False)
            | data["condition"].astype(str).str.lower().str.contains(search, na=False)
        )
        data = data[search_mask]

    total_rows = len(data)
    total_pages = max((total_rows + per_page - 1) // per_page, 1)
    page = min(page, total_pages)
    start_index = (page - 1) * per_page
    page_rows = data.iloc[start_index : start_index + per_page]

    return jsonify(
        {
            "columns": ["make", "model", "year", "mileage", "condition", "price"],
            "rows": page_rows.to_dict(orient="records"),
            "page": page,
            "per_page": per_page,
            "total_rows": total_rows,
            "total_pages": total_pages,
        }
    )


@app.route("/health")
def health():
    # Simple endpoint used to check if the server is running.
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # This runs the Flask server if you execute: python app.py
    app.run(debug=True)
