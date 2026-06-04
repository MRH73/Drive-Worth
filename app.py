from flask import Flask, jsonify, render_template, request
import pandas as pd

from src.data import BODY_TYPES, BRANDS, FUEL_TYPES, TRANSMISSIONS
from src.training import load_or_train_model


# Flask creates the web application object.
app = Flask(__name__)

# Load the trained model when the server starts.
# If the model file does not exist yet, the app will train one automatically.
model_bundle = load_or_train_model()


def _prediction_input(payload: dict) -> pd.DataFrame:
    # The API receives JSON from the browser.
    # The model expects a pandas DataFrame, so we convert the JSON into one row.
    row = {
        # Numeric fields are converted to int or float.
        "year": int(payload["year"]),
        "mileage": int(payload["mileage"]),
        "engine_size": float(payload["engine_size"]),
        "horsepower": int(payload["horsepower"]),
        "accident_history": int(payload["accident_history"]),
        "owners": int(payload["owners"]),
        # Text fields stay as strings.
        # The scikit-learn pipeline will encode these categories later.
        "brand": payload["brand"],
        "fuel_type": payload["fuel_type"],
        "transmission": payload["transmission"],
        "body_type": payload["body_type"],
    }
    return pd.DataFrame([row])


@app.route("/")
def index():
    # Render the main HTML page.
    # These lists fill the dropdown menus in the form.
    return render_template(
        "index.html",
        brands=BRANDS,
        fuel_types=FUEL_TYPES,
        transmissions=TRANSMISSIONS,
        body_types=BODY_TYPES,
        best_model=model_bundle["best_model_name"],
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        # Read the JSON body sent by JavaScript.
        payload = request.get_json(force=True)

        # Convert the JSON into the same feature format used during training.
        input_data = _prediction_input(payload)

        # Ask the trained regression model to predict the car price.
        prediction = model_bundle["model"].predict(input_data)[0]

        # Show only the top three most important features in the result panel.
        top_impact = model_bundle["feature_impact"][:3]

        # Return JSON so the frontend can update the page without reloading.
        return jsonify(
            {
                "predicted_price": round(float(prediction), 2),
                "best_model": model_bundle["best_model_name"],
                "top_impact": top_impact,
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        # If a required field is missing or has the wrong type, return a bad request.
        return jsonify({"error": f"Invalid input: {error}"}), 400


@app.route("/api/metrics")
def metrics():
    # This endpoint gives the frontend the model comparison and feature impact data.
    return jsonify(
        {
            "best_model": model_bundle["best_model_name"],
            "metrics": model_bundle["metrics"],
            "feature_impact": model_bundle["feature_impact"],
        }
    )


@app.route("/health")
def health():
    # Simple endpoint used to check if the server is running.
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # This runs the Flask server if you execute: python app.py
    app.run(debug=True)
