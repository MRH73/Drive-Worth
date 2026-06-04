from flask import Flask, jsonify, render_template, request

from src.training import load_or_train_model, predict_price, prediction_line


# Flask creates the web application object.
app = Flask(__name__)

# Load the trained model when the server starts.
# If the model file does not exist yet, the app will train one automatically.
model_bundle = load_or_train_model()


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


@app.route("/health")
def health():
    # Simple endpoint used to check if the server is running.
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # This runs the Flask server if you execute: python app.py
    app.run(debug=True)
