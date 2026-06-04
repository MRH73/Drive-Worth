from app import app


def test_health_endpoint():
    # Create a fake test browser/client for Flask.
    client = app.test_client()

    # Call the health endpoint.
    response = client.get("/health")

    # The server should say it is ok.
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_prediction_endpoint_returns_price():
    # Create a fake test browser/client for Flask.
    client = app.test_client()

    # This is the same kind of input the frontend sends to the API.
    payload = {
        "brand": "Toyota",
        "year": 2020,
        "mileage": 42000,
        "engine_size": 2.4,
        "horsepower": 210,
        "fuel_type": "Hybrid",
        "transmission": "Automatic",
        "body_type": "SUV",
        "accident_history": 0,
        "owners": 1,
    }

    # Send the payload to the prediction endpoint.
    response = client.post("/api/predict", json=payload)
    data = response.get_json()

    # The API should return a valid prediction.
    assert response.status_code == 200
    assert data["predicted_price"] > 0
    assert data["best_model"]
    assert len(data["top_impact"]) == 3
