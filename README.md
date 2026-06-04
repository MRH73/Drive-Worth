# AutoValue ML

AutoValue ML is an end-to-end regression project that predicts used car prices. It trains several machine learning models, compares their performance, saves the best one, exposes predictions through a Flask API, and shows the result in a simple HTML/CSS/JavaScript interface.

This is meant to be a portfolio project: understandable business problem, complete ML pipeline, clean API, and a UI that recruiters can try.

## What The Project Does

- Generates a realistic used car price dataset if no dataset exists yet.
- Trains multiple regression models with scikit-learn.
- Compares models using MAE, RMSE, and R2.
- Saves the best model with Joblib.
- Serves predictions through a Flask API.
- Shows model metrics and feature impact in the browser.
- Includes tests and Docker support.

## Technology Stack

- **Python**: main programming language for data processing, machine learning, and backend logic.
- **NumPy**: creates numeric data and handles fast math operations.
- **Pandas**: stores the dataset in tables and prepares data for training.
- **Scikit-learn**: trains regression models, preprocesses features, evaluates performance, and calculates feature impact.
- **Flask**: creates the backend web server and prediction API.
- **HTML**: structures the web page.
- **CSS**: styles the interface.
- **JavaScript**: sends form data to the API and updates the page without reloading.
- **Chart.js**: draws model comparison and feature impact charts in the UI.
- **Joblib**: saves the trained model so the app can reuse it without retraining every time.
- **Pytest**: tests the API endpoints.
- **Docker**: packages the app so it can run the same way on another computer or deployment platform.

## Regression Models Used

Regression means the model predicts a number. In this project, the number is the car price.

- **Linear Regression**: learns a straight-line relationship between inputs and price. It is simple, fast, and easy to understand.
- **Ridge Regression**: similar to linear regression, but it controls overly large coefficients. This can make the model more stable.
- **Random Forest Regressor**: builds many decision trees and averages their predictions. It can learn non-linear patterns, such as mileage hurting old cars more than newer cars.
- **Gradient Boosting Regressor**: builds decision trees one after another, where each new tree improves the previous errors. It is often strong for tabular data like vehicle listings.

The project is still regression even when the model is not linear, because every model predicts a continuous numeric value.

## Biggest Impact Explanation

The app uses permutation importance. The idea is simple:

1. The trained model makes predictions normally.
2. One feature is shuffled, for example mileage.
3. If the model performs much worse, that feature had a big impact.

This gives you a clear portfolio-friendly explanation like:

> Mileage, year, and horsepower had the biggest impact on predicted vehicle price.

## Project Structure

```text
.
├── app.py
├── train.py
├── Dockerfile
├── requirements.txt
├── data/
│   └── car_prices.csv
├── models/
│   └── best_model.joblib
├── src/
│   ├── data.py
│   └── training.py
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html
└── tests/
    └── test_api.py
```

## How To Run Locally

Use Python 3.11 or 3.12 for the smoothest setup.

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python train.py
```

Run the Flask app:

```bash
flask --app app run --debug
```

Open:

```text
http://127.0.0.1:5000
```

## API Example

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Toyota",
    "year": 2020,
    "mileage": 42000,
    "engine_size": 2.4,
    "horsepower": 210,
    "fuel_type": "Hybrid",
    "transmission": "Automatic",
    "body_type": "SUV",
    "accident_history": 0,
    "owners": 1
  }'
```

## Run Tests

```bash
pytest
```

## Docker

Build the image:

```bash
docker build -t autovalue-ml .
```

Run the container:

```bash
docker run -p 5000:5000 autovalue-ml
```

## Portfolio Talking Points

- Built an end-to-end regression machine learning pipeline.
- Compared linear and non-linear regression models.
- Used preprocessing pipelines for numeric and categorical features.
- Deployed the trained model through a Flask prediction API.
- Built a browser UI with live predictions and charts.
- Added model interpretability with feature impact.
- Added tests and Docker support for production readiness.
