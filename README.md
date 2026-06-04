# Drive Worth

Drive Worth is a beginner-friendly machine learning web app that predicts a used car price with Linear Regression.

The project has a complete flow:

```text
CSV data
-> pandas cleaning
-> scikit-learn preprocessing
-> Linear Regression training
-> saved model
-> Flask API
-> HTML/CSS/JavaScript UI
```

## What The App Does

- Loads used-car data from a public CSV when possible.
- Cleans the data into simple columns.
- Trains one Linear Regression model.
- Uses brand, model, year, mileage, fuel type, transmission, and condition.
- Skips engine size and horsepower to keep the project simpler.
- Saves the trained model with Joblib.
- Serves predictions through a Flask API.
- Shows a web form with brand and model dropdowns.
- Shows which features had the biggest impact on the prediction.
- Includes simple API tests.

## Data Source

The first version of this project used generated fake data from `src/data.py`.

The current version tries to use this public CSV first:

```text
https://gist.githubusercontent.com/harshitmanda35/888474a03966b678294d2dfb29bff888/raw/car_price_prediction.csv
```

This CSV includes:

- Brand
- Model
- Year
- Mileage
- Fuel Type
- Transmission
- Condition
- Price

There is also a stronger recommended dataset for a future upgrade:

```text
https://www.opendatabay.com/data/consumer/7d8bf56f-b0e3-42e9-a9d3-a0ed4f3896e9
```

That dataset is described as coming from cars.com, with 4,009 vehicle listings, 57 brands, 1,898 models, model years from 1974 to 2024, and a CC BY 4.0 license. It is better for a serious portfolio version, but the direct CSV download is not as simple as the GitHub Gist source.

## Important Data Note

The current public GitHub CSV is easy to download, but its Linear Regression score is weak:

```text
MAE:  $23,868.80
RMSE: $27,791.13
R2:   -0.0195
```

That means the model is not very accurate with this dataset and the simplified features.

This is still useful for learning because it shows an honest machine learning lesson: simple models are easier to understand, but they need good data and useful features. Removing engine size and horsepower also removes information that may help the model.

## Features Used By The Model

The model uses:

- `brand`
- `model`
- `year`
- `mileage`
- `fuel_type`
- `transmission`
- `condition`

The model does not use:

- `engine_size`
- `horsepower`
- `body_type`
- `accident_history`
- `owners`

## Why Linear Regression

Linear Regression predicts a number by learning weighted relationships between inputs and the target.

In this project, the target is:

```text
price
```

Example idea:

```text
newer year -> higher predicted price
higher mileage -> lower predicted price
premium brand/model -> different predicted price
```

Text values like brand and model cannot go directly into Linear Regression. The app uses OneHotEncoder to convert categories into numeric 0/1 columns first.

## Technology Stack

- **Python**: main language for backend and machine learning.
- **pandas**: loads, cleans, and saves the CSV data.
- **scikit-learn**: builds the preprocessing pipeline and trains Linear Regression.
- **Flask**: creates the web server and prediction API.
- **HTML**: creates the form and page structure.
- **CSS**: styles the page.
- **JavaScript**: updates the model dropdown and sends prediction requests.
- **Chart.js**: displays metric and feature-impact charts.
- **Joblib**: saves and loads the trained model.
- **Pytest**: tests the Flask API.
- **Docker**: packages the app for deployment.

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
    ├── conftest.py
    └── test_api.py
```

## Run From Zero

Use Python 3.11 or 3.12.

Go to the project folder:

```bash
cd "/Users/miguelruiz/Documents/Me/MLProject/Drive Worth"
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Run the Flask app:

```bash
flask --app app run --debug
```

Open this URL:

```text
http://127.0.0.1:5000
```

## API Example

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Toyota",
    "model": "Corolla",
    "year": 2020,
    "mileage": 42000,
    "fuel_type": "Gasoline",
    "transmission": "Automatic",
    "condition": "Used"
  }'
```

## Run Tests

```bash
pytest
```

## Docker

Build the image:

```bash
docker build -t drive-worth .
```

Run the container:

```bash
docker run -p 5000:5000 drive-worth
```

## Portfolio Talking Points

- Built a full-stack machine learning app with Flask and scikit-learn.
- Trained and deployed a Linear Regression model.
- Added data cleaning for a public CSV dataset.
- Added dependent brand and model dropdowns.
- Removed complex vehicle specs to keep the model beginner-friendly.
- Added feature-impact explanation with permutation importance.
- Added API tests and Docker support.

## Future Improvements

- Replace the public Gist CSV with the cars.com-based OpenDataBay/Kaggle dataset.
- Add more real-world fields such as accident history and clean title.
- Compare Linear Regression with Ridge, Random Forest, and Gradient Boosting again.
- Add better model validation and data quality checks.
- Deploy the app on Render, Railway, or Fly.io.
