# Drive Worth

Drive Worth is a simple machine learning web app that predicts a used car price with a NumPy regression model and a Flask API.

The goal of this project is to keep the machine learning pipeline easy to understand:

```text
real CSV data -> clean data -> train with NumPy -> save model -> Flask API -> web UI
```

## What It Does

- Loads real car-price data from a public CSV.
- Keeps the original downloaded data in `data/source_car_prices.csv`.
- Keeps the cleaned training data in `data/car_prices.csv`.
- Trains one regression model using NumPy.
- Uses `make`, `model`, `condition`, `year`, and `mileage`.
- Uses nonlinear feature engineering like `log(mileage)` and squared values.
- Shows the cost function, final cost, MAE, and RMSE on the page.
- Lets a user submit car details through a Flask API.
- Shows a prediction curve and plots the entered car as a point.
- Shows which input features affect the model most.
- Lets users search and page through the cleaned training rows in the browser.

## Data Source

The app uses this public CSV:

```text
https://raw.githubusercontent.com/stat-lu/dataviz/main/data/us_cars.csv
```

The raw CSV includes these useful columns:

- `brand`
- `model`
- `year`
- `mileage`
- `title_status`
- `price`

The cleaned training CSV uses matching project-friendly names:

- `make`
- `model`
- `year`
- `mileage`
- `condition`
- `price`

The project uses `title_status` as the app's `condition` field because it tells whether the car is a `Clean Vehicle` or `Salvage Insurance` vehicle.

The dropdowns for make, model, and condition are created from the cleaned CSV. That means the UI options come from the same data used to train the model.

## Important Data Note

The original source has some rows that are not helpful for a beginner regression model, such as `$0` prices and extreme mileage outliers. The data cleaner removes those rows before training.

In this dataset:

- Rows used: `2385`
- Makes: `25`
- Models: `120`
- Minimum year: `2001`
- Maximum year: `2020`
- Minimum mileage: `122`
- Maximum mileage: `240740`
- Conditions: `Clean Vehicle`, `Salvage Insurance`

This source works better than the first dataset because price has a real relationship with year and mileage. Newer cars usually cost more, and higher-mileage cars usually cost less.

## Data Expansion Note

I checked for larger public used-car datasets. A stronger candidate is the Kaggle/Gigasheet Used Car Price Prediction Dataset, which has `4009` rows and `13` columns, including brand, model, year, mileage, fuel type, engine, transmission, accident history, title status, and price.

I did not wire it in automatically because Kaggle/Gigasheet download access is not a stable raw CSV link like the current GitHub CSV. For a beginner-friendly project, the current source is better because it downloads directly with `pandas.read_csv()` and does not require account setup.

If you manually download that larger `used_cars.csv`, the project can be expanded later by adapting `src/data.py` to merge it with the current cleaned CSV.

## How The Model Works

The project still uses Linear Regression, but it gives the model smarter input columns.

Base inputs from the form:

- `make`
- `model`
- `condition`
- `year`
- `mileage`

Engineered numeric inputs:

- `year`
- `mileage`
- `vehicle_age`
- `vehicle_age_squared`
- `mileage_log`
- `mileage_squared`

This means the final equation is linear in the training columns, but the relationship with the original values is more flexible:

```text
predicted_price =
  w1*year
  + w2*mileage
  + w3*vehicle_age
  + w4*vehicle_age_squared
  + w5*log(mileage)
  + w6*mileage_squared
  + make/model/condition weights
  + bias
```

Why this helps:

- `vehicle_age_squared` lets older cars lose value in a curved way.
- `mileage_log` lets mileage matter strongly at first, then flatten out.
- `mileage_squared` lets very high mileage have a stronger effect.
- One-hot encoding lets text values like `Ford`, `Door`, and `Clean Vehicle` become numbers.

## Cost Function

The cost function is:

```text
J(w, b) = (1 / 2m) * sum((predicted_price - actual_price)^2)
```

Simple explanation:

- The model predicts every car price.
- It subtracts the real price from the predicted price.
- It squares every error.
- It averages the squared errors.
- Lower cost means the model fits the training data better.

The cost number is very large because it uses squared dollar errors. For example, a `$20,000` error becomes `400,000,000` after squaring. That is why MAE and RMSE are easier to understand.

Current training results:

```text
Final cost: $20,308,986.49
MAE:        $4,173.12
RMSE:       $6,358.22
R2:         0.7059
```

## Feature Impact

The feature-impact chart uses a simple permutation idea:

1. Predict normally.
2. Measure the normal error.
3. Shuffle one input column.
4. Predict again.
5. If the error gets worse, that input matters more.

Current feature impact after training:

```text
model     -> $4,482.53
mileage   -> $2,173.38
make      -> $1,829.20
year      -> $242.43
condition -> $170.97
```

All inputs are included in the model equation. In this improved dataset, model, mileage, and make affect the prediction much more than condition.

## Prediction Curve

The UI draws predicted price versus mileage.

The curve changes based on:

- selected make
- selected model
- selected condition
- selected year

When the user submits a car, the app plots that car as a dot on the curve.

## Training Data Viewer

The UI includes an `Explore Training Data` table. It calls:

```text
/api/training-data
```

Optional query parameters:

```text
/api/training-data?page=1&per_page=10&search=ford
```

This returns paginated rows from `data/car_prices.csv`, so users can see the exact data the model learned from.

## Tech Stack

- **Python**: backend and machine learning code.
- **NumPy**: regression training and math.
- **pandas**: CSV loading and cleaning.
- **Flask**: web server and prediction API.
- **HTML/CSS/JavaScript**: browser UI.
- **Chart.js**: charts for the prediction curve and feature impact.



## Project Structure

```text
.
├── app.py
├── train.py
├── requirements.txt
├── data/
│   ├── source_car_prices.csv
│   └── car_prices.csv
├── models/
│   └── linear_model.json
├── src/
│   ├── data.py
│   └── training.py
├── static/
│   ├── css/style.css
│   └── js/app.js
└── templates/
    └── index.html
```

## Run From Zero

Go to the project folder:

```bash
cd "/Users/miguelruiz/Documents/Me/MLProject/Drive Worth"
```

Create a virtual environment with Python 3.12:

```bash
python3.12 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
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

Open the app:

```text
http://127.0.0.1:5000
```

If port `5000` is already being used, run:

```bash
flask --app app run --debug --port 5001
```

Then open:

```text
http://127.0.0.1:5001
```

## API Example

Run this while Flask is running:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "make": "Ford",
    "model": "Door",
    "condition": "Clean Vehicle",
    "year": 2018,
    "mileage": 42000
  }'
```

Example response includes:

```json
{
  "model_name": "NumPy Linear Regression with Nonlinear Features",
  "predicted_price": 20767.49,
  "final_cost": 20308986.49,
  "prediction_point": {
    "mileage": 42000,
    "predicted_price": 20767.49
  },
  "feature_impacts": [
    {
      "feature": "model",
      "impact": 4482.53
    }
  ]
}
```
