# Drive Worth

Drive Worth is a simple machine learning web app that predicts a used car price using Linear Regression trained from scratch with NumPy.

The goal of this project is to show the full basic ML flow without extra complexity:

```text
real CSV data -> NumPy training -> saved model parameters -> Flask API -> web UI
```

## What It Does

- Loads real used-car data from a public CSV.
- Keeps only three columns: `year`, `mileage`, and `price`.
- Trains Linear Regression with NumPy and gradient descent.
- Shows the cost function on the web page.
- Shows the final training cost and simple error metrics.
- Lets a user enter year and mileage through a web form.
- Sends the form data to a Flask API.
- Returns and displays a predicted car price.

## Data Source

The app uses this public CSV when available:

```text
https://gist.githubusercontent.com/harshitmanda35/888474a03966b678294d2dfb29bff888/raw/car_price_prediction.csv
```

The original CSV has more fields, but this beginner version only uses:

- `Year`
- `Mileage`
- `Price`

The cleaned copy is saved here:

```text
data/car_prices.csv
```

If the CSV cannot be downloaded, the app uses a tiny local fallback sample so the project still runs.

## How The Model Works

This project uses a simple linear equation:

```text
predicted_price = w1 * year + w2 * mileage + b
```

Because year and mileage have very different sizes, the code scales them before training.

The cost function is:

```text
J(w, b) = (1 / 2m) * sum((prediction - actual_price)^2)
```

The model starts with weights equal to zero. During training, gradient descent updates the weights many times to reduce the cost.

## Tech Stack

- **Python**: main language for backend and ML code.
- **NumPy**: trains Linear Regression from scratch.
- **pandas**: loads and cleans the CSV data.
- **Flask**: serves the web page and prediction API.
- **HTML/CSS/JavaScript**: builds the browser interface.
- **Chart.js**: draws the training cost chart.

No Docker. No tests. No scikit-learn. This version is intentionally small.

## Project Structure

```text
.
├── app.py
├── train.py
├── requirements.txt
├── data/
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

Open:

```text
http://127.0.0.1:5000
```

## API Example

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2020,
    "mileage": 42000
  }'
```

Example response:

```json
{
  "cost_function": "J(w, b) = (1 / 2m) * sum((prediction - actual_price)^2)",
  "final_cost": 371851013.5,
  "metrics": {
    "mae": 23666.37,
    "rmse": 27270.9,
    "r2": 0.0014
  },
  "model_name": "NumPy Linear Regression",
  "predicted_price": 51719.31
}
```

## Portfolio Talking Points

- Built Linear Regression from scratch using NumPy.
- Used gradient descent to minimize a cost function.
- Connected a trained ML model to a Flask API.
- Built a simple web UI for live predictions.
- Used real CSV data and documented the data source.
- Kept the project beginner-friendly and easy to explain.
