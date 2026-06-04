from pathlib import Path

import numpy as np
import pandas as pd


BRANDS = ["Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes", "Hyundai", "Kia"]
FUEL_TYPES = ["Gasoline", "Hybrid", "Diesel", "Electric"]
TRANSMISSIONS = ["Automatic", "Manual"]
BODY_TYPES = ["Sedan", "SUV", "Hatchback", "Truck", "Coupe"]


def generate_car_price_data(rows: int = 1800, random_state: int = 42) -> pd.DataFrame:
    # The random generator makes fake data.
    # The random_state keeps the results reproducible.
    rng = np.random.default_rng(random_state)

    # Create random car information.
    # The probabilities make common cars appear more often than rare cars.
    brand = rng.choice(BRANDS, size=rows, p=[0.18, 0.16, 0.14, 0.12, 0.12, 0.1, 0.1, 0.08])
    year = rng.integers(2005, 2025, size=rows)

    # Mileage is generated around an average, then clipped to realistic limits.
    mileage = np.clip(rng.normal(70000, 38000, size=rows), 2000, 220000).round()
    engine_size = np.clip(rng.normal(2.4, 0.9, size=rows), 1.0, 6.5).round(1)
    horsepower = np.clip(engine_size * rng.normal(88, 14, size=rows), 90, 520).round()
    fuel_type = rng.choice(FUEL_TYPES, size=rows, p=[0.68, 0.15, 0.12, 0.05])
    transmission = rng.choice(TRANSMISSIONS, size=rows, p=[0.78, 0.22])
    body_type = rng.choice(BODY_TYPES, size=rows, p=[0.32, 0.33, 0.16, 0.14, 0.05])
    accident_history = rng.choice([0, 1], size=rows, p=[0.82, 0.18])
    owners = rng.choice([1, 2, 3, 4], size=rows, p=[0.42, 0.34, 0.18, 0.06])

    # These dictionaries simulate how different categories affect car price.
    # Example: luxury brands usually cost more than economy brands.
    brand_value = {
        "Toyota": 2600,
        "Honda": 2300,
        "Ford": 1200,
        "Chevrolet": 900,
        "BMW": 8200,
        "Mercedes": 9000,
        "Hyundai": 500,
        "Kia": 300,
    }
    fuel_value = {"Gasoline": 0, "Hybrid": 2200, "Diesel": 900, "Electric": 5400}
    transmission_value = {"Automatic": 900, "Manual": -800}
    body_value = {"Sedan": 0, "SUV": 3000, "Hatchback": -700, "Truck": 4200, "Coupe": 1800}

    # Age is one of the strongest factors in car price.
    age = 2025 - year
    base_price = 31500

    # These helper variables create non-linear behavior.
    # That gives tree models something more interesting to learn.
    premium_brand = np.isin(brand, ["BMW", "Mercedes"]).astype(int)
    suv_or_truck = np.isin(body_type, ["SUV", "Truck"]).astype(int)
    high_mileage_penalty = np.maximum(mileage - 95000, 0) * 0.045

    # This formula creates the target value: price.
    # It is fake data, but the logic tries to act like real used-car pricing.
    price = (
        base_price
        - np.power(age, 1.18) * 1150
        - mileage * 0.075
        - high_mileage_penalty
        + engine_size * 1900
        + horsepower * 52
        - accident_history * 4300
        - (owners - 1) * 1200
        + premium_brand * horsepower * 18
        + suv_or_truck * engine_size * 650
        - accident_history * age * 210
        + np.vectorize(brand_value.get)(brand)
        + np.vectorize(fuel_value.get)(fuel_type)
        + np.vectorize(transmission_value.get)(transmission)
        + np.vectorize(body_value.get)(body_type)
        + rng.normal(0, 2600, size=rows)
    )

    # Keep prices inside a realistic range.
    price = np.clip(price, 2500, 95000).round(2)

    # Return everything as a pandas table.
    return pd.DataFrame(
        {
            "brand": brand,
            "year": year,
            "mileage": mileage.astype(int),
            "engine_size": engine_size,
            "horsepower": horsepower.astype(int),
            "fuel_type": fuel_type,
            "transmission": transmission,
            "body_type": body_type,
            "accident_history": accident_history,
            "owners": owners,
            "price": price,
        }
    )


def ensure_dataset(path: str | Path = "data/car_prices.csv") -> Path:
    # This function makes sure the dataset exists before training starts.
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        # If there is no CSV yet, generate one and save it.
        data = generate_car_price_data()
        data.to_csv(dataset_path, index=False)

    # Return the path so the training code can read the file.
    return dataset_path
