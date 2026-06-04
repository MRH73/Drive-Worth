from src.training import train_and_save


if __name__ == "__main__":
    # Train all models and save the best one.
    bundle = train_and_save()

    # Print the winner in the terminal.
    print(f"Best model: {bundle['best_model_name']}")
    print(f"Data source: {bundle['data_source']}")
    print(f"Rows used: {bundle['row_count']}")

    # Print the score for the Linear Regression model.
    for result in bundle["metrics"]:
        print(
            f"{result['name']}: "
            f"MAE=${result['mae']:,.2f}, "
            f"RMSE=${result['rmse']:,.2f}, "
            f"R2={result['r2']:.4f}"
        )
