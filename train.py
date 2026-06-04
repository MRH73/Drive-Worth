from src.training import train_and_save


if __name__ == "__main__":
    # Train all models and save the best one.
    bundle = train_and_save()

    # Print the winner in the terminal.
    print(f"Best model: {bundle['best_model_name']}")

    # Print the score for every model so we can compare them.
    for result in bundle["metrics"]:
        print(
            f"{result['name']}: "
            f"MAE=${result['mae']:,.2f}, "
            f"RMSE=${result['rmse']:,.2f}, "
            f"R2={result['r2']:.4f}"
        )
