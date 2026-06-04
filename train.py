from src.training import train_and_save


if __name__ == "__main__":
    # Train all models and save the best one.
    bundle = train_and_save()

    # Print the winner in the terminal.
    print(f"Model: {bundle['model_name']}")
    print(f"Data source: {bundle['data_source']}")
    print(f"Rows used: {bundle['row_count']}")
    print(f"Final cost: ${bundle['final_cost']:,.2f}")
    print(
        f"Metrics: "
        f"MAE=${bundle['metrics']['mae']:,.2f}, "
        f"RMSE=${bundle['metrics']['rmse']:,.2f}, "
        f"R2={bundle['metrics']['r2']:.4f}"
    )
