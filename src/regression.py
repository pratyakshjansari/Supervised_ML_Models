
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.preprocessing import (
    create_scaled_pipeline,
    create_tree_pipeline
)


def run_regression_models(X_train, X_test, y_train, y_test):

    print("\n" + "=" * 65)
    print("             REGRESSION MODEL TRAINING")
    print("=" * 65)

    os.makedirs("results/graphs", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    models = {
        "Linear Regression": create_scaled_pipeline(
            LinearRegression()
        ),

        "Decision Tree": create_tree_pipeline(
            DecisionTreeRegressor(random_state=42)
        ),

        "Random Forest": create_tree_pipeline(
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    }

    results = []

    for name, model in models.items():

        print("\n" + "-" * 65)
        print("Training Model:", name)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        })

        filename = name.lower().replace(" ", "_")
        joblib.dump(model, f"results/models/{filename}.pkl")

        print("Model trained successfully!")
        print(f"MAE:      {mae:.4f}")
        print(f"MSE:      {mse:.4f}")
        print(f"RMSE:     {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="RMSE",
        ascending=True
    )

    print("\n" + "=" * 65)
    print("          REGRESSION MODEL COMPARISON")
    print("=" * 65)

    print(results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    ))

    results_df.to_csv(
        "results/regression_results.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))
    plt.bar(results_df["Model"], results_df["RMSE"])
    plt.title("Regression Model Comparison - RMSE")
    plt.xlabel("Machine Learning Models")
    plt.ylabel("RMSE (Lower is Better)")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        "results/graphs/regression_comparison.png",
        dpi=300
    )
    plt.close()

    best_model = results_df.iloc[0]

    print("\n" + "=" * 65)
    print("BEST REGRESSION MODEL")
    print("=" * 65)

    print("Model:", best_model["Model"])
    print(f"RMSE: {best_model['RMSE']:.4f}")
    print(f"R2 Score: {best_model['R2 Score']:.4f}")

    print("\nRegression results saved successfully!")
    print("CSV: results/regression_results.csv")
    print("Graph: results/graphs/regression_comparison.png")
    print("Models: results/models/")

    return results_df
