import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

from src.preprocessing import (
    create_scaled_pipeline,
    create_tree_pipeline
)


def run_classification_models(
    X_train,
    X_test,
    y_train,
    y_test,
    class_names=None
):

    print("\n" + "=" * 65)
    print("             CLASSIFICATION MODEL TRAINING")
    print("=" * 65)

    os.makedirs("results/graphs", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    if class_names is None:
        class_names = sorted(pd.Series(y_train).unique())

    models = {
        "Logistic Regression": create_scaled_pipeline(
            LogisticRegression(max_iter=1000, random_state=42)
        ),

        "Decision Tree": create_tree_pipeline(
            DecisionTreeClassifier(random_state=42)
        ),

        "Random Forest": create_tree_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        ),

        "K-Nearest Neighbors": create_scaled_pipeline(
            KNeighborsClassifier(n_neighbors=5)
        )
    }

    results = []
    predictions_store = {}
    report_text = []

    for name, model in models.items():

        print("\n" + "-" * 65)
        print("Training Model:", name)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        predictions_store[name] = predictions

        filename = name.lower().replace(" ", "_")
        joblib.dump(
            model,
            f"results/models/{filename}.pkl"
        )

        model_report = classification_report(
            y_test,
            predictions,
            labels=range(len(class_names)),
            target_names=[str(x) for x in class_names],
            zero_division=0
        )

        report_text.append(
            f"\n{'=' * 65}\n"
            f"MODEL: {name}\n"
            f"{'=' * 65}\n"
            f"{model_report}\n"
        )

        print("Model trained successfully!")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        print("\nClassification Report:")
        print(model_report)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    print("\n" + "=" * 65)
    print("         CLASSIFICATION MODEL COMPARISON")
    print("=" * 65)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results_df.to_csv(
        "results/classification_results.csv",
        index=False
    )

    # Model comparison graph
    plt.figure(figsize=(11, 6))

    plt.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )

    plt.title("Classification Model Comparison - Accuracy")
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        "results/graphs/classification_comparison.png",
        dpi=300
    )

    plt.close()

    # Confusion matrices for all four models
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (name, predictions) in zip(
        axes.flatten(),
        predictions_store.items()
    ):

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=class_names,
            cmap="Blues",
            ax=ax,
            colorbar=False
        )

        ax.set_title(name)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/confusion_matrices.png",
        dpi=300
    )

    plt.close()

    # Save detailed classification reports
    with open(
        "reports/classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SUPERVISED MACHINE LEARNING PROJECT\n"
            "CLASSIFICATION MODEL EVALUATION\n"
        )

        for report in report_text:
            file.write(report)

    best_model = results_df.iloc[0]

    print("\n" + "=" * 65)
    print("BEST CLASSIFICATION MODEL")
    print("=" * 65)

    print("Model:", best_model["Model"])
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"F1 Score: {best_model['F1 Score']:.4f}")

    print("\nClassification results saved successfully!")
    print("CSV: results/classification_results.csv")
    print("Comparison Graph: results/graphs/classification_comparison.png")
    print("Confusion Matrices: results/graphs/confusion_matrices.png")
    print("Detailed Report: reports/classification_report.txt")
    print("Models: results/models/")

    return results_df
