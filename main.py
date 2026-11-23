# ============================================================
# SUPERVISED MACHINE LEARNING MODELS
# Internship Task - Week 2
# ============================================================

import pandas as pd

from sklearn.datasets import (
    fetch_california_housing,
    load_iris
)

from src.preprocessing import (
    prepare_regression_data,
    prepare_classification_data
)

from src.regression import run_regression_models
from src.classification import run_classification_models

# ============================================================
# PROJECT INFORMATION
# ============================================================

print("=" * 65)
print("       SUPERVISED MACHINE LEARNING PROJECT")
print("=" * 65)

print("\nProject: Supervised Machine Learning Models")
print("Library: Scikit-learn")
print("Language: Python")


# ============================================================
# STEP 1: LOAD REGRESSION DATASET
# ============================================================

print("\n" + "=" * 65)
print("STEP 1: LOADING REGRESSION DATASET")
print("=" * 65)

# Load California Housing dataset
housing = fetch_california_housing()

# Features
X_reg = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

# Target
y_reg = pd.Series(
    housing.target,
    name="HousePrice"
)

print("\nDataset Name: California Housing")
print("Dataset Shape:", X_reg.shape)

print("\nFirst Five Records:")
print(X_reg.head())

print("\nDataset Information:")
X_reg.info()

print("\nMissing Values:")
print(X_reg.isnull().sum())

print("\nStatistical Description:")
print(X_reg.describe())


# ============================================================
# STEP 2: LOAD CLASSIFICATION DATASET
# ============================================================

print("\n" + "=" * 65)
print("STEP 2: LOADING CLASSIFICATION DATASET")
print("=" * 65)

# Load Iris dataset
iris = load_iris()

# Features
X_clf = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Target
y_clf = pd.Series(
    iris.target,
    name="Species"
)

print("\nDataset Name: Iris")
print("Dataset Shape:", X_clf.shape)

print("\nFirst Five Records:")
print(X_clf.head())

print("\nTarget Classes:")
print(iris.target_names)

print("\nClass Distribution:")
print(y_clf.value_counts().sort_index())

print("\nMissing Values:")
print(X_clf.isnull().sum())

print("\nStatistical Description:")
print(X_clf.describe())


# ============================================================
# STEP 3: REGRESSION DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 65)
print("STEP 3: REGRESSION DATA PREPROCESSING")
print("=" * 65)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = (
    prepare_regression_data(X_reg, y_reg)
)

print("\nRegression Training Features:", X_train_reg.shape)
print("Regression Testing Features:", X_test_reg.shape)

print("Regression Training Target:", y_train_reg.shape)
print("Regression Testing Target:", y_test_reg.shape)


# ============================================================
# STEP 4: CLASSIFICATION DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 65)
print("STEP 4: CLASSIFICATION DATA PREPROCESSING")
print("=" * 65)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = (
    prepare_classification_data(X_clf, y_clf)
)

print("\nClassification Training Features:", X_train_clf.shape)
print("Classification Testing Features:", X_test_clf.shape)

print("Classification Training Target:", y_train_clf.shape)
print("Classification Testing Target:", y_test_clf.shape)


# ============================================================
# STEP 5: FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("              PROJECT PREPROCESSING SUMMARY")
print("=" * 65)

print("\nREGRESSION DATASET")
print("Dataset: California Housing")
print("Total Records:", len(X_reg))
print("Training Records:", len(X_train_reg))
print("Testing Records:", len(X_test_reg))
print("Total Features:", X_reg.shape[1])

print("\nCLASSIFICATION DATASET")
print("Dataset: Iris")
print("Total Records:", len(X_clf))
print("Training Records:", len(X_train_clf))
print("Testing Records:", len(X_test_clf))
print("Total Features:", X_clf.shape[1])
print("Number of Classes:", len(iris.target_names))

print("\n" + "=" * 65)
print("DATASET LOADING AND PREPROCESSING COMPLETED!")
print("=" * 65)
print("Ready for Machine Learning Model Training.")
print("=" * 65)

print("\nSTEP 5: REGRESSION MODEL TRAINING")
regression_results = run_regression_models(
    X_train_reg,
    X_test_reg,
    y_train_reg,
    y_test_reg
)

print("\n" + "=" * 65)
print("STEP 6: CLASSIFICATION MODEL TRAINING")
print("=" * 65)

classification_results = run_classification_models(
    X_train_clf,
    X_test_clf,
    y_train_clf,
    y_test_clf,
    class_names=iris.target_names
)

print("\n" + "=" * 65)
print("CLASSIFICATION TRAINING COMPLETED")
print("=" * 65)

print(classification_results.to_string(index=False))

print("\n" + "=" * 65)
print("SUPERVISED MACHINE LEARNING PROJECT COMPLETED!")
print("=" * 65)