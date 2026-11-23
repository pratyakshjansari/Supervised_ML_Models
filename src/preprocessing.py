
# ==========================================
# DATA PREPROCESSING MODULE
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def prepare_regression_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nREGRESSION DATA SPLIT")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    return X_train, X_test, y_train, y_test


def prepare_classification_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nCLASSIFICATION DATA SPLIT")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    return X_train, X_test, y_train, y_test


def create_scaled_pipeline(model):

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    return pipeline


def create_tree_pipeline(model):

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model)
    ])

    return pipeline
