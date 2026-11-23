
# Supervised Machine Learning Models

## Project Overview

This project implements and compares supervised machine learning algorithms using Python and Scikit-learn.

It covers both regression and classification tasks, including model training, prediction, evaluation, and performance comparison.

## Technologies Used

- Python 3.13
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

## Datasets

### 1. California Housing Dataset
- Task: Regression
- Records: 20,640
- Features: 8
- Target: House value

### 2. Iris Dataset
- Task: Classification
- Records: 150
- Features: 4
- Classes: Setosa, Versicolor, Virginica

## Machine Learning Algorithms

### Regression
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

### Classification
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)

## Evaluation Metrics

### Regression
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Classification
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Regression Results

| Model | RMSE | R² Score |
|---|---:|---:|
| Linear Regression | 0.7456 | 0.5758 |
| Decision Tree | 0.7030 | 0.6228 |
| Random Forest | 0.5043 | 0.8059 |

Best regression model: Random Forest.

## Classification Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 93.33% |
| Decision Tree | 93.33% |
| Random Forest | 90.00% |
| KNN | 93.33% |

Logistic Regression and Decision Tree achieved the highest accuracy.

## Project Structure

```text
Supervised_ML_Models/
├── src/
│   ├── preprocessing.py
│   ├── regression.py
│   └── classification.py
├── results/
│   ├── graphs/
│   ├── models/
│   ├── regression_results.csv
│   └── classification_results.csv
├── reports/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd "Supervised_ML_Models task 2"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## Project Objective

The objective is to understand supervised machine learning techniques, implement different algorithms, evaluate their performance, and compare their results using appropriate metrics.

## Author

Machine Learning Enthusiast
