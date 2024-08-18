"""
This script trains a K-Nearest Neighbors (KNN) regressor to predict diabetes progression 
based on various features and produces a heatmap revealing correlations between patient 
features and disease progression.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def load_diabetes_data():
    """Loads the diabetes dataset from scikit-learn."""
    data = load_diabetes()
    X = data["data"]
    y = data["target"]
    feature_names = data["feature_names"]
    return X, y, feature_names


def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_knn_regressor(X_train, y_train):
    """Trains a K-Nearest Neighbors regressor."""
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates the model using R-squared score."""
    score = model.score(X_test, y_test)
    print("Model R-squared Score:", round(score, 2))
    return score


def create_correlation_heatmap(X, y, feature_names):
    """Creates and displays a heatmap of feature correlations."""
    column_data = np.concatenate([X, y[:, None]], axis=1)
    column_names = np.concatenate([feature_names, ["diabetes_progression"]])
    df = pd.DataFrame(column_data, columns=column_names)

    sns.heatmap(df.corr(), cmap="coolwarm", annot=True, annot_kws={"size": 8})
    plt.tight_layout()
    plt.title("Feature Correlation Heatmap (Diabetes)")
    plt.show()


def main():
    X, y, feature_names = load_diabetes_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_knn_regressor(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    create_correlation_heatmap(X, y, feature_names)

if __name__ == "__main__":
    main()
