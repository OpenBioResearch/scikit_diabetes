"""
This script applies Lasso regression (L1 regularization) to the scikit-learn 
diabetes dataset to predict diabetes progression. It performs feature selection 
to identify the most important predictors and evaluates model performance using
Mean Absolute Error (MAE). A residual plot for lasso regression should show no 
clear pattern if the model fits well. 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error

def create_output_directory(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_and_split_data(test_size=0.2, random_state=42):
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return data, X_train, X_test, y_train, y_test

def perform_lasso_regression(X_train, y_train, alpha=0.1, max_iter=100000):
    lasso = Lasso(alpha=alpha, max_iter=max_iter)
    lasso.fit(X_train, y_train)
    return lasso

def select_important_features(lasso, X_train, X_test, feature_names):
    model = SelectFromModel(lasso, prefit=True)
    X_train_new = model.transform(X_train)
    X_test_new = model.transform(X_test)
    selected_features = feature_names[model.get_support()]
    return X_train_new, X_test_new, selected_features

def evaluate_model(lasso, X_test_new, y_test):
    y_pred = lasso.predict(X_test_new)
    mae = mean_absolute_error(y_test, y_pred)
    return y_pred, mae

def plot_residuals(y_pred, y_test, mae, selected_features, output_dir):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred, y_test - y_pred)
    plt.axhline(y=0, color="r", linestyle="-")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot for Lasso Regression")

    textstr = f"MAE: {mae:.2f}\nSelected Features: {', '.join(selected_features)}"
    plt.figtext(
        0.5, 
        0.01, 
        textstr,
        horizontalalignment="center",  
        fontsize=12,  
        weight='bold',
        bbox=dict(facecolor='lightblue', alpha=0.5), 
        wrap=True, 
    )
    plt.figtext(
        0.5,
        -0.05,
        "The residual plot helps assess model fit. Ideally, residuals should be "
        "randomly scattered around zero with no clear patterns.",
        horizontalalignment="center",  
        fontsize=10,  
        wrap=True, 
    )
    plt.savefig(os.path.join(output_dir, "lasso_residual_plot.png"))
    plt.show()

def main():
    output_dir = create_output_directory()
    
    try:
        data, X_train, X_test, y_train, y_test = load_and_split_data()
    except Exception as e:
        print(f"Error loading or splitting data: {e}")
        return

    try:
        lasso = perform_lasso_regression(X_train, y_train)
        X_train_new, X_test_new, selected_features = select_important_features(lasso, X_train, X_test, np.array(data.feature_names))
        lasso_new = perform_lasso_regression(X_train_new, y_train)
    except Exception as e:
        print(f"Error during Lasso regression: {e}")
        return

    try:
        y_pred, mae = evaluate_model(lasso_new, X_test_new, y_test)
        print(f"Model Mean Absolute Error (with feature selection): {mae:.2f}")
        print("Selected features:", selected_features)
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return

    try:
        plot_residuals(y_pred, y_test, mae, selected_features, output_dir)
        print("Residual plot saved successfully.")
    except Exception as e:
        print(f"Error saving residual plot: {e}")

if __name__ == "__main__":
    main()