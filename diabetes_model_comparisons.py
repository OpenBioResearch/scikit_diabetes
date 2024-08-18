"""
This script performs cross-validation on multiple regression models (Linear, 
Ridge, Lasso, SVR) using the scikit-learn diabetes dataset. It evaluates their 
performance using Mean Squared Error (MSE) and generates a table summarizing 
the results, including mean and standard deviation of MSE for each model. The 
table is then saved as a PNG image in the 'outputs' folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR


def create_output_directory(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_data():
    try:
        X, y = load_diabetes(return_X_y=True)
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)


def evaluate_models(models, X, y, cv):
    """Evaluates models using cross-validation """
    results = []
    for model in models:
        mse_scores = -cross_val_score(
            model, X, y, cv=cv, scoring="neg_mean_squared_error"
        )
        results.append(
            {
                "Model": model.__class__.__name__,
                "Mean MSE": mse_scores.mean(),
                "Std MSE": mse_scores.std(),
            }
        )
    return results


def create_results_table(results, output_dir):
    """Generates results table and saves it as png """
    results_df = pd.DataFrame(results)
    results_df = results_df.round(2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table = ax.table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title(
        "Diabetes Progression Prediction Model Comparison\n"
        "(Lower MSE indicates better predictive performance.\n"
        "Linear Regression and Lasso show comparable performance.)",
        y=1.2,
    )

    plt.savefig(
        os.path.join(output_dir, "model_comparison_table.png"),
        bbox_inches="tight",
        dpi=300,
    )
    print("Model comparison table saved successfully.")


def main():
    output_dir = create_output_directory()
    X, y = load_data()

    models = [
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=0.1),
        SVR(kernel="linear"),
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = evaluate_models(models, X, y, kf)
    create_results_table(results, output_dir)


if __name__ == "__main__":
    main()