"""
This script applies Lasso regression (L1 regularization) to the scikit-learn 
diabetes dataset to predict diabetes progression. It performs feature selection 
to identify the most important predictors and evaluates model performance using
Mean Absolute Error (MAE). A residual plot is generated to visualize prediction 
errors and saved for analysis.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error

# Create output directory if it doesn't exist
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Load and split data
try:
    data = load_diabetes()  # Load the data and assign to 'data'
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
except Exception as e:
    print(f"Error loading or splitting data: {e}")
    exit(1) 

# Apply Lasso Regression with feature selection
try:
    lasso = Lasso(alpha=0.1, max_iter=100000).fit(X_train, y_train)
    model = SelectFromModel(lasso, prefit=True)
    X_new = model.transform(X)
    lasso_new = Lasso(alpha=0.1, max_iter=100000).fit(X_new, y)
except Exception as e:
    print(f"Error during Lasso regression: {e}")
    exit(1)

# Evaluate on test set
try:
    y_pred = lasso_new.predict(model.transform(X_test))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model Mean Absolute Error (with feature selection): {mae:.2f}")
    print(
        "Selected features:",
        np.array(data.feature_names)[model.get_support()],  # Use data.feature_names
    )
except Exception as e:
    print(f"Error during model evaluation: {e}")
    exit(1)

# Create residual plot with explanations
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test - y_pred)
plt.axhline(y=0, color="r", linestyle="-")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot for Lasso Regression\nExplanations:")

# Add plot explanations with automatic wrapping
plt.text(
    0.05,
    0.95,
    "The residual plot helps assess model fit.\nIdeally, residuals should be "
    "randomly scattered around zero with no clear patterns.",
    transform=plt.gca().transAxes,
    horizontalalignment="left",
    verticalalignment="top",
    wrap=True,
)

# Save plot
try:
    plt.savefig(os.path.join(output_dir, "lasso_residual_plot.png"))
    print("Residual plot saved successfully.")
except Exception as e:
    print(f"Error saving residual plot: {e}")
