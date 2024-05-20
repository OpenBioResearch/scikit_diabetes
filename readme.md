# scikit-learn-diabetes

## Overview

This project uses Scikit-learn to explore and model the diabetes dataset available within the library. The primary goal is to build a predictive model for diabetes progression using the K-Nearest Neighbors algorithm. The project also includes an analysis of feature correlations to understand the relationships between different variables and the target outcome.

## Usage
The diabetes_sklearn.py script performs the following steps:

Loads the diabetes dataset: Utilizes the built-in load_diabetes function from scikit-learn.
Explores the data: Prints the dataset's keys, feature names, and a sample of target values (diabetes progression).
Splits the data: Divides the dataset into training (80%) and testing (20%) sets.
Builds a model: Trains a K-Nearest Neighbors regression model.
Evaluates the model: Calculates and prints the R-squared score on the test set.
Analyzes correlations: Creates a correlation matrix between features and visualizes it using a heatmap.

## Dependencies
scikit-learn
pandas
seaborn
matplotlib
numpy

## Dataset
The project uses the diabetes dataset from the scikit-learn library. This dataset includes:
Ten baseline variables: age, sex, body mass index, average blood pressure, and six blood serum measurements. A quantitative measure of disease progression one year after baseline.

## Model
The project uses a K-Nearest Neighbors regression model. This model predicts the target value (diabetes progression) by averaging the values of the closest neighbors in the feature space. The number of neighbors used is determined by the algorithm's default settings.

## Results
The script outputs the R-squared score of the model on the test set and displays a heatmap visualizing the correlations between the features.

## License
This project is licensed under the BSD 3-Clause

## Project Structure

* **.venv:** A virtual environment containing project dependencies.
* **diabetes_sklearn.py:** The main Python script for data loading, exploration, model training, evaluation, and correlation visualization.
* **.gitignore:**  Specifies files and folders to be excluded from version control (Git).
* **README.md:** This documentation file.
* **requirements.txt:** Specifies python project library dependencies.

## Getting Started

1. **Clone the Repository:** 
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/scikit-learn-diabetes.git
