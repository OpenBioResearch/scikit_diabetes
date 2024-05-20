import numpy as np  # Added import for NumPy
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Diabetes Dataset
data = load_diabetes()

# Explore Data
print(data.keys())
print(data['feature_names'])
print("Target (Diabetes progression):", data['target'][:5]) 

# Split into Training and Testing Sets
X = data['data']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and Fit Model
model = KNeighborsRegressor()
model.fit(X_train, y_train)

# Evaluate Model
score = model.score(X_test, y_test)
print("Model R-squared Score:", score)

# Creating dataframe of correlation between features
column_data = np.concatenate([data['data'], data['target'][:,None]], axis=1)
column_names = np.concatenate([data['feature_names'], ["diabetes_progression"]])

df = pd.DataFrame(column_data, columns=column_names)

# Visualizing Correlations through heatmap
sns.heatmap(df.corr(), cmap="coolwarm", annot=True, annot_kws={"size": 8})
plt.tight_layout()
plt.title("Feature Correlation Heatmap (Diabetes)") 
plt.show()
