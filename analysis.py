import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset (Boston Housing dataset alternative)
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data["price"] = housing.target

# Basic exploration
print("Dataset Info:")
print(data.head())

print("\nSummary Statistics:")
print(data.describe())

# Visualization (relationship with price)
data.corr()["price"].sort_values(ascending=False).plot(kind="bar")
plt.title("Feature Correlation with House Price")
plt.ylabel("Correlation")
plt.show()

# Machine Learning Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Features and target
X = data.drop("price", axis=1)
y = data["price"]

# Split data (train/test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

print("\nModel Coefficients:")
print(pd.Series(model.coef_, index=X.columns).sort_values(ascending=False))
