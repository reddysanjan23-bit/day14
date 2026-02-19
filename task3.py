import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# Feature (non-linear relationship)
X = np.linspace(-10, 10, 100).reshape(-1, 1)

# Target with curve
y = X**2 + np.random.normal(0, 10, size=(100, 1))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Predictions
y_pred_linear = lin_reg.predict(X)

# R² score
r2_linear = r2_score(y, y_pred_linear)

print("R² (Original Features):", r2_linear)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X)

print("Original shape:", X.shape)
print("Polynomial shape:", X_poly.shape)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

# Predictions
y_pred_poly = lin_reg_poly.predict(X_poly)

# R² score
r2_poly = r2_score(y, y_pred_poly)

print("R² (Polynomial Features):", r2_poly)

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(X, y, label="Actual Data")

# Linear line
plt.plot(X, y_pred_linear, label="Linear Fit")

# Polynomial curve
plt.plot(X, y_pred_poly, label="Polynomial Fit")

plt.legend()
plt.title("Linear vs Polynomial Regression")
plt.show()