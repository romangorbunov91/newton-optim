# version 0.0.1 by romangorbunov91
# 19-Sep-2025

import numpy as np

def f_trend(x):
    return x**3 - 3*x**2 + 2*x - 5

def predict(X, w_coeff):
    return X @ w_coeff

def loss_func(X, y, w_coeff):
    residuals = X @ w_coeff - y
    return 0.5 * np.sum(residuals**2) / len(y)

def grad_func(X, y, w_coeff):
    residuals = X @ w_coeff - y
    return (X.T @ residuals) / len(y)

def hess_func(X):
    return (X.T @ X) / X.shape[0]

def jacob_func(X):
    return X

# delete.
def polynomial_features(x, degree):
    return np.vander(x, degree + 1, increasing=True)