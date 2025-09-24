# version 1.0.0 by romangorbunov91
# 24-Sep-2025

import numpy as np

def f_trend(x):
    return x**3 - 3*x**2 + 2*x - 5

def predict(X, w_coeff):
    return X @ w_coeff

def loss_func(X, y, w_coeff):
    residuals = predict(X, w_coeff) - y
    return 0.5 * np.sum(residuals**2) / X.shape[0]

def grad_func(X, y, w_coeff):
    residuals = predict(X, w_coeff) - y
    return (X.T @ residuals) / X.shape[0]

def hess_func(X):
    return (X.T @ X) / X.shape[0]

def jacob_func(X):
    return X