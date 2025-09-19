# version 0.0.1 by romangorbunov91
# 19-Sep-2025

import numpy as np

def f_trend(x):
    return x**3 - 3*x**2 + 2*x - 5

def f_poly(X, w_coeff):
    Y = np.zeros_like(X)
    for n, x in enumerate(X):
        for idx, w in enumerate(w_coeff.tolist()[0]):
            Y[n] = Y[n] + w * x**idx
    return Y

def loss_func(X, y, w_coeff):
    return np.sum((y - f_poly(X, w_coeff))**2)

def polynomial_features(x, degree):
    """Создает матрицу признаков для полинома степени `degree`"""
    X = np.vander(x, degree + 1, increasing=True)
    return X

def predict(X, w):
    """Предсказание: y_hat = X @ w"""
    return X @ w

def loss(w, X, y):
    """MSE: L(w) = 1/(2n) * ||Xw - y||^2"""
    n = len(y)
    residuals = X @ w - y
    return 0.5 * np.sum(residuals**2) / n

def gradient(w, X, y):
    """Градиент MSE: ∇L(w) = (1/n) * X^T (Xw - y)"""
    n = len(y)
    residuals = X @ w - y
    return (X.T @ residuals) / n

def hessian(w, X, y):
    """Гессиан MSE: H(w) = (1/n) * X^T X (не зависит от w!)"""
    n = len(y)
    return (X.T @ X) / n

def jacobian(w, X, y):
    """Якобиан остатков r_i = (Xw - y)_i по w: J = X"""
    return X