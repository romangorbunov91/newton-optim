# version 0.0.1 by romangorbunov91
# 19-Sep-2025
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from functions_under_study import loss_func, grad_func, hess_func, jacob_func

def newton(X, y, w_init, tolerance):
    w = w_init.copy()
    losses = [loss_func(X, y, w)]
    iteration_max =100

    for i in range(iteration_max):
        grad = grad_func(X, y, w)
        H = hess_func(X, y, w)
        
        # Проверка положительной определённости (для Холецкого)
        try:
            c, low = cho_factor(H)
            dw = cho_solve((c, low), -grad)
        except np.linalg.LinAlgError:
            print("Гессиан не положительно определён, используем градиентный шаг")
            dw = -grad  # fallback на градиентный спуск
        
        w += dw
        
        losses.append(loss_func(X, y, w))
        
        if np.linalg.norm(dw) < tolerance:
            print(f"Метод Ньютона сошёлся на итерации {i+1}")
            break
        
    return w, losses, i+1

def fit_newton_cholesky(self, X, y, max_iter=100, tol=1e-8, verbose=False):
    """Метод Ньютона с разложением Холецкого"""
    Phi = self._design_matrix(X)
    n, p = Phi.shape
    theta = np.zeros(p)

    for i in range(max_iter):
        grad = self.gradient(theta, Phi, y)
        if np.linalg.norm(grad) < tol:
            if verbose:
                print(f"Newton-Cholesky converged at iteration {i}")
            break

        H = self.hessian(theta, Phi, y)
        # Проверка положительной определённости (для Холецкого)
        try:
            L = cholesky(H, lower=True)
            # Решаем H @ d = -grad → L L^T d = -grad
            z = solve_triangular(L, -grad, lower=True)
            d = solve_triangular(L.T, z, lower=False)
        except np.linalg.LinAlgError:
            # Если Холецкий не прошёл — используем обычное обращение
            d = -np.linalg.solve(H, grad)
            if verbose:
                print("Cholesky failed, using direct solve.")

        theta = theta + d

    self.theta = theta
    return theta