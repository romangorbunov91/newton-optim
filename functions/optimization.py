# version 0.0.2 by romangorbunov91
# 20-Sep-2025
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from functions.functions_under_study import predict, loss_func, grad_func, hess_func, jacob_func

def newton(X, y, w_init, tolerance):
    iteration_max = 1000
    
    w = w_init.copy()
    losses = [loss_func(X, y, w)]
    func_counter = 1
    
    grad_counter = 0
    hess_counter = 0

    for i in range(iteration_max):
        grad = grad_func(X, y, w)
        grad_counter += 1
        
        H = hess_func(X)
        hess_counter += 1
        
        try:
            c, low = cho_factor(H)
            dw = cho_solve((c, low), -grad)
        except np.linalg.LinAlgError:
            dw = -grad
        
        w += dw
        
        losses.append(loss_func(X, y, w))
        func_counter += 1

        if np.linalg.norm(grad) < tolerance:
            break
    
    jacob_counter = 0    
    return w, losses, i+1, func_counter, grad_counter, hess_counter, jacob_counter

def gauss_newton(X, y, w_init, tolerance):
    iteration_max = 1000
    
    w = w_init.copy()
    losses = [loss_func(X, y, w)]
    func_counter = 1
    
    grad_counter = 0
    jacob_counter = 0

    for i in range(iteration_max):
        grad = grad_func(X, y, w)
        grad_counter += 1
        
        J = jacob_func(X)
        jacob_counter += 1
        r = predict(X, w) - y
        #r = X @ w - y  # остатки
        
        # Система: (J^T J) dw = -J^T r
        try:
            H_gn = J.T @ J
            g_gn = J.T @ r
            c, low = cho_factor(H_gn)
            dw = cho_solve((c, low), -g_gn)
        except np.linalg.LinAlgError:
            dw = -grad
            
        w += dw
        
        losses.append(loss_func(X, y, w))
        func_counter += 1

        if np.linalg.norm(grad) < tolerance:
            break
    
    hess_counter = 0    
    return w, losses, i+1, func_counter, grad_counter, hess_counter, jacob_counter