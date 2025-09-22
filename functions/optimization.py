# version 0.3.2 by romangorbunov91
# 22-Sep-2025
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

        if np.linalg.norm(dw) < tolerance:
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
        J = jacob_func(X)
        jacob_counter += 1
        # Residuals.
        r = predict(X, w) - y

        try:
            # Estimate Hessian via Jacobian.
            H_gn = J.T @ J
            c, low = cho_factor(H_gn)
            g_gn = J.T @ r
            dw = cho_solve((c, low), -g_gn)
        except np.linalg.LinAlgError:
            dw = -grad_func(X, y, w)
            grad_counter += 1
            
        w += dw
        
        losses.append(loss_func(X, y, w))
        func_counter += 1

        if np.linalg.norm(dw) < tolerance:
            break
    
    hess_counter = 0    
    return w, losses, i+1, func_counter, grad_counter, hess_counter, jacob_counter


# Davidon–Fletcher–Powell.
def dfp(X, y, w_init, tolerance):
    iteration_max = int(1e4)
    
    w = w_init.copy()
    
    # Initialize inverse Hessian.
    H_inv = np.eye(len(w))
    
    grad = grad_func(X, y, w)
    grad_counter = 1

    loss = loss_func(X, y, w)
    losses = [loss]
    func_counter = 1

    # lr search parameters.
    lr_multiplier = 0.5
    lr_coeff = 1e-4
    eps_zero = 1e-10

    for i in range(iteration_max):
        learning_rate = 1.0
        p = H_inv @ grad
        
        dw = learning_rate * p
        w_test = w - dw
        new_loss = loss_func(X, y, w_test)
        func_counter += 1
        
        grad_dot_dw = np.dot(grad, dw)
        while ((loss - new_loss) < lr_coeff * learning_rate * grad_dot_dw) and (learning_rate > eps_zero):
            learning_rate *= lr_multiplier
            dw = learning_rate * p
            w_test = w - dw
            new_loss = loss_func(X, y, w_test)
            func_counter += 1
        
        # Update.
        w = w_test.copy()
        losses.append(new_loss)
        
        grad_prev = grad.copy()
        grad = grad_func(X, y, w)
        grad_counter += 1
        dgrad = grad - grad_prev        

        # Curvature.
        dwdgrad = np.dot(dw, dgrad)
        
        if dwdgrad > eps_zero:
            # DFP Update:
            # H⁺ = H + (s sᵀ)/(sᵀy) - (H y yᵀ H)/(yᵀ H y)
            Hdgrad = H_inv @ dgrad
            denom = np.dot(dgrad, Hdgrad)
            if abs(denom) > eps_zero:
                H_inv += (np.outer(dw, dw) / dwdgrad - np.outer(Hdgrad, Hdgrad) / denom)
            else:
                print(f"Iter {i+1}: denominator near zero, skipping H_inv update.")
        else:
            print(f"Iter {i+1}: Curvature condition violated (sᵀy = {dwdgrad:.2e}), resetting H_inv.")
            # Reset to identity.
            H_inv = np.eye(len(w))

        if np.linalg.norm(dw) < tolerance:
            break
        
    hess_counter = 0
    jacob_counter = 0  
    return w, losses, i+1, func_counter, grad_counter, hess_counter, jacob_counter

'''
Hy = H_inv @ dgrad
#H_inv -= (np.outer(Hy, Hy) / np.dot(dgrad, Hy) - np.outer(dw, dw) / np.dot(dw, dgrad))
H_inv = H_inv - np.outer(Hy, Hy) / np.dot(dgrad, Hy) + np.outer(dw, dw) / np.dot(dw, dgrad)
'''


'''
prod = np.dot(dw, dgrad)
if prod <= eps_zero:
    while prod <= eps_zero:
        print('condition', dw, dgrad, prod)
        
        learning_rate *= lr_multiplier
        dw = -learning_rate * p
        
        grad_prev = grad.copy()
        grad = grad_func(X, y, w+dw)
        grad_counter += 1

        dgrad = grad - grad_prev
        
        p = H_inv @ grad
        
        prod = np.dot(dw, dgrad)
        #loss = loss_func(X, y, w)
        #losses.append(loss)
        #func_counter += 1
        #continue
'''

'''
grad_norm = np.linalg.norm(grad, ord=None, axis=None)
while (loss - loss_func(X, y, w - learning_rate * grad)) < lr_coeff * learning_rate * grad_norm**2:
    func_counter += 1
    learning_rate *= lr_multiplier
# One more increment because of 'while'.
func_counter += 1
'''

'''
# BFGS (Broyden–Fletcher–Goldfarb–Shanno
def bfgs(X, y, w_init, tolerance):

    # BFGS update:
    # H_{k+1} = (I - ρ s y^T) H_k (I - ρ y s^T) + ρ s s^T, где ρ = 1/(y^T s)
    rho = 1.0 / np.dot(y_vec, s)
    I = np.eye(n_params)
    V = I - rho * np.outer(s, y_vec)
    H_inv = V.T @ H_inv @ V + rho * np.outer(s, s)
'''