# version 0.3.1 by romangorbunov91
# 21-Sep-2025
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


def dfp(X, y, w_init, tolerance):
    """
    Davidon-Fletcher-Powell (DFP) Quasi-Newton Method

    Solves: min_w f(w) = loss_func(X, y, w)
    Using gradient information and iterative inverse Hessian approximation.

    Parameters:
        X : ndarray (n_samples, n_features)
        y : ndarray (n_samples,)
        w_init : ndarray (n_features,) — initial weights
        tolerance : float — convergence threshold on step norm
        max_iter : int — maximum iterations
        verbose : bool — print debug info

    Returns:
        w : optimized weights
        losses : list of loss values per iteration
        iterations : int — number of iterations performed
        func_counter : int — number of loss evaluations
        grad_counter : int — number of gradient evaluations
        hess_counter : int — (unused, for compatibility) 0
        jacob_counter : int — (unused, for compatibility) 0
    """
    iteration_max = int(1e4)
    # Make sure we don't modify the original
    w = w_init.copy()
    n = len(w)

    # Initialize inverse Hessian approximation as identity
    H_inv = np.eye(n)

    # Evaluate initial gradient
    grad = grad_func(X, y, w)
    grad_prev = grad.copy()
    grad_counter = 1
    func_counter = 0

    # Line search parameters
    lr_init = 1.0
    lr_multiplier = 0.5
    c1 = 1e-4  # Armijo condition constant
    eps_zero = 1e-10

    losses = []
    loss = loss_func(X, y, w)
    losses.append(loss)
    func_counter += 1

    for i in range(iteration_max):
        # Search direction: p = H_inv @ grad
        p = H_inv @ grad
        dw = -p  # full step direction

        # Backtracking line search
        learning_rate = lr_init
        current_loss = loss
        grad_dot_dw = np.dot(grad, dw)

        # Armijo condition: f(w + a*dw) <= f(w) + c1 * a * grad^T dw
        trial_w = w + learning_rate * dw
        trial_loss = loss_func(X, y, trial_w)
        func_counter += 1

        while trial_loss > current_loss + c1 * learning_rate * grad_dot_dw and learning_rate > 1e-10:
            learning_rate *= lr_multiplier
            trial_w = w + learning_rate * dw
            trial_loss = loss_func(X, y, trial_w)
            func_counter += 1

        # Take the step
        step = learning_rate * dw
        w = trial_w
        loss = trial_loss
        losses.append(loss)

        # Compute new gradient
        grad_new = grad_func(X, y, w)
        grad_counter += 1
        dgrad = grad_new - grad_prev  # y = grad_new - grad_prev
        s = step                      # s = w_new - w_old

        # Curvature condition: sᵀy > 0
        sy = np.dot(s, dgrad)

        if sy > eps_zero:
            # DFP Update:
            # H⁺ = H + (s sᵀ)/(sᵀy) - (H y yᵀ H)/(yᵀ H y)
            Hy = H_inv @ dgrad
            term1 = np.outer(s, s) / sy
            denom = np.dot(dgrad, Hy)
            if abs(denom) > eps_zero:
                term2 = np.outer(Hy, Hy) / denom
                H_inv = H_inv + term1 - term2
            else:
                print(f"Iter {i+1}: denominator near zero, skipping H_inv update.")
        else:
            print(f"Iter {i+1}: Curvature condition violated (sᵀy = {sy:.2e}), resetting H_inv.")
            H_inv = np.eye(n)  # Reset to identity

        # Update gradients for next iteration
        grad_prev = grad.copy()
        grad = grad_new

        # Check convergence
        if np.linalg.norm(step) < tolerance:
            break

    return w, losses, i+1, func_counter, grad_counter, 0, 0

'''
# Davidon–Fletcher–Powell.
def dfp(X, y, w_init, tolerance):
    iteration_max = int(1e4)
    
    w = w_init.copy()
    
    grad = grad_func(X, y, w)
    grad_counter = 1
    grad_prev = grad.copy()
    
    func_counter = 0
    hess_counter = 0
    jacob_counter = 0
    
    losses = []
    
    eps_zero = 1e-10
    #lr_coeff = 1e-1
    lr_multiplier = 0.5

    for i in range(iteration_max):
        learning_rate = 1.0
        dw = -learning_rate * grad
        
        grad = grad_func(X, y, w+dw)
        grad_counter += 1
        dgrad = grad - grad_prev
              
        xTx = dw @ dw
        H = np.outer(dgrad, dw) / xTx
        
        try:
            c, low = cho_factor(H)
            dw = cho_solve((c, low), -grad)
        except np.linalg.LinAlgError:
            dw = -grad

        w += dw
        
        loss = loss_func(X, y, w)
        losses.append(loss)
        func_counter += 1

        if np.linalg.norm(dw) < tolerance:
            break
      
    return w, losses, i+1, func_counter, grad_counter, hess_counter, jacob_counter
'''

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