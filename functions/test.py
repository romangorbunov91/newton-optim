def dfp2(X, y, w_init, tolerance):
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
