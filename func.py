import numpy as np

def cpfunc(lmbda, w, bb, eigenvalue,left,right,i):
    if i % 2 == 1:
        if lmbda == left:
            return -1
        if lmbda == right:
            return 1
    if i % 2 == 0:
        if lmbda == left:
            return 1
        if lmbda == right:
            return -1

    ev_lambda = eigenvalue - lmbda
    SN = np.sign(ev_lambda)
    signal = np.sum(SN == -1)  # Count how many are -1
    if signal % 2 == 1:
        P1 = -1
    else:
        P1 = 1
    y = P1 * (w - lmbda - np.sum(bb / ev_lambda))
    return y

def cpfunc2(lmbda, w, bb, eigenvalue,left,right,i):
    return np.array([cpfunc(lmbda[j],w, bb, eigenvalue,left[j],right[j],i[j]) for j in range(len(lmbda))])



def cpfunc_standard(lmbda, rho, bb, eigenvalue, left, right, i):
    """
    Standard Secular Equation: f(lambda) = 1 + rho * sum( z_i^2 / (d_i - lambda) )
    Used for Standard Rank-1 Update.

    Params:
        rho: The scalar rho in A + rho*u*u^H. (Passed via 'w' in bisection)
        bb: Squared components of z vector (z_i^2).
        eigenvalue: The old eigenvalues d_i.
    """
    # Boundary handling to help bisection determine signs
    # For standard update (rho > 0), function is increasing.
    # Left limit -> -inf, Right limit -> +inf (between poles)
    if lmbda == left:
        return -1
    if lmbda == right:
        return 1

    ev_lambda = eigenvalue - lmbda

    # Calculate the secular function value
    # f(x) = 1 + rho * sum( z^2 / (d - x) )
    # Note: P1 parity logic is omitted here for simplicity as standard interlacing
    # is often strictly monotonic between poles. However, to handle poles robustly
    # like the augmented case, we can rely on bisection's sign check.

    val = 1.0 + rho * np.sum(bb / ev_lambda)
    return val

def cpfunc3_standard(lmbda, rho, bb, eigenvalue, left, right, i):
    # Vectorized wrapper for standard secular equation
    return np.array([cpfunc_standard(lmbda[j], rho, bb, eigenvalue, left[j], right[j], i[j])
                     for j in range(len(lmbda))])
