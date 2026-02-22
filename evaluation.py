import numpy as np

def calculate_residual(T, D, Q, N):
    """
    calculate (max_i ||T q_i - d_i q_i||₂) / (N ||T||₂)
    """
    max_norm = 0.0
    for i in range(N):
        q_i = Q[:, i]
        d_i = D[i, i]
        T_qi = np.dot(T, q_i)
        d_i_qi = d_i * q_i
        norm = np.linalg.norm(T_qi - d_i_qi, 2)
        if norm > max_norm:
            max_norm = norm

    T_norm = np.linalg.norm(T, 2)
#     denominator = N * np.finfo(float).eps * T_norm
    denominator = (N) * T_norm

    result = max_norm / denominator

    return result

def compute_orthogonality(Q, N):
    """
    calculate max_i ||Q^T q_i - e_i||_2 / (N )
    """
    epsilon = np.finfo(float).eps
    max_norm = 0.0
    QT = Q.T

    for i in range(N):
        qi = Q[:, i]
        ei = np.zeros(N)
        ei[i] = 1.0
        QT_qi = QT @ qi
        diff = QT_qi - ei
        norm = np.linalg.norm(diff,2)
        if norm > max_norm:
            max_norm = norm

#     result = max_norm / (N * epsilon)
    result = max_norm / N
    return result

def calculate_eigenvalues_error(computed_eigenvalues, true_eigenvalues, N):
    """
    Calculate error between computed eigenvalues and true eigenvalues.

    Args:
        computed_eigenvalues: Computed eigenvalues matrix (diagonal matrix)
        true_eigenvalues: True eigenvalues matrix (diagonal matrix)
        N: Size of the original matrix

    Returns:
        max_eigenvalue_error: Maximum absolute error between corresponding eigenvalues
        mean_eigenvalue_error: Mean absolute error between corresponding eigenvalues
        relative_error: Relative error normalized by the norm of true eigenvalues
    """
    # Extract diagonal elements (eigenvalues)
    computed = np.diag(computed_eigenvalues)
    true = np.diag(true_eigenvalues)

    # Sort eigenvalues to ensure proper correspondence
    computed_sorted = np.sort(computed)
    true_sorted = np.sort(true)

    # Calculate absolute errors
    absolute_errors = np.abs(computed_sorted - true_sorted)

    # Compute error metrics
    max_eigenvalue_error = np.max(absolute_errors)
    mean_eigenvalue_error = np.mean(absolute_errors)

    # Calculate relative error
    true_norm = np.linalg.norm(true_sorted)
    if true_norm > 1e-10:  # Avoid division by zero
        relative_error = np.linalg.norm(computed_sorted - true_sorted) / true_norm
    else:
        relative_error = np.linalg.norm(computed_sorted - true_sorted)

    return max_eigenvalue_error, mean_eigenvalue_error, relative_error

def error_analysis(B, B_eigenvalue, B_eigenvector, N):
    """
    Perform error analysis for eigenvalue decomposition.

    """
    # Residual calculation
#     residual = np.linalg.norm(B - B_eigenvector @ B_eigenvalue @ B_eigenvector.T) / (np.linalg.norm(B) * (N + 1) * np.finfo(float).eps)
#     residual = np.linalg.norm(B - B_eigenvector @ B_eigenvalue @ B_eigenvector.T) / (np.linalg.norm(B) * np.sqrt((N + 1)))
    # residual = calculate_residual(B, B_eigenvalue, B_eigenvector, N+1)
    residual = calculate_residual(B, B_eigenvalue, B_eigenvector, N)
    max_residual = 0

    # Orthogonality check
    # ERRORve = np.linalg.norm(B_eigenvector @ B_eigenvector.T - np.eye(N + 1))
#     orthogonality = ERRORve / ((N + 1) * np.finfo(float).eps)
#     orthogonality = ERRORve / np.sqrt((N + 1))
    # orthogonality = compute_orthogonality(B_eigenvector, N+1)
    orthogonality = compute_orthogonality(B_eigenvector, N)
    return residual, orthogonality