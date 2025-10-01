import numpy as np

def calculate_residual(T, D, Q, N):
    """
    calculate (max_i ||T q_i - d_i q_i||₂) / (N ε ||T||₂)
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
    denominator = np.sqrt((N + 1))* T_norm

    result = max_norm / denominator

    return result

def compute_orthogonality(Q, N):
    """
    calculate max_i ||Q^T q_i - e_i||_2 / (N * ε)
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
    result = max_norm / np.sqrt((N + 1))
    return result

def error_analysis(B, B_eigenvalue, B_eigenvector, N):
    """
    Perform error analysis for eigenvalue decomposition.

    """
    # Residual calculation
#     residual = np.linalg.norm(B - B_eigenvector @ B_eigenvalue @ B_eigenvector.T) / (np.linalg.norm(B) * (N + 1) * np.finfo(float).eps)
#     residual = np.linalg.norm(B - B_eigenvector @ B_eigenvalue @ B_eigenvector.T) / (np.linalg.norm(B) * np.sqrt((N + 1)))
    residual = calculate_residual(B, B_eigenvalue, B_eigenvector, N+1)
    max_residual = 0

    # Orthogonality check
    ERRORve = np.linalg.norm(B_eigenvector @ B_eigenvector.T - np.eye(N + 1))
#     orthogonality = ERRORve / ((N + 1) * np.finfo(float).eps)
#     orthogonality = ERRORve / np.sqrt((N + 1))
    orthogonality = compute_orthogonality(B_eigenvector, N+1)
    return residual, orthogonality