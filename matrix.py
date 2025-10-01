"""
Defines functions related to matrix generation, including generating Laplacian matrices and random matrices.
"""
import numpy as np
import time
import threading
from scipy.linalg import lapack
th = threading.Semaphore(1)

def generate_laplacian_matrix1(N):
    adj_matrix = np.random.randint(0, 2, size=(N, N))
    adj_matrix = (adj_matrix + adj_matrix.T) // 2

    degrees = np.sum(adj_matrix, axis=1)

    degree_matrix = np.diag(degrees)

    laplacian_matrix = degree_matrix - adj_matrix

    return laplacian_matrix

def generate_matrix(N, mode, seed):
    np.random.seed(seed)
    """
    Generate a matrix with specified eigenvalues and random eigenvectors.
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")
    if mode not in range(2):
        raise ValueError("Invalid mode. Mode must be an integer between 0 and 2.")

    if mode == 0:
        eigenvalues = np.random.rand(N)
    elif mode == 1:
        return generate_laplacian_matrix1(N)
    Q, _ = np.linalg.qr(np.random.randn(N, N))
    # Construct the matrix with eigenvalues and eigenvectors
    D = np.diag(eigenvalues)  # Diagonal matrix of eigenvalues
    A = Q @ D @ Q.T  # Matrix reconstruction
    return A