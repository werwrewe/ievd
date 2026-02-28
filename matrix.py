"""
Defines functions related to matrix generation, including generating Laplacian matrices and random matrices.
"""
import numpy as np
import time
import threading
from scipy.linalg import lapack
# th = threading.Semaphore(1)

def generate_laplacian_matrix1(N):
    """Generate a Laplacian matrix from a random adjacency matrix."""
    adj_matrix = np.random.randint(0, 2, size=(N, N))
    adj_matrix = (adj_matrix + adj_matrix.T) // 2

    degrees = np.sum(adj_matrix, axis=1)

    degree_matrix = np.diag(degrees)

    laplacian_matrix = degree_matrix - adj_matrix

    return laplacian_matrix

def generate_matrix(N, mode, seed=None, cond=1.0e4):
    """
    Generate symmetric matrices for numerical linear algebra tests.

    Parameters:
    N (int): Matrix dimension.
    mode (int): Matrix generation mode (0-8).
    seed (int): Random seed.
    cond (float): Condition number (only valid for Modes 3, 4, 5).

    Mode descriptions:
    [Physical/Structural Tests]
    0: Random Symmetric (uniform distribution [-1, 1]) - general testing
    1: Toeplitz [1, 2, 1] (discrete Laplacian) - physical/PDE background
    2: Weak Diag Dominant [1, mu, 1] - non-diagonally dominant

    [Eigenvalue Distribution Tests (LAPACK style)]
    3: Geometric Distribution (geometric progression) - small eigenvalue testing
    4: Arithmetic Distribution (arithmetic progression) - uniform gap testing
    5: Log-Uniform Distribution (log uniform) - multi-scale/mixed precision testing


    6: Laplacian matrix with random adjacency matrix.
    """

    if seed is not None:
        np.random.seed(seed)

    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    # --- Mode 0: Random Symmetric Uniform [-1, 1] ---
    if mode == 0:
        # Generate fully random matrix, then symmetrize
        A = np.random.uniform(-1, 1, (N, N))
        A = (A + A.T) / 2
        return A

    # --- Mode 1: Wilkinson Matrix W_N^+ ---
    elif mode == 1:
        # Diagonal: |(N-1)/2 - i|, sub-diagonal: 1
        # Note: Python indices start from 0, center adjusted to (N-1)/2
        diag = np.abs((N - 1) / 2 - np.arange(N))
        off_diag = np.ones(N - 1)
        A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return A


    # --- Mode 1: Toeplitz Matrix [1, 2, 1] ---
    elif mode == 1:
        # Main diagonal 2, sub-diagonal 1
        diag = 2 * np.ones(N)
        off_diag = np.ones(N - 1)
        A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return A

    # --- Mode 2: Weakly Diagonally Dominant [1, mu_i, 1] ---
    elif mode == 2:
        # Sub-diagonal 1 (strong coupling)
        # Main diagonal mu_i = i * 10^-6 (weak variation)
        indices = np.arange(1, N + 1) # 1 to N
        diag = indices * 1e-6
        off_diag = np.ones(N - 1)
        A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return A


    # --- Modes 3-5: Specified Eigenvalue Distributions (LAPACK Style) ---
    elif mode in [3, 4, 5]:
        # 1. Generate eigenvalue vector D
        D = np.zeros(N)

        if mode == 3: # Geometric (Mode 3 in LAPACK)
            # D(i) = cond^(-(i)/(N-1))
            # Range: [1, 1/cond]
            for i in range(N):
                power = -float(i) / (N - 1) if N > 1 else 0
                D[i] = cond ** power

        elif mode == 4: # Arithmetic (Mode 4 in LAPACK)
            # D(i) = 1 - (i)/(N-1) * (1 - 1/cond)
            # Uniformly distributed in [1/cond, 1]
            slope = (1.0 - 1.0/cond) / (N - 1) if N > 1 else 0
            for i in range(N):
                D[i] = 1.0 - i * slope

        elif mode == 5: # Log-Uniform (Mode 5 in LAPACK)
            # log(D) uniformly distributed in [log(1/cond), 0]
            log_min = np.log(1.0/cond)
            log_max = 0.0
            random_logs = np.random.uniform(log_min, log_max, N)
            D = np.exp(random_logs)
            # For better testing, eigenvalues are usually sorted or randomized, kept random here

        # 2. Generate random orthogonal matrix Q (using QR decomposition)
        # Generate Gaussian random matrix
        X = np.random.randn(N, N)
        Q, _ = np.linalg.qr(X)

        # 3. Construct symmetric matrix A = Q D Q^T
        A = Q @ np.diag(D) @ Q.T

        # Ensure perfect symmetry (eliminate floating point errors)
        A = (A + A.T) / 2
        return A
    elif mode == 6:
        return generate_laplacian_matrix1(N)
    else:
        raise ValueError(f"Unknown mode: {mode}")