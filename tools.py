import numpy as np
import yaml

'''
    Useful tools
'''

def sorted_eig(QA, EA):
    """
    Sort eigenvalues and eigenvectors in ascending order.
    """
    # Extract eigenvalues from the diagonal of EA
    eigenvalues = np.diag(EA)

    # Sort eigenvalues and get the sorting indices
    idx = np.argsort(eigenvalues)

    # Sort eigenvalues and eigenvectors
    EA_sorted = np.diag(eigenvalues[idx])
    QA_sorted = QA[:, idx]

    return QA_sorted, EA_sorted


def compute_beta_matrix(roots, poles, beta_orig):
    """
    Recompute beta using matrix operations for improved efficiency.
    """

    num_diffs = roots[np.newaxis, :] - poles[:, np.newaxis]
    den_diffs = poles[np.newaxis, :] - poles[:, np.newaxis]
    np.fill_diagonal(den_diffs, 1.0)

    # sum_j log|num_ij|
    sum_log_num = np.sum(np.log(np.abs(num_diffs)), axis=1)
    # sum_k log|den_ik|
    sum_log_den = np.sum(np.log(np.abs(den_diffs)), axis=1)

    log_beta_sq = sum_log_num - sum_log_den
    beta_new = np.sqrt(np.exp(log_beta_sq))

    return beta_new * np.sign(beta_orig)

def compute_beta(C_eigenvalue, eigenvalue, N, beta,update4vec):
    """
    Recomputes beta values.
    """
    beta_ = np.zeros(N)

    tmp_eigenvalue_i = np.zeros(N + 1)
    C_eigenvalue_j_minus_eigenvalue_i = np.zeros(N + 1)
    eigenvalue_j_minus_eigenvalue_i = np.zeros(N + 1)
    tmp_divide = np.zeros(N + 1)
    for i in range(N):
        if update4vec[i]==1 or update4vec[min(i+1,N)]==1 or update4vec[max(0,i-1)]==1:
            beta_[i] = np.abs(beta[i])
        else:
            tmp_eigenvalue_i = eigenvalue[i]

            C_eigenvalue_j_minus_eigenvalue_i = C_eigenvalue - tmp_eigenvalue_i;
            eigenvalue_j_minus_eigenvalue_i = eigenvalue - tmp_eigenvalue_i;
            eigenvalue_j_minus_eigenvalue_i.resize(N + 1)
            eigenvalue_j_minus_eigenvalue_i[N] = 1
            eigenvalue_j_minus_eigenvalue_i[i] = 1

            tmp_divide = C_eigenvalue_j_minus_eigenvalue_i / eigenvalue_j_minus_eigenvalue_i
            tmp_divide[i] = C_eigenvalue_j_minus_eigenvalue_i[i]
            tmp_divide[N] = C_eigenvalue_j_minus_eigenvalue_i[N]

            log_prod = np.sum(np.log(np.abs(tmp_divide)))
            beta_[i] = np.sqrt(np.exp(log_prod))
    # Adjust sign using the original beta
    beta_ *= np.sign(beta)

    return beta_

def compute_z(new_eigenvalues, old_eigenvalues, N, rho, z, update4vec=None):
    """
    Recompute z.
    """
    num_diffs = new_eigenvalues[np.newaxis, :] - old_eigenvalues[:, np.newaxis]

    den_diffs = old_eigenvalues[np.newaxis, :] - old_eigenvalues[:, np.newaxis]

    np.fill_diagonal(den_diffs, 1.0)

    sum_log_num = np.sum(np.log(np.abs(num_diffs)), axis=1)
    sum_log_den = np.sum(np.log(np.abs(den_diffs)), axis=1)

    log_z_sq = sum_log_num - sum_log_den - np.log(np.abs(rho))
    z_new = np.sqrt(np.exp(log_z_sq))

    return z_new * np.sign(z)

def rearrange(deleted_indices, eigenvalues1, eigenvectors1, eigenvalues2, eigenvectors2):
    """
    Concatenate two sets of eigenvalues and eigenvectors into a complete N×N matrix.
    Both eigenvalues1 and eigenvalues2 are diagonal matrices, which are concatenated along the diagonal to form an N×N matrix.
    eigenvectors1 and eigenvectors2 are the corresponding eigenvectors, and their concatenation also forms an N×N matrix.
    deleted_indices specifies the positions of eigenvalues1 and eigenvectors1 in the entire concatenated matrix.
    """
    N = len(deleted_indices) + eigenvalues2.shape[0]

    combined_eigenvalues = np.zeros((N, N))
    combined_eigenvectors = np.zeros((N, N))
    # Place eigenvalues1 and eigenvectors1 in their corresponding positions
    for i, idx in enumerate(deleted_indices):
        combined_eigenvalues[idx, idx] = eigenvalues1[i, i]
        combined_eigenvectors[:, idx] = np.eye(N)[idx].reshape(N,)

    remaining_indices = [i for i in range(N) if i not in deleted_indices]
    for i, idx in enumerate(remaining_indices):
        inserted_count = 0
        temp = np.zeros(N, dtype=eigenvectors2.dtype)
        for j, value in enumerate(eigenvectors2[:, i]):
            # Calculate the actual position of the current value in the new vector
            new_index = j + inserted_count
            # Check if a 0 needs to be inserted
            if new_index in deleted_indices:
                inserted_count += 1
                new_index += 1
            # Fill in the original value
            temp[new_index] = value
        combined_eigenvalues[idx, idx] = eigenvalues2[i, i]
        combined_eigenvectors[:, idx] = temp
#         print('not del',i,idx,combined_eigenvectors)

    return combined_eigenvalues, combined_eigenvectors


def rearrange2(deleted_indices, eigenvalues1, eigenvectors1, eigenvalues2, eigenvectors2):
    """
    Concatenate two sets of eigenvalues and eigenvectors into a complete N×N matrix.
    Both eigenvalues1 and eigenvalues2 are diagonal matrices, which are concatenated along the diagonal to form an N×N matrix.
    eigenvectors1 and eigenvectors2 are the corresponding eigenvectors, and their concatenation also forms an N×N matrix.
    deleted_indices specifies the positions of eigenvalues1 and eigenvectors1 in the entire concatenated matrix.
    """
    N = len(deleted_indices) + eigenvalues2.shape[0]

    combined_eigenvalues = np.zeros((N, N))
    combined_eigenvectors = np.zeros((N, N))

    # Place eigenvalues1 and eigenvectors1 in their corresponding positions
    for i, idx in enumerate(deleted_indices):
        combined_eigenvalues[idx, idx] = eigenvalues1[i]
        combined_eigenvectors[:, idx] = eigenvectors1[i]
    remaining_indices = [i for i in range(N) if i not in deleted_indices]
    for i, idx in enumerate(remaining_indices):
        combined_eigenvalues[idx, idx] = eigenvalues2[i, i]
        combined_eigenvectors[:, idx] = eigenvectors2[:, i]

    return combined_eigenvalues, combined_eigenvectors

def construct_eigenvectors_matrix(beta_hat, poles, roots):
    """
    compute eigenvectors for incremental rank-1 update.
    """
    denom = poles[:, np.newaxis] - roots[np.newaxis, :]
    V_top = -beta_hat[:, np.newaxis] / denom
    V_bot = np.ones((1, V_top.shape[1]))

    V = np.vstack([V_top, V_bot])

    norms = np.linalg.norm(V, axis=0)
    V_normalized = V / norms[np.newaxis, :]

    return V_normalized

def construct_eigenvectors_standard(z_recomputed, EA, roots):
    """
     compute eigenvectors for standard rank-1 update.
    """
    denom = EA[:, np.newaxis] - roots[np.newaxis, :]

    V = z_recomputed[:, np.newaxis] / denom

    norms = np.linalg.norm(V, axis=0)
    V_normalized = V / norms[np.newaxis, :]

    return V_normalized

