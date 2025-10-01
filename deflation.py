import numpy as np

def deflation_of_A_in_beta(Arrow_Matrix, N, tau):
    """
    Perform a deflation operation on the given symmetric arrow matrix, handling elements in z that are smaller than tau*||H||.
    The matrix is in the form of:
    H = [[D, z],
         [z^T, alpha]]
    The arrow points to the bottom-right.

    Parameters:
    Arrow_Matrix (numpy.array): Symmetric arrow matrix.
    N (int): Size of the matrix.
    tau (float): Threshold for determining if elements in z are small enough.
    """
    H = Arrow_Matrix.copy()
    deleted_indices = []
    eigenvalues = []
    eigenvectors = []

    threshold = np.finfo(float).eps * N
    temp = 0
    count = 0

    for i in range(N-1):
#         print(i)
        if np.abs(H[i, N-1]) < threshold:
            count += 1
            deleted_indices.append(i)
            eigenvalues.append(H[i, i])
            eigenvector = np.eye(N)[i].reshape(N,1)
            eigenvectors.append(eigenvector)
            temp += 1

    N -= temp

    if deleted_indices:
        H = np.delete(H, deleted_indices, axis=0)
        H = np.delete(H, deleted_indices, axis=1)

    return H, deleted_indices, np.diag(np.array(eigenvalues)), np.array(eigenvectors), N


def deflate_matrix(H, tau):
    """
    Perform a deflation operation on the given symmetric arrow matrix H, handling a series of close elements in D.
    Assume that the eigenvalues of H are already arranged in ascending order on the diagonal, and store the eigenvalues and eigenvectors in the specified matrices.

    Parameters:
    H (numpy.array): Symmetric arrow matrix with eigenvalues already arranged in ascending order on the diagonal.
    tau (float): Threshold for determining if eigenvalues are close.
    """
    N = H.shape[0]
    eigenvalue_count = 0
    deleted_indices = []
    eigenvalues = []
    eigenvectors = []
    identity_matrix = np.eye(N)
    threhold = np.finfo(float).eps * N
    H_deflated = H.copy()
    i = 0
    while i < N:
        if i < N - 1 and np.abs(H_deflated[i, i] - H_deflated[i+1, i+1]) <= threhold:
            # print('H_deflated',H_deflated[i, -1],H_deflated[i+1, -1])
            r = np.sqrt(H_deflated[i, -1]**2 + H_deflated[i+1, -1]**2)
            c = H_deflated[i, -1] / r
            s = H_deflated[i+1, -1] / r

            # Givens
            G = np.eye(N)
            G[i, i] = s
            G[i+1, i+1] = s
            G[i, i+1] = -c
            G[i+1, i] = c

            H_deflated[i,i] =s*s*H_deflated[i,i]+c*c*H_deflated[i+1,i+1]
            H_deflated[i+1,i+1] =c*c*H_deflated[i,i]+s*s*H_deflated[i+1,i+1]
            H_deflated[i+1,-1] = r
            H_deflated[i,-1] = 0
            H_deflated[-1,i+1] = r
            H_deflated[-1,i] = 0

            H_deflated[i+1,i] = 0
            H_deflated[i,i+1] = 0
            identity_matrix[:, [i, i+1]] = identity_matrix[:, [i, i+1]] @ np.array([
                                                                                    [s, c],
                                                                                    [-c, s]
                                                                                ])

            eigenvalues.append(H_deflated[i,i])
            deleted_indices.append(i)
            i += 1
        else:
            i += 1
    for item in deleted_indices:
        eigenvectors.append(identity_matrix[:,item])
    if deleted_indices:
        H_deflated = np.delete(H_deflated, deleted_indices, axis=0)
        H_deflated = np.delete(H_deflated, deleted_indices, axis=1)
        identity_matrix = np.delete(identity_matrix, deleted_indices, axis=1)
    N_small = N - len(deleted_indices)
    return H_deflated,deleted_indices,eigenvalues, eigenvectors, N_small,identity_matrix