# %%%%% eigensolver %%%%%%
import numpy as np
from func import *
from bisection import *
from deflation import *
from tools import *
import time

def evd_of_C(EA,beta,w,N):
    """
    Compute eigenvalues and eigenvectors of matrix C
    """
    if len(EA)==0:
        return [],[]
    # beta = QA.T @ Alpha
    # beta = beta.squeeze(1)
    norm2square = np.sum(beta ** 2)
    EA = np.diag(EA)

    theta_left = (EA[0] + w - np.sqrt((w - EA[0])**2 + 4 * norm2square)) / 2
    theta_right = (EA[-1] + w + np.sqrt((w - EA[-1])**2 + 4 * norm2square)) / 2

    rt_interval = np.concatenate((
        np.atleast_1d(theta_left),
        EA,
        np.atleast_1d(theta_right)
    ))
    C_eigenvalue = np.zeros((N + 1, N + 1))
    C_eigenvector = np.zeros((N + 1, N + 1))
    update4value = np.zeros(N + 1)
    update4vec = np.zeros(N + 1)
    epsilon = np.finfo(float).eps
    bsquare = beta ** 2
    i = 0

    #compute eigenvalue of C
    time0 = time.perf_counter()
    rt_left = []
    rt_right = []
    seq = []
    for i in range(N + 1):
        if update4value[i] == 0:
            seq.append(i)
            # Calculate the margin of the current interval by subtracting a small value proportional to the machine epsilon
            left_margin = rt_interval[i] - np.finfo(float).eps * abs(rt_interval[i])
            right_margin = rt_interval[i+1] + np.finfo(float).eps * abs(rt_interval[i+1])
            rt_left.append(left_margin)
            rt_right.append(right_margin)
    rt_left = np.array(rt_left)
    rt_right = np.array(rt_right)
    root = np.zeros(N+1)
    #print('rt_left=',rt_left)
    itermax_vector = np.log2(rt_right - rt_left) - np.log2(epsilon)
    #itermax = np.linalg.norm(x = itermax_vector,ord = np.Inf)
    bifunc_vector(cpfunc, rt_left, rt_right, epsilon, w, bsquare, EA,itermax_vector, root, seq)
    #bifunc_vector(cpfunc, rt_left, rt_right, epsilon, w, bsquare, EA,itermax, root)
    #print('root=',root)
    count = 0
    i = 0
    while i<=N:
        if update4value[i]==0:
            C_eigenvalue[i, i] = root[count]
            count += 1
        i += 1
    #bifunc_vector2(cpfunc, rt_left, rt_right, epsilon, w, bsquare, EA,itermax_vector, root)
    #for i in range(N + 1):
    #        root[i] = bifunc(cpfunc, rt_left[i], rt_right[i], epsilon, w, bsquare, EA,i, itermax_vector[i])
    #for i in range(N + 1):
    #        C_eigenvalue[i, i]= root[i]
    time1 = time.perf_counter()
    time_eig = time1 - time0
    #print(f'compute eigenvalue of C: {time_eig:.4e}')
    #recompute beta
    #print('C_eigenvector1\n',C_eigenvector)
    beta = compute_beta(np.diag(C_eigenvalue), EA, N, beta,update4value)
    #print('C_eigenvector2\n',C_eigenvector)
    time2 = time.perf_counter()
    time_rebeta = time2 - time1

    for i in range(N + 1):
        if update4vec[i] != 1:
            if i > 0:
                if (C_eigenvalue[i, i] - C_eigenvalue[i - 1, i - 1]) < 1e-8:
                    # Print information about found duplicate eigenvalues
                    print(f"\nFound adjacent close eigenvalues (C_eigenvalue):")
                    print(f"Positions: i={i-1} and i={i}")
                    print(f"Eigenvalue {i-1}: {C_eigenvalue[i-1, i-1].item():.6e}")
                    print(f"Eigenvalue {i}: {C_eigenvalue[i, i].item():.6e}")
                    print(f"Difference: {(C_eigenvalue[i, i] - C_eigenvalue[i-1, i-1]).item():.2e}")
                    print(f"Threshold: {2 * np.finfo(float).eps:.2e}")

                    # If needed, print surrounding eigenvalues for comparison
                    start_idx = max(0, i-2)
                    end_idx = min(i+2, C_eigenvalue.shape[0])
                    print(f"\nSurrounding eigenvalues ({start_idx} to {end_idx-1}):")
                    for k in range(start_idx, end_idx):
                        print(f"Eigenvalue {k}: {C_eigenvalue[k, k].item():.6e}")
                        if k < end_idx-1:
                            diff = (C_eigenvalue[k+1, k+1] - C_eigenvalue[k, k]).item()
                            print(f"Difference with next eigenvalue: {diff:.2e}")

                    iden = np.zeros(N + 1)
                    iden[i] = 1
                    C_eigenvector[:, i] = iden
                    C_eigenvector[i, i - 1] = 0
                else:
                    C_eigenvector[:, i] = np.concatenate((-beta / (EA - C_eigenvalue[i, i]), [1.0]))
                    C_eigenvector[:, i] = C_eigenvector[:, i] / np.linalg.norm(C_eigenvector[:, i])
            else:
                C_eigenvector[:, i] = np.concatenate((-beta / (EA - C_eigenvalue[i, i]), [1.0]))
                C_eigenvector[:, i] = C_eigenvector[:, i] / np.linalg.norm(C_eigenvector[:, i])

    time3 = time.perf_counter()
    time_eigv = time3 - time2
    #print(f'compute eigenvectors of C: {time_eigv:.4e}')
    #compute eigenvalue and eigenvectors of B
    return C_eigenvector,C_eigenvalue



def evd_of_B(QA,C_eigenvector,C_eigenvalue,N):
    """
    Compute eigenvalues and eigenvectors of matrix B
    """
    time3 = time.perf_counter()
    EB = C_eigenvalue
    QB = np.dot(
        np.block([[QA, np.zeros((N, 1))], [np.zeros((1, N)), 1]]),
        C_eigenvector
    )
    time4 = time.perf_counter()
    time_eigvb = time4 - time3
    #print(f'compute eigenvalue and eigenvectors of B: {time_eigvb:.4e}')
    return QB,EB

def evd(QA,EA,Alpha,w,N):
    beta = QA.T @ Alpha
    Arrow_Matrix = np.block([[EA, beta], [beta.T, w]])
    tau = 0.2
    N = N + 1
    H, deleted_indices, eigenvalues1, eigenvectors1, N_small, identity_matrix = deflate_matrix(Arrow_Matrix, tau)
    EA_samll,beta_small,w_small  = np.diag(np.diag(H[:N_small-1, :N_small-1])),H[:N_small-1, N_small-1],np.array(H[N_small-1, N_small-1])
    try:
        beta_small = beta_small.reshape(N_small-1,)
    except:
        beta_small = beta_small

    if len(EA_samll) != 0:
        C_eigenvector,C_eigenvalue = evd_of_C(EA_samll,beta_small,w_small,N_small-1)
        eigenvalues2, eigenvectors2 = C_eigenvalue,identity_matrix @ C_eigenvector
        C_eigenvalue,C_eigenvector = rearrange2(deleted_indices,eigenvalues1, eigenvectors1,eigenvalues2, eigenvectors2)
    else:
        C_eigenvalue,C_eigenvector = np.diag(np.append(np.diag(eigenvalues1), w_small)), np.eye(eigenvectors1.shape[0]+1)
    QB,EB = evd_of_B(QA,C_eigenvector,C_eigenvalue,N-1)
    return QB,EB
# %%
