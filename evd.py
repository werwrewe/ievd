# %%%%% eigensolver %%%%%%
from os import error
import numpy as np
from func import *
from bisection import *
from deflation import *
from tools import *
import time

import numpy as np
import time
from bisection import bifunc_relative


def evd_of_C(EA, beta, w, N, config):
    """
    Compute eigenvalues and eigenvectors of matrix C for the incremental rank-1 update problem.
    Target: C = [[EA, beta], [beta.T, w]] (Arrowhead matrix)

    Steps:
    1. Calculate theta boundaries using eigenvalue interlacing property
    2. Determine search intervals for eigenvalues
    3. Solve secular equation using bisection method
    4. Recompute beta for stability (Gu-Eisenstat method)
    5. Construct eigenvectors using the recomputed beta
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
    itermax_vector = np.log2(rt_right - rt_left) - np.log2(epsilon) + 1000
    # print('itermax_vector=',itermax_vector)
    #itermax = np.linalg.norm(x = itermax_vector,ord = np.Inf)
    if config['stop_criterion'] == 'gu':
        bifunc_vector_gu(cpfunc, rt_left, rt_right, epsilon, w, bsquare, EA,itermax_vector, root, seq,config)
    elif config['stop_criterion'] == 'std':
        bifunc_vector_std(cpfunc, rt_left, rt_right, epsilon, w, bsquare, EA,itermax_vector, root, seq)
    else:
        raise error
    # bifunc_vector_muti(cpfunc, rt_left, rt_right, epsilon, w, bsquare, EA,itermax_vector, root)
    #bifunc_vector(cpfunc, rt_left, rt_right, epsilon, w, bsquare, EA,itermax, root)
    #print('root=',root)
    count = 0
    i = 0
    while i<=N:
        if update4value[i]==0:
            C_eigenvalue[i, i] = root[count]
            count += 1
        i += 1

    time1 = time.perf_counter()
    time_eig = time1 - time0
    #print(f'compute eigenvalue of C: {time_eig:.4e}')
    #recompute beta
    #print('C_eigenvector1\n',C_eigenvector)
    # beta = compute_beta(np.diag(C_eigenvalue), EA, N, beta,update4value)
    beta = compute_beta_matrix(root, EA, beta)
    #print('C_eigenvector2\n',C_eigenvector)
    time2 = time.perf_counter()
    time_rebeta = time2 - time1
    C_eigenvector = construct_eigenvectors_matrix(beta,EA,root)

    time3 = time.perf_counter()
    time_eigv = time3 - time2
    #print(f'compute eigenvectors of C: {time_eigv:.4e}')
    #compute eigenvalue and eigenvectors of B
    return C_eigenvector,C_eigenvalue


def evd_of_B(QA, C_eigenvector, C_eigenvalue, N):
    """
    Compute eigenvalues and eigenvectors of matrix B from matrix C.
    Target: B = [[QA, 0], [0, 1]] * C * [[QA.T, 0], [0, 1]]

    Steps:
    1. Use eigenvalues from matrix C directly (EB = C_eigenvalue)
    2. Transform eigenvectors from C's basis to B's basis using QA
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


def evd(QA, EA, Alpha, w, N, config):
    """
    Main Entry Point for Incremental Eigenvalue Decomposition.
    Target: B = A + w * Alpha * Alpha^T, where A = QA * EA * QA^T

    Steps:
    1. Compute beta = QA^T * Alpha
    2. Construct arrowhead matrix
    3. Deflate the problem to reduce size
    4. Solve the reduced eigenvalue problem
    5. Rearrange results if deflation was applied
    6. Compute final eigenvalues and eigenvectors of B
    """
    beta = QA.T @ Alpha
    Arrow_Matrix = np.block([[EA, beta], [beta.T, w]])
    tau = 0.2
    N = N + 1
    H, deleted_indices, eigenvalues1, eigenvectors1, N_small, identity_matrix = deflate_matrix(Arrow_Matrix, tau)
    EA_samll, beta_small, w_small = np.diag(np.diag(H[:N_small-1, :N_small-1])), H[:N_small-1, N_small-1], np.array(H[N_small-1, N_small-1])
    try:
        beta_small = beta_small.reshape(N_small-1,)
    except:
        beta_small = beta_small

    if len(EA_samll) != 0:
        C_eigenvector, C_eigenvalue = evd_of_C(EA_samll, beta_small, w_small, N_small-1, config)
        eigenvalues2, eigenvectors2 = C_eigenvalue, identity_matrix @ C_eigenvector
        C_eigenvalue, C_eigenvector = rearrange2(deleted_indices, eigenvalues1, eigenvectors1, eigenvalues2, eigenvectors2)
    else:
        C_eigenvalue, C_eigenvector = np.diag(np.append(np.diag(eigenvalues1), w_small)), np.eye(eigenvectors1.shape[0]+1)
    QB, EB = evd_of_B(QA, C_eigenvector, C_eigenvalue, N-1)
    return QB, EB


def evd_of_C_standard(EA, z, rho, N, config):
    """
    Solves the secular equation for the Standard Rank-1 Update.
    Equation: 1 + rho * sum( z_i^2 / (d_i - lambda) ) = 0

    Steps:
    1. Determine search intervals using interlacing property
    2. Set up bisection method with dynamic error tolerance
    3. Solve secular equation using bisection
    4. Recompute z vector for stability (Gu-Eisenstat method)
    5. Construct eigenvectors using the formula: v_i = z / (D - lambda_i)
    """
    z_square = z**2


    rt_interval = np.zeros(N + 1)

    if rho > 0:
        # Standard Interlacing
        rt_interval[:-1] = EA
        # Upper bound for last root
        rt_interval[-1] = EA[-1] + rho * np.sum(z_square)
    else:
        # Reverse Interlacing
        rt_interval[1:] = EA
        # Lower bound for first root
        rt_interval = EA + rho * np.sum(z_square)

    # Margins for bisection
    epsilon = np.finfo(float).eps

    rt_left = rt_interval[:-1] - epsilon * np.abs(rt_interval[:-1])
    rt_right = rt_interval[1:] + epsilon * np.abs(rt_interval[1:])

    interval_widths = np.maximum(rt_right - rt_left, epsilon)

    itermax_vector = np.log2(interval_widths) - np.log2(epsilon) + 1000

    seq = np.arange(N)
    root = np.zeros(N)

    if config['stop_criterion'] == 'gu':
        bifunc_vector_gu(cpfunc_standard, rt_left, rt_right, epsilon, rho, z_square, EA, itermax_vector, root, seq, config)
    elif config['stop_criterion'] == 'std':
        bifunc_vector_std(cpfunc_standard, rt_left, rt_right, epsilon, rho, z_square, EA, itermax_vector, root, seq)
    else:
        raise error

    C_eigenvalue = np.diag(root)

    update4value = np.zeros(N) # No locked values
    z_recomputed = compute_z(root, EA, N, rho, z, update4value)

    C_eigenvector = construct_eigenvectors_standard(z_recomputed, EA, root)
    return C_eigenvector, C_eigenvalue


def evd_standard(QA, EA, u, rho, config):
    """
    Main Entry Point for Standard Rank-1 Update EVD.
    Target: A_new = A + rho * u * u^H
    Known: A = QA * EA * QA^H

    Steps:
    1. Project u to z = QA^H * u
    2. Deflate problem (D + rho * z * z^T)
    3. Solve secular equation
    4. Recover eigenvectors
    """
    N = len(EA)
    EA = np.diag(EA)
    z = QA.T @ u

    z = z.flatten()
    EA_small, z_small, deleted_indices, deflated_vals, perm_idx = deflate_standard(EA, z, rho)

    N_small = len(EA_small)

    if N_small > 0:
        C_evec_small, C_eval_small = evd_of_C_standard(EA_small, z_small, rho, N_small, config)

        # Extract diagonal eigenvalues
        solved_vals = np.diag(C_eval_small)
        solved_vecs = C_evec_small
    else:
        solved_vals = np.array()
        solved_vecs = np.zeros((0, 0))

    # Combine values
    final_evals = np.zeros(N)
    final_evecs_in_D_basis = np.zeros((N, N))

    # Pointer for small array
    ptr_small = 0
    # Pointer for deflated array
    ptr_defl = 0

    # Current sorted order is perm_idx.
    # deleted_indices refers to positions in the SORTED arrays.

    for i in range(N):
        if i in deleted_indices:
            final_evals[i] = deflated_vals[ptr_defl]
            final_evecs_in_D_basis[i, i] = 1.0 # Standard unit vector
            ptr_defl += 1
        else:
            final_evals[i] = solved_vals[ptr_small]
            # Map the small eigenvector column to the correct rows
            # Rows corresponding to deleted_indices are 0
            # Rows corresponding to remaining are from solved_vecs

            # Map rows
            row_ptr = 0
            for r in range(N):
                if r not in deleted_indices:
                    final_evecs_in_D_basis[r, i] = solved_vecs[row_ptr, ptr_small]
                    row_ptr += 1
                else:
                    final_evecs_in_D_basis[r, i] = 0.0

            ptr_small += 1

    QA_sorted = QA[:, perm_idx]

    # QB = QA_sorted * Final_Evecs_in_D_basis
    QB = QA_sorted @ final_evecs_in_D_basis

    # Return as diagonal matrix and matrix
    return QB, np.diag(final_evals)