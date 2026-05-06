# %%%%% eigensolver %%%%%%
from os import error
import numpy as np
import time
from func import *
from bisection import *
from deflation import deflate_incremental, deflate_standard, deflation_tol
from tools import *


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
            left_margin = rt_interval[i] - np.finfo(float).eps * abs(rt_interval[i])
            right_margin = rt_interval[i+1] + np.finfo(float).eps * abs(rt_interval[i+1])
            rt_left.append(left_margin)
            rt_right.append(right_margin)
    rt_left = np.array(rt_left)
    rt_right = np.array(rt_right)
    root = np.zeros(N+1)
    itermax_vector = np.log2(rt_right - rt_left) - np.log2(epsilon) + 1000
    if config['stop_criterion'] == 'gu':
        bifunc_vector_gu(cpfunc, rt_left, rt_right, epsilon, w, bsquare, EA,itermax_vector, root, seq,config)
    elif config['stop_criterion'] == 'std':
        bifunc_vector_std(cpfunc, rt_left, rt_right, epsilon, w, bsquare, EA,itermax_vector, root, seq)
    else:
        raise error
    count = 0
    i = 0
    while i<=N:
        if update4value[i]==0:
            C_eigenvalue[i, i] = root[count]
            count += 1
        i += 1

    time1 = time.perf_counter()
    time_eig = time1 - time0
    #recompute beta
    beta = compute_beta_matrix(root, EA, beta)
    time2 = time.perf_counter()
    time_rebeta = time2 - time1
    C_eigenvector = construct_eigenvectors_matrix(beta,EA,root)

    time3 = time.perf_counter()
    time_eigv = time3 - time2
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
    Target: B = [[A, Alpha], [Alpha^T, w]], where A = QA * EA * QA^T

    Steps:
    1. Compute beta = QA^T * Alpha
    2. Deflate the arrowhead problem to reduce size
    3. Solve the reduced secular equation
    4. Recover full eigenvectors using V_defl
    5. Compute final eigenvalues and eigenvectors of B
    """
    beta = QA.T @ Alpha
    EA_diag = np.diag(EA).copy()

    # --- Deflation ---
    EA_sub, beta_sub, w_sub, V_defl, deflated_evals, deflated_indices = \
        deflate_incremental(EA_diag, beta.flatten(), w)

    n_total = N + 1  # total arrowhead matrix size
    n_sub = len(EA_sub)

    if n_sub > 0:
        C_evec_sub, C_eval_sub = evd_of_C(
            np.diag(EA_sub), beta_sub, float(w_sub), n_sub, config
        )
    else:
        C_evec_sub = np.ones((1, 1))
        C_eval_sub = np.array([[float(w_sub)]])

    # --- Recover full eigenvectors using V_defl ---
    # Paper: Q_C = V_defl @ block_diag(Q_sub, I_defl)
    remaining_indices = [i for i in range(n_total) if i not in deflated_indices]

    # Build a combined sorted list to determine correct global ordering
    combined = []
    for i, idx in enumerate(deflated_indices):
        combined.append((deflated_evals[i], True, i, idx))
    for i, idx in enumerate(remaining_indices):
        combined.append((np.diag(C_eval_sub)[i], False, i, idx))
    combined.sort(key=lambda x: x[0])

    Q_block = np.zeros((n_total, n_total))
    C_eigenvalue = np.zeros((n_total, n_total))

    for j_sorted, (val, is_deflated, src_idx, global_idx) in enumerate(combined):
        C_eigenvalue[j_sorted, j_sorted] = val
        if is_deflated:
            Q_block[global_idx, j_sorted] = 1.0
        else:
            row_ptr = 0
            for r in range(n_total):
                if r not in deflated_indices:
                    Q_block[r, j_sorted] = C_evec_sub[row_ptr, src_idx]
                    row_ptr += 1

    C_eigenvector = V_defl @ Q_block

    QB, EB = evd_of_B(QA, C_eigenvector, C_eigenvalue, N)
    return QB, EB


def evd_of_C_standard(EA, z, rho, N, config):
    """
    Solves the secular equation for the Standard Rank-1 Update.
    Equation: 1 + rho * sum( z_i^2 / (d_i - lambda) ) = 0

    Steps:
    1. Determine search intervals using interlacing property
    2. Detect near-degenerate intervals (multiple close poles in one interval)
       and handle them directly
    3. Solve remaining intervals via bisection
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

    update4value = np.zeros(N)
    z_recomputed = compute_z(root, EA, N, rho, z, update4value)

    C_eigenvector = construct_eigenvectors_standard(z_recomputed, EA, root)
    return C_eigenvector, C_eigenvalue


def evd_standard(QA, EA, u, rho, config):
    """
    Main Entry Point for Standard Rank-1 Update EVD.
    Target: A_new = A + rho * u * u^T
    Known: A = QA * EA * QA^T

    Steps:
    1. Project u to z = QA^T * u
    2. Deflate problem (D + rho * z * z^T)
    3. Solve secular equation
    4. Recover eigenvectors via U_defl
    """
    N = len(EA)
    EA_diag = np.diag(EA).copy()
    z = QA.T @ u
    z = z.flatten()

    # --- Deflation ---
    EA_sub, z_sub, rho_sub, U_defl, deflated_evals, deflated_indices = \
        deflate_standard(EA_diag, z, rho)

    n_sub = len(EA_sub)

    if n_sub > 0:
        C_evec_sub, C_eval_sub = evd_of_C_standard(EA_sub, z_sub, rho, n_sub, config)
        solved_vals = np.diag(C_eval_sub)
        solved_vecs = C_evec_sub
    else:
        solved_vals = np.array([])
        solved_vecs = np.zeros((0, 0))

    # --- Recover full eigenvectors using U_defl ---
    # Paper: Q_D = U_defl @ block_diag(Q_sub, I_defl)
    remaining_indices = [i for i in range(N) if i not in deflated_indices]

    # Build a combined sorted list to determine correct global ordering
    # Each entry: (eigenvalue, is_deflated, source_index, global_index_in_unsorted_D)
    combined = []
    for i, idx in enumerate(deflated_indices):
        combined.append((deflated_evals[i], True, i, idx))
    for i, idx in enumerate(remaining_indices):
        combined.append((solved_vals[i], False, i, idx))
    combined.sort(key=lambda x: x[0])

    Q_block = np.zeros((N, N))
    final_evals = np.zeros(N)

    for j_sorted, (val, is_deflated, src_idx, global_idx) in enumerate(combined):
        final_evals[j_sorted] = val
        if is_deflated:
            # Unit vector e_{global_idx} at column j_sorted
            Q_block[global_idx, j_sorted] = 1.0
        else:
            # Place src_idx-th column of solved_vecs into undeflated rows, column j_sorted
            row_ptr = 0
            for r in range(N):
                if r not in deflated_indices:
                    Q_block[r, j_sorted] = solved_vecs[row_ptr, src_idx]
                    row_ptr += 1

    final_evecs_in_D_basis = U_defl @ Q_block

    # Transform from D basis to original basis using QA
    QB = QA @ final_evecs_in_D_basis

    return QB, np.diag(final_evals)