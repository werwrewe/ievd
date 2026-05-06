import numpy as np


def deflation_tol(matrix_diag, n):
    """
    Compute the deflation tolerance as described in the paper.
    tol = 2 * n^2 * eps * max(|lambda_1|, |lambda_n|)

    Parameters:
    matrix_diag: diagonal elements of the matrix (1D array)
    n: matrix dimension
    """
    eps = np.finfo(float).eps
    tau = 2.0 * n * n * eps
    return tau * max(abs(matrix_diag[0]), abs(matrix_diag[-1]))


def deflate_incremental(EA, beta, w):
    """
    Full deflation for incremental rank-1 update (arrowhead matrix C).

    C = [[Sigma, beta], [beta^T, w]] where Sigma = diag(EA)

    Implements paper Section 4.2:
    1. Zero-component deflation: |beta_i| <= tol -> shrink EA[i], e_i
    2. Repeated diagonal deflation: |EA[i] - EA[i+1]| <= tol ->
       Givens rotation to zero out beta_{i+1}, then shrink the
       rotated eigenvalue.

    The cumulative transformation matrix V_defl (size (n+1) x (n+1))
    is maintained for eigenvector recovery.

    Parameters:
    EA: sorted diagonal eigenvalues of A (1D array, length n)
    beta: projection vector QA^T @ Alpha (1D array, length n)
    w: scalar tip element

    Returns:
    EA_sub: un-deflated diagonal elements (1D array)
    beta_sub: un-deflated beta components (1D array)
    w: unchanged w
    V_defl: cumulative transformation matrix ((n+1) x (n+1))
    deflated_evals: array of deflated eigenvalues
    deflated_indices: list of column indices in V_defl corresponding to
                      deflated eigenvectors (unit vectors)
    """
    n = len(EA)
    n_full = n + 1  # full arrowhead matrix dimension
    EA = np.asarray(EA, dtype=float).copy()
    beta = np.asarray(beta, dtype=float).copy().flatten()
    w = float(w)

    n_current = n
    tol = deflation_tol(EA, n_full)

    V_defl = np.eye(n_full)
    deflated_evals = []
    deflated_indices = []  # indices in the FULL space (0..n-1, not n)

    active = list(range(n_current))

    # --- Phase 1: Zero-component deflation ---
    changed = True
    while changed:
        changed = False
        tol = deflation_tol(EA, n_full)
        to_remove = []
        for idx_in_active, i_global in enumerate(active):
            if abs(beta[idx_in_active]) <= tol:
                deflated_evals.append(EA[idx_in_active])
                deflated_indices.append(i_global)
                to_remove.append(idx_in_active)
                changed = True

        if to_remove:
            for idx_in_active in sorted(to_remove, reverse=True):
                active.pop(idx_in_active)
                EA = np.delete(EA, idx_in_active)
                beta = np.delete(beta, idx_in_active)
            n_current = len(EA)
            if n_current == 0:
                break

    # --- Phase 2: Repeated diagonal deflation ---
    if n_current > 0:
        changed = True
        while changed:
            changed = False
            if n_current <= 1:
                break

            tol_diag = deflation_tol(EA, n_full)
            i = 0
            while i < n_current - 1:
                if abs(EA[i] - EA[i+1]) <= tol_diag:
                    r = np.hypot(beta[i], beta[i+1])
                    if r < np.finfo(float).eps:
                        c, s = 1.0, 0.0
                    else:
                        # Real Givens: G = [[c, s], [-s, c]]
                        # G^T @ [beta_i, beta_{i+1}]^T = [r, 0]^T
                        # => c = beta_i/r, s = -beta_{i+1}/r
                        c = beta[i] / r
                        s = -beta[i+1] / r

                    # Apply rotation: G^T @ [beta_i, beta_{i+1}]^T = [r, 0]^T
                    beta_i_old = beta[i]
                    beta_ip1_old = beta[i+1]
                    beta[i] = c * beta_i_old - s * beta_ip1_old      # = r
                    beta[i+1] = s * beta_i_old + c * beta_ip1_old   # = 0

                    # Diagonal block after G^T @ diag(EA[i], EA[i+1]) @ G:
                    diag_i_new = c*c * EA[i] + s*s * EA[i+1]

                    # Deflated eigenvalue: |s|^2*EA[i] + |c|^2*EA[i+1]  (at pos i+1)
                    deflated_eval = s*s * EA[i] + c*c * EA[i+1]
                    deflated_evals.append(deflated_eval)

                    # The deflated eigenvector is e_{i+1} in the rotated basis
                    # -> global index = active[i+1]
                    deflated_global_idx = active[i+1]
                    deflated_indices.append(deflated_global_idx)

                    # Update V_defl with Givens in full space
                    g_i = active[i]
                    g_j = active[i+1]
                    G = np.eye(n_full)
                    G[g_i, g_i] = c
                    G[g_i, g_j] = s
                    G[g_j, g_i] = -s
                    G[g_j, g_j] = c
                    V_defl = V_defl @ G

                    # Remove deflated entry
                    EA[i] = diag_i_new
                    EA = np.delete(EA, i+1)
                    beta = np.delete(beta, i+1)
                    active.pop(i+1)
                    n_current -= 1
                    changed = True

                    # Check if new beta[i] is now zero
                    if n_current > 0 and abs(beta[i]) <= tol:
                        deflated_evals.append(EA[i])
                        deflated_indices.append(active[i])
                        EA = np.delete(EA, i)
                        beta = np.delete(beta, i)
                        active.pop(i)
                        n_current -= 1
                    break
                i += 1

            # Re-check zero-component after a round of diagonal deflation
            if n_current > 0:
                tol = deflation_tol(EA, n_full)
                to_remove = []
                for idx_in_active in range(n_current):
                    if abs(beta[idx_in_active]) <= tol:
                        deflated_evals.append(EA[idx_in_active])
                        deflated_indices.append(active[idx_in_active])
                        to_remove.append(idx_in_active)
                        changed = True
                for idx_in_active in sorted(to_remove, reverse=True):
                    active.pop(idx_in_active)
                    EA = np.delete(EA, idx_in_active)
                    beta = np.delete(beta, idx_in_active)
                n_current = len(EA)

    return EA, beta, w, V_defl, np.array(deflated_evals), deflated_indices


def deflate_standard(EA, z, rho):
    """
    Full deflation for standard rank-1 update.
    D = Sigma + rho * z * z^T, where Sigma = diag(EA)

    Implements paper Section 4.1:
    1. Zero-component deflation: |z_i| <= tol -> shrink EA[i], e_i
    2. Repeated diagonal deflation: |EA[i] - EA[i+1]| <= tol ->
       Givens rotation to zero out z_{i+1}, then shrink.

    The cumulative transformation matrix U_defl (size n x n)
    is maintained for eigenvector recovery.

    Parameters:
    EA: sorted diagonal eigenvalues of A (1D array, length n)
    z: QA^T @ u (1D array, length n)
    rho: scalar coefficient

    Returns:
    EA_sub: un-deflated diagonal elements (1D array)
    z_sub: un-deflated z components (1D array)
    rho: unchanged rho
    U_defl: cumulative transformation matrix (n x n)
    deflated_evals: array of deflated eigenvalues
    deflated_indices: list of column indices in U_defl for deflated eigenvectors
    """
    n = len(EA)
    EA = np.asarray(EA, dtype=float).copy()
    z = np.asarray(z, dtype=float).copy().flatten()
    rho = float(rho)

    n_current = n
    tol = deflation_tol(EA, n)

    U_defl = np.eye(n)
    deflated_evals = []
    deflated_indices = []

    active = list(range(n_current))

    # --- Phase 1: Zero-component deflation ---
    changed = True
    while changed:
        changed = False
        tol = deflation_tol(EA, n)
        to_remove = []
        for idx_in_active, i_global in enumerate(active):
            if abs(z[idx_in_active]) <= tol:
                deflated_evals.append(EA[idx_in_active])
                deflated_indices.append(i_global)
                to_remove.append(idx_in_active)
                changed = True

        if to_remove:
            for idx_in_active in sorted(to_remove, reverse=True):
                active.pop(idx_in_active)
                EA = np.delete(EA, idx_in_active)
                z = np.delete(z, idx_in_active)
            n_current = len(EA)
            if n_current == 0:
                break

    # --- Phase 2: Repeated diagonal deflation ---
    if n_current > 0:
        changed = True
        while changed:
            changed = False
            if n_current <= 1:
                break

            tol_diag = deflation_tol(EA, n)
            i = 0
            while i < n_current - 1:
                if abs(EA[i] - EA[i+1]) <= tol_diag:
                    r = np.hypot(z[i], z[i+1])
                    if r < np.finfo(float).eps:
                        c, s = 1.0, 0.0
                    else:
                        # Real Givens: G = [[c, s], [-s, c]]
                        # G^T @ [z_i, z_{i+1}]^T = [r, 0]^T
                        # => c = z_i/r, s = -z_{i+1}/r
                        c = z[i] / r
                        s = -z[i+1] / r

                    # Apply rotation: G^T @ [z_i, z_{i+1}]^T = [r, 0]^T
                    z_i_old = z[i]
                    z_ip1_old = z[i+1]
                    z[i] = c * z_i_old - s * z_ip1_old       # = r
                    z[i+1] = s * z_i_old + c * z_ip1_old    # = 0

                    # Diagonal block after G^T @ diag(EA[i], EA[i+1]) @ G:
                    diag_i_new = c*c * EA[i] + s*s * EA[i+1]

                    # Deflated eigenvalue: |s|^2*EA[i] + |c|^2*EA[i+1]  (at pos i+1)
                    deflated_eval = s*s * EA[i] + c*c * EA[i+1]
                    deflated_evals.append(deflated_eval)

                    deflated_global_idx = active[i+1]
                    deflated_indices.append(deflated_global_idx)

                    # Update U_defl with Givens in full space
                    g_i = active[i]
                    g_j = active[i+1]
                    G = np.eye(n)
                    G[g_i, g_i] = c
                    G[g_i, g_j] = s
                    G[g_j, g_i] = -s
                    G[g_j, g_j] = c
                    U_defl = U_defl @ G

                    # Remove deflated entry
                    EA[i] = diag_i_new
                    EA = np.delete(EA, i+1)
                    z = np.delete(z, i+1)
                    active.pop(i+1)
                    n_current -= 1
                    changed = True

                    # Check if new z[i] is now zero
                    if n_current > 0 and abs(z[i]) <= tol:
                        deflated_evals.append(EA[i])
                        deflated_indices.append(active[i])
                        EA = np.delete(EA, i)
                        z = np.delete(z, i)
                        active.pop(i)
                        n_current -= 1
                    break
                i += 1

            # Re-check zero-component
            if n_current > 0:
                tol = deflation_tol(EA, n)
                to_remove = []
                for idx_in_active in range(n_current):
                    if abs(z[idx_in_active]) <= tol:
                        deflated_evals.append(EA[idx_in_active])
                        deflated_indices.append(active[idx_in_active])
                        to_remove.append(idx_in_active)
                        changed = True
                for idx_in_active in sorted(to_remove, reverse=True):
                    active.pop(idx_in_active)
                    EA = np.delete(EA, idx_in_active)
                    z = np.delete(z, idx_in_active)
                n_current = len(EA)

    return EA, z, rho, U_defl, np.array(deflated_evals), deflated_indices
