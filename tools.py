import numpy as np
import yaml

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
    向量化 Gu-Eisenstat beta 重构
    """
    # 1. 分子差分矩阵 (roots_j - poles_i) -> 形状 (N, N+1)
    num_diffs = roots[np.newaxis, :] - poles[:, np.newaxis]

    # 2. 分母差分矩阵 (poles_j - poles_i) -> 形状 (N, N)
    # 需要处理 i=j 的奇异点。在原代码中是将该项设为 1 (即对数设为 0)
    den_diffs = poles[np.newaxis, :] - poles[:, np.newaxis]
    np.fill_diagonal(den_diffs, 1.0) # 填充对角线为 1，避免除零

    # 3. Log-Sum 运算
    # sum_j log|num_ij|
    sum_log_num = np.sum(np.log(np.abs(num_diffs)), axis=1)
    # sum_k log|den_ik|
    sum_log_den = np.sum(np.log(np.abs(den_diffs)), axis=1)

    # 4. 组合并还原
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
    Recomputes z values for Standard Rank-1 Update based on the formula (4.26):
    z_i = sqrt( (1/rho) * prod(new_lambda_j - old_lambda_i) / prod_{k!=i}(old_lambda_k - old_lambda_i) )

    Args:
        new_eigenvalues: Array of new eigenvalues (length N)
        old_eigenvalues: Array of old eigenvalues (length N)
        N: Matrix dimension
        rho: Update scalar
        z: Original z vector (for sign restoration)
        update4vec: Optional mask to skip recomputation for stable/converged values
    """
    z_new = np.zeros(N)

    for i in range(N):
        # Skip recomputation if the value is already converged/stable
        if update4vec is not None and update4vec[i] == 1:
            z_new[i] = np.abs(z[i])
            continue

        lambda_i = old_eigenvalues[i]

        # Calculate differences for Numerator terms: (\hat{\lambda}_j - \lambda_i)
        diff_new = new_eigenvalues - lambda_i

        # Calculate differences for Denominator terms: (\lambda_k - \lambda_i)
        diff_old = old_eigenvalues - lambda_i

        # We construct the term-wise ratios to avoid numerical overflow in the product.
        # We pair terms to keep ratios close to 1.
        # Ratio term k:
        # If k != i: (new_lambda_k - old_lambda_i) / (old_lambda_k - old_lambda_i)
        # If k == i: (new_lambda_i - old_lambda_i) / rho  (This handles the 1/rho factor and the i-th numerator term)

        ratios = np.zeros(N)

        # Mask for indices where k != i
        mask = np.arange(N) != i

        # Compute ratios for k != i
        # Note: We assume deflation has handled cases where denominators would be zero.
        ratios[mask] = diff_new[mask] / diff_old[mask]

        # Compute ratio for k == i
        ratios[i] = diff_new[i] / rho

        # Compute the product using logarithm to ensure numerical stability
        # log(product) = sum(log(abs(ratios)))
        # z_i = sqrt(exp(log(product)))

        log_prod = np.sum(np.log(np.abs(ratios)))
        z_new[i] = np.sqrt(np.exp(log_prod))

    # Restore the signs from the original z vector
    z_new *= np.sign(z)

    return z_new


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
    一次性构建所有特征向量矩阵
    """
    # 构建分母矩阵 (N, N+1)
    # poles (N,), roots (N+1,)

    denom = poles[:, np.newaxis] - roots[np.newaxis, :]
    # 构建分子并执行除法
    # beta_hat (N,) 广播列向量
    V_top = -beta_hat[:, np.newaxis] / denom

    # 构建底部行 (1, N+1)
    V_bot = np.ones((1, V_top.shape[1]))

    # 堆叠矩阵
    V = np.vstack([V_top, V_bot])

    # 列归一化
    # axis=0 表示沿列计算范数
    norms = np.linalg.norm(V, axis=0)
    V_normalized = V / norms[np.newaxis, :]

    return V_normalized

def construct_eigenvectors_standard(z_recomputed, EA, roots):
    """
    一次性构建所有特征向量矩阵 (标准秩1修正)

    参数:
    z_recomputed: (N,) 重构后的更新向量 \hat{z}
    EA: (N,) 原特征值 \lambda (poles)
    roots: (N,) 新特征值 \hat{\lambda}

    返回:
    V_normalized: (N, N) 归一化后的特征向量矩阵
    """
    # 构建分母矩阵 (N, N)
    # EA 为列扩展，roots 为行扩展，形成 N x N 矩阵
    denom = EA[:, np.newaxis] - roots[np.newaxis, :]

    # 构建分子并执行除法
    # z_recomputed (N,) 广播列向量
    V = z_recomputed[:, np.newaxis] / denom

    # 列归一化
    # axis=0 表示沿列计算 L2 范数
    norms = np.linalg.norm(V, axis=0)
    V_normalized = V / norms[np.newaxis, :]

    return V_normalized

