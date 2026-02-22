"""
Defines functions related to matrix generation, including generating Laplacian matrices and random matrices.
"""
import numpy as np
import time
import threading
from scipy.linalg import lapack
# th = threading.Semaphore(1)

def generate_laplacian_matrix1(N):
    adj_matrix = np.random.randint(0, 2, size=(N, N))
    adj_matrix = (adj_matrix + adj_matrix.T) // 2

    degrees = np.sum(adj_matrix, axis=1)

    degree_matrix = np.diag(degrees)

    laplacian_matrix = degree_matrix - adj_matrix

    return laplacian_matrix

def generate_matrix(N, mode, seed=None, cond=1.0e4):
    """
    生成用于数值线性代数测试的对称矩阵。

    参数:
    N (int): 矩阵维度。
    mode (int): 矩阵生成模式 (0-8)。
    seed (int): 随机数种子。
    cond (float): 条件数 (仅对 Mode 6, 7, 8 有效)。

    模式说明:
    [物理/结构测试]
    0: Random Symmetric (均匀分布 [-1, 1]) - 一般性测试
    1: Wilkinson Matrix W_N^+ (极近特征值) - 分离度测试
    2: Glued Wilkinson Matrix (特征值簇) - 秩一修正收缩测试
    3: Toeplitz [1, 2, 1] (离散拉普拉斯) - 物理/PDE背景
    4: Weak Diag Dominant [1, mu, 1] - 非对角占优
    5: Strong Diag Dominant [0.01, 1, 0.01] - 强对角占优

    [特征值分布测试 (LAPACK 风格)]
    6: Geometric Distribution (等比分布) - 极小特征值测试
    7: Arithmetic Distribution (等差分布) - 均匀间隙测试
    8: Log-Uniform Distribution (对数均匀) - 多尺度/混合精度测试
    """

    if seed is not None:
        np.random.seed(seed)

    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    # --- Mode 0: Random Symmetric Uniform [-1, 1] ---
    if mode == 0:
        # 生成全随机矩阵，然后对称化
        A = np.random.uniform(-1, 1, (N, N))
        A = (A + A.T) / 2
        return A

    # --- Mode 1: Wilkinson Matrix W_N^+ ---
    elif mode == 1:
        # 对角线: |(N-1)/2 - i|, 次对角线: 1
        # 注意：Python索引从0开始，中心点调整为 (N-1)/2
        diag = np.abs((N - 1) / 2 - np.arange(N))
        off_diag = np.ones(N - 1)
        A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return A

    # --- Mode 2: Glued Wilkinson Matrix ---
    elif mode == 2:
        # 为了适应任意 N，我们将两个 W_{N/2}^+ 粘合在一起
        # 如果 N 太小，直接返回 Wilkinson
        if N < 4:
            return generate_matrix(N, 1)

        n1 = N // 2
        n2 = N - n1

        # 生成两个块
        W1 = generate_matrix(n1, 1) # 递归调用生成 Wilkinson
        W2 = generate_matrix(n2, 1)

        # 构造分块对角矩阵
        A = np.zeros((N, N))
        A[:n1, :n1] = W1
        A[n1:, n1:] = W2

        # "Glue": 在连接处添加微小扰动 (通常是 1e-6 或更小)
        glue = 1e-6
        A[n1-1, n1] = glue
        A[n1, n1-1] = glue
        return A

    # --- Mode 3: Toeplitz Matrix [1, 2, 1] ---
    elif mode == 3:
        # 主对角线 2，次对角线 1
        diag = 2 * np.ones(N)
        off_diag = np.ones(N - 1)
        A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return A

    # --- Mode 4: Weakly Diagonally Dominant [1, mu_i, 1] ---
    elif mode == 4:
        # 次对角线 1 (强耦合)
        # 主对角线 mu_i = i * 10^-6 (微弱变化)
        indices = np.arange(1, N + 1) # 1 to N
        diag = indices * 1e-6
        off_diag = np.ones(N - 1)
        A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return A

    # --- Mode 5: Strongly Diagonally Dominant [0.01, 1+mu, 0.01] ---
    elif mode == 5:
        # 次对角线 0.01
        # 主对角线 1 + i * 10^-6
        indices = np.arange(1, N + 1)
        diag = 1.0 + indices * 1e-6
        off_diag = 0.01 * np.ones(N - 1)
        A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return A

    # --- Modes 6-8: Specified Eigenvalue Distributions (LAPACK Style) ---
    elif mode in [6, 7, 8]:
        # 1. 生成特征值向量 D
        D = np.zeros(N)

        if mode == 6: # Geometric (Mode 3 in LAPACK)
            # D(i) = cond^(-(i)/(N-1))
            # 范围: [1, 1/cond]
            for i in range(N):
                power = -float(i) / (N - 1) if N > 1 else 0
                D[i] = cond ** power

        elif mode == 7: # Arithmetic (Mode 4 in LAPACK)
            # D(i) = 1 - (i)/(N-1) * (1 - 1/cond)
            # 均匀分布在 [1/cond, 1]
            slope = (1.0 - 1.0/cond) / (N - 1) if N > 1 else 0
            for i in range(N):
                D[i] = 1.0 - i * slope

        elif mode == 8: # Log-Uniform (Mode 5 in LAPACK)
            # log(D) 均匀分布在 [log(1/cond), 0]
            log_min = np.log(1.0/cond)
            log_max = 0.0
            random_logs = np.random.uniform(log_min, log_max, N)
            D = np.exp(random_logs)
            # 为了更好的测试，通常会让特征值排序或乱序，这里保持随机乱序

        # 2. 生成随机正交矩阵 Q (使用 QR 分解)
        # 生成高斯随机矩阵
        X = np.random.randn(N, N)
        Q, _ = np.linalg.qr(X)

        # 3. 构造对称矩阵 A = Q D Q^T
        A = Q @ np.diag(D) @ Q.T

        # 确保完全对称 (消除浮点误差)
        A = (A + A.T) / 2
        return A
    elif mode == 9:
        return generate_laplacian_matrix1(N)
    else:
        raise ValueError(f"Unknown mode: {mode}")