from os import error
import numpy as np
import pymp
from concurrent.futures import ThreadPoolExecutor
from tools import *



def bifunc_relative(mu_low, mu_high, origins, epsilon, w, bsquare, EA, itermax_vector, seq):
    """
    使用 Gu-Eisenstat 停止准则在相对坐标系下求解久期方程。

    参数:
    - mu_low, mu_high: 每个特征值的搜索区间 [low, high] (相对于 origins).
    - origins: 每个特征值的参考原点 (Shift). 真根 = origins[i] + mu[i].
    - epsilon: 机器精度.
    - w: 箭头矩阵的尖端元素 (alpha).
    - bsquare: 箭杆元素的平方 (z_j^2).
    - EA: 对角元素 (d_j).
    - seq: 需要计算的特征值索引列表.

    返回:
    - mu_roots: 计算出的偏移量 mu.
    """

    # 初始化结果数组
    N_roots = len(mu_low)
    mu_roots = np.zeros(N_roots)

    # 设定误差界限参数 eta (tr932 建议 eta * N < 0.1)
    # 这里取 conservative 值以保证稳定性
    eta = epsilon
    N = len(EA) + 1  # 矩阵总维度

    for i in seq:
        low = mu_low[i]
        high = mu_high[i]
        origin = origins[i]
        max_it = int(itermax_vector[i])

        # 预先计算 d_j - origin (Shifted Diagonals)
        # 这是一个高精度操作，因为 d_j 和 origin 都是原始数据
        shifted_EA = EA - origin

        # alpha - origin (Shifted Tip)
        shifted_w = w - origin

        it = 0
        converged = False
        mid = low

        while it < max_it:
            # 二分中点 (在 mu 空间)
            mid = (low + high) / 2.0

            # -----------------------------------------------------------------
            # 1. 计算久期方程的值 f(mu)
            # -----------------------------------------------------------------
            # 我们需要计算: f(mu) = (w - origin - mu) + sum( z^2 / (d_j - origin - mu) )
            # 注意: w 在方程中是 lambda - w... 即 (origin + mu) - w
            # 所以项是: mu - (w - origin) + sum...

            # 分母计算: (d_j - origin) - mu
            # 当 d_j == origin 时，shifted_EA 为 0，分母精确为 -mu
            # 当 d_j!= origin 时，shifted_EA 为大数，减去小数 mu 不损失精度
            denoms = shifted_EA - mid

            # 避免除以零（极罕见情况，二分法通常不会直接命中极点）
            # 在 numpy 中处理 inf
            with np.errstate(divide='ignore', invalid='ignore'):
                terms = bsquare / denoms

            # 久期方程值
            # f(lambda) = lambda - w + sum(...)
            # f(origin + mu) = (origin + mu) - w + sum(...)
            #                = mu - (w - origin) + sum(...)
            f_val = mid - shifted_w + np.sum(terms)

            # -----------------------------------------------------------------
            # 2. 计算 Gu-Eisenstat 停止准则的误差界 (Bound)
            # -----------------------------------------------------------------
            # Bound = eta * N * ( |mu| + |w - origin| + sum( |terms| ) )
            sum_abs_terms = np.sum(np.abs(terms))
            bound = eta * N * (abs(mid) + abs(shifted_w) + sum_abs_terms)

            # -----------------------------------------------------------------
            # 3. 检查停止条件
            # -----------------------------------------------------------------
            if abs(f_val) <= bound:
                mu_roots[i] = mid
                converged = True
                break

            # -----------------------------------------------------------------
            # 4. 更新区间
            # -----------------------------------------------------------------
            # f(lambda) 在极点间单调递增
            if f_val > 0:
                high = mid
            else:
                low = mid

            it += 1

        if not converged:
            mu_roots[i] = (low + high) / 2.0

    return mu_roots

def bifunc_relative_std(mu_low, mu_high, origins, epsilon, rho, z_square, EA, itermax_vector, seq):
    """
    使用 Gu-Eisenstat 停止准则在相对坐标系下求解标准秩1更新的久期方程。

    方程形式: f_std(λ) = 1 + ρ Σ |z_i|² / (λ_i - λ) = 0

    参数:
    - mu_low, mu_high: 每个特征值的搜索区间 [low, high] (相对于 origins).
    - origins: 每个特征值的参考原点 (Shift). 真根 = origins[i] + mu[i].
    - epsilon: 机器精度.
    - rho: 秩1更新的标量系数.
    - z_square: 相互作用向量 z 的平方 (|z_i|²).
    - EA: 对角元素 (λ_i).
    - seq: 需要计算的特征值索引列表.

    返回:
    - mu_roots: 计算出的偏移量 mu.
    """

    # 初始化结果数组
    N_roots = len(mu_low)
    mu_roots = np.zeros(N_roots)

    # 设定误差界限参数 eta (tr932 建议 eta * N < 0.1)
    # 这里取 conservative 值以保证稳定性
    eta = epsilon
    N = len(EA)  # 矩阵总维度

    for i in seq:
        low = mu_low[i]
        high = mu_high[i]
        origin = origins[i]
        max_it = int(itermax_vector[i])

        # 预先计算 λ_i - origin (Shifted Diagonals)
        # 这是一个高精度操作，因为 λ_i 和 origin 都是原始数据
        shifted_EA = EA - origin

        it = 0
        converged = False
        mid = low

        while it < max_it:
            # 二分中点 (在 mu 空间)
            mid = (low + high) / 2.0

            # -----------------------------------------------------------------
            # 1. 计算久期方程的值 f(mu)
            # -----------------------------------------------------------------
            # 我们需要计算: f(mu) = 1 + ρ Σ |z_i|² / (λ_i - (origin + mu))
            # 换元后: λ_i - (origin + mu) = (λ_i - origin) - mu = shifted_EA - mu

            # 分母计算: (λ_i - origin) - mu
            # 当 λ_i == origin 时，shifted_EA 为 0，分母精确为 -mu
            # 当 λ_i != origin 时，shifted_EA 为大数，减去小数 mu 不损失精度
            denoms = shifted_EA - mid

            # 避免除以零（极罕见情况，二分法通常不会直接命中极点）
            # 在 numpy 中处理 inf
            with np.errstate(divide='ignore', invalid='ignore'):
                terms = z_square / denoms

            # 久期方程值
            f_val = 1.0 + rho * np.sum(terms)

            # -----------------------------------------------------------------
            # 2. 计算 Gu-Eisenstat 停止准则的误差界 (Bound)
            # -----------------------------------------------------------------
            # Bound = eta * N * ( |rho| * sum( |terms| ) )
            sum_abs_terms = np.sum(np.abs(terms))
            bound = eta * N * (np.abs(rho) * sum_abs_terms)

            # -----------------------------------------------------------------
            # 3. 检查停止条件
            # -----------------------------------------------------------------
            if abs(f_val) <= bound:
                mu_roots[i] = mid
                converged = True
                break

            # -----------------------------------------------------------------
            # 4. 更新区间
            # -----------------------------------------------------------------
            # f(λ) 在极点间的单调性取决于 rho 的符号
            # 当 rho > 0 时，f(λ) 在每个区间内单调递增
            # 当 rho < 0 时，f(λ) 在每个区间内单调递减
            if rho > 0:
                if f_val > 0:
                    high = mid
                else:
                    low = mid
            else:
                if f_val > 0:
                    low = mid
                else:
                    high = mid

            it += 1

        if not converged:
            mu_roots[i] = (low + high) / 2.0

    return mu_roots

# -------------------------------------------------------------------------
# 为了保持兼容性，我们可以保留旧接口的壳，但强烈建议直接使用新接口
# -------------------------------------------------------------------------
def bifunc_vector_gu(cpfunc, rt_left, rt_right, epsilon, w, bsquare, EA, itermax_vector, root, seq,config):
    """
    旧接口的适配器。如果外部代码强行调用此函数，
    我们将其转换为相对坐标调用以保证精度。
    """
    N = len(root)
    origins = np.zeros(N)
    mu_low = np.zeros(N)
    mu_high = np.zeros(N)

    # 简单的原点选择策略：选择区间中点最近的极点
    # 注意：这不如 evd.py 中的智能选择策略好，但作为兼容层够用了
    for i in seq:
        mid_abs = (rt_left[i] + rt_right[i]) / 2.0
        # 寻找最近的极点作为原点
        dist = np.abs(EA - mid_abs)
        idx = np.argmin(dist)
        origins[i] = EA[idx]

        mu_low[i] = rt_left[i] - origins[i]
        mu_high[i] = rt_right[i] - origins[i]
    if config['target_type'] == 'incremental_rank1':
        mus = bifunc_relative(mu_low, mu_high, origins, epsilon, w, bsquare, EA, itermax_vector, seq)
    elif config['target_type'] == 'standard_rank1':
        mus = bifunc_relative_std(mu_low, mu_high, origins, epsilon, w, bsquare, EA, itermax_vector, seq)
    else:
        raise error
    # 返回真根
    for i in seq:
        root[i] = origins[i] + mus[i]

def bifunc(func, left, right, error, w, bsquare, eigenvalue,i,itermax):
    """
    Single-variable bisection method for root finding
    """
    mid = (left + right) / 2
    iter_ = 0
    # Evaluate function at the left endpoint
    f_left = func(left, w, bsquare, eigenvalue, left, right,i)

    # Get machine epsilon for double precision
    eps = np.finfo(float).eps

    # Bisection loop with new termination condition
    while iter_ < itermax:
        # Calculate termination condition
        interval_length = abs(right - left)
        max_val = max(1.0, abs(left), abs(right))
        termination_condition = 2 * eps * max_val

        if interval_length <= termination_condition:
            break

        iter_ += 1
        f_mid = func(mid, w, bsquare, eigenvalue, left, right,i)
        tmp = f_left * f_mid
        if  tmp < 0:
            right = mid
        elif tmp > 0:
            f_left = f_mid
            left = mid
        else :
            return mid
        mid = (left + right) / 2

    root = mid
    return root



def bifunc_vector_std(func, left, right, error, w, bsquare, eigenvalue,itermax, root,seq):
    mid = (left + right) / 2
    N = left.size
    f_left = np.zeros(N)
    f_mid = np.zeros(N)
    tmp = np.zeros(N)
    # Evaluate function at the left endpoint
    for i in range(N):
        f_left[i] = func(left[i], w, bsquare, eigenvalue, left[i], right[i],seq[i])

    # Get machine epsilon for double precision
    eps = np.finfo(float).eps

    NB = N
    ci = int((N - 1) / NB)
    res = N % NB
    for k in range(ci + 1):
        begin = k * NB
        end = min(N, begin + NB)
        #print(begin, end)
        itmax = np.linalg.norm(x = itermax[begin:end], ord = np.inf)
        iter_ = 0

        # Bisection loop with new termination condition
        while iter_ < itmax:
            # Calculate termination condition for each element
            interval_lengths = np.abs(right[begin:end] - left[begin:end])
            max_vals = np.maximum(1.0, np.maximum(np.abs(left[begin:end]), np.abs(right[begin:end])))
            termination_conditions = 2 * eps * max_vals

            # Check if all elements meet the termination condition
            if np.all(interval_lengths <= termination_conditions):
                break

            iter_ += 1
            for i in range(begin,end):
                f_mid[i] = func(mid[i], w, bsquare, eigenvalue, left[i], right[i],seq[i])
            tmp[begin:end] = f_left[begin:end] * f_mid[begin:end]
            for i in range(begin,end):
                if  tmp[i] < 0:
                    right[i] = mid[i]
                elif tmp[i] > 0:
                    f_left[i] = f_mid[i]
                    left[i] = mid[i]
                else:
                    f_left[i] = 0.
                    right[i] = mid[i]
                    left[i] = mid[i]

            mid[begin:end] = (left[begin:end] + right[begin:end]) / 2
    for i in range(N):
        root[i] = mid[i]


def process_chunk(args):
    """
    Process a single chunk for parallel execution
    """
    func, left, right, error, w, bsquare, eigenvalue, itermax, begin, end = args
    N = left.size

    f_left = np.zeros(N)
    f_mid = np.zeros(N)
    tmp = np.zeros(N)
    mid = np.zeros(N)

    # Initialize f_left and mid
    for i in range(begin, end):
        f_left[i] = func(left[i], w, bsquare, eigenvalue, left[i], right[i], i)
        mid[i] = (left[i] + right[i]) / 2

    itmax = np.linalg.norm(x=itermax[begin:end], ord=np.Inf)
    iter_ = 0

    while iter_ < itmax and np.linalg.norm(x=(right[begin:end] - left[begin:end]) / 2, ord=np.Inf) > error:
        iter_ += 1

        for i in range(begin, end):
            f_mid[i] = func(mid[i], w, bsquare, eigenvalue, left[i], right[i], i)

        for i in range(begin, end):
            tmp[i] = f_left[i] * f_mid[i]

        for i in range(begin, end):
            if tmp[i] < 0:
                right[i] = mid[i]
            elif tmp[i] > 0:
                f_left[i] = f_mid[i]
                left[i] = mid[i]
            else:
                f_left[i] = 0.
                right[i] = mid[i]
                left[i] = mid[i]

        for i in range(begin, end):
            mid[i] = (left[i] + right[i]) / 2

    return mid

def bifunc_vector_muti(func, lleft, rright, error, w, bsquare, eigenvalue, itermax, root):
    """
    Parallel vectorized bisection method for root finding (multi-threaded)
    """
    N = lleft.size

    # Create numpy arrays
    left = np.array(lleft)
    right = np.array(rright)
    mid = np.zeros(N)

    NB = 128
    ci = int((N - 1) / NB)
    res = N % NB

    # Prepare chunks for parallel processing
    chunks = []
    for k in range(ci + 1):
        begin = k * NB
        end = min(N, begin + NB)
        chunks.append((func, left, right, error, w, bsquare, eigenvalue, itermax, begin, end))

    # Use ThreadPoolExecutor for parallel processing
    max_workers = 24  # Adjust based on your system
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_chunk, chunks))

    # Combine results
    for k, result in enumerate(results):
        begin = k * NB
        end = min(N, begin + NB)
        mid[begin:end] = result[begin:end]

    # Copy results to root array
    for i in range(N):
        root[i] = mid[i]


def bifunc_vector_muti2(func, lleft, rright, error, w, bsquare, eigenvalue,itermax, root):
    """
    Parallel vectorized bisection method for root finding (multi-threaded)
    """
    N = lleft.size

    f_left = pymp.shared.array(N)
    f_mid = pymp.shared.array(N)
    tmp = pymp.shared.array(N)
    #root = pymp.shared.array(N)
    mid = pymp.shared.array(N)
    #mid = (left + right) / 2
    # Evaluate function at the left endpoint
    #for i in range(N):
    #    f_left[i] = func(left[i], w, bsquare, eigenvalue, left[i], right[i],i)
    left = pymp.shared.array(N)
    right = pymp.shared.array(N)
    for i in range(N):
        left[i] =  lleft[i]
        right[i] = rright[i]
    NB = 128
    ci = int((N - 1) / NB)
    res = N % NB

    with pymp.Parallel(24) as p:
        for k in p.range(0, ci + 1):

            begin = k * NB
            end = min(N, begin + NB)
            for i in range(begin,end):
                f_left[i] = func(left[i], w, bsquare, eigenvalue, left[i], right[i],i)
            mid[begin:end] = (left[begin:end] + right[begin:end]) / 2
            #print(begin, end)
            itmax = np.linalg.norm(x = itermax[begin:end], ord = np.Inf)
            iter_ = 0
            while iter_ < itmax and np.linalg.norm(x = (right[begin:end] - left[begin:end]) / 2, ord = np.Inf) > error:
                iter_ += 1

                for i in range(begin,end):

                    f_mid[i] = func(mid[i], w, bsquare, eigenvalue, left[i], right[i],i)
                tmp[begin:end] = f_left[begin:end] * f_mid[begin:end]
                for i in range(begin,end):
                    if  tmp[i] < 0:
                        right[i] = mid[i]
                    elif tmp[i] > 0:
                        f_left[i] = f_mid[i]
                        left[i] = mid[i]
                    else :
                        print("!!!!!!")
                        f_left[i] = 0.
                        right[i] = mid[i]
                        left[i] = mid[i]
                mid[begin:end] = (left[begin:end] + right[begin:end]) / 2
            #print("iter_=", iter_)
    for i in range(N):
        #print(i, mid[i])
        root[i] = mid[i]