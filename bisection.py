from os import error
import numpy as np
import pymp
from concurrent.futures import ThreadPoolExecutor
from tools import *


def bifunc_relative(mu_low, mu_high, origins, epsilon, w, bsquare, EA, itermax_vector, seq):
    """
    solve the incremental rank-1 secular equation in relative coordinate system using Gu-Eisenstat stopping criterion.

    Parameters:
    - mu_low, mu_high: Search intervals [low, high] for each eigenvalue (relative to origins).
    - origins: Reference origins (Shift) for each eigenvalue. True root = origins[i] + mu[i].
    - epsilon: Machine precision.
    - w: Tip element of the arrowhead matrix (alpha).
    - bsquare: Squared arrow elements (beta_j^2).
    - EA: Diagonal elements.
    - seq: List of eigenvalue indices to compute.

    Returns:
    - mu_roots: Computed offsets mu.
    """

    # Initialize result array
    N_roots = len(mu_low)
    mu_roots = np.zeros(N_roots)

    # Set error bound parameter eta (tr932 suggests eta * N < 0.1)
    # Using conservative value for stability
    eta = epsilon
    N = len(EA) + 1  # Total matrix dimension

    for i in seq:
        low = mu_low[i]
        high = mu_high[i]
        origin = origins[i]
        max_it = int(itermax_vector[i])

        # Precompute d_j - origin (Shifted Diagonals)
        # This is a high-precision operation since d_j and origin are original data
        shifted_EA = EA - origin

        # alpha - origin (Shifted Tip)
        shifted_w = w - origin

        it = 0
        converged = False
        mid = low

        while it < max_it:
            # Bisection midpoint (in mu space)
            mid = (low + high) / 2.0

            denoms = shifted_EA - mid
            with np.errstate(divide='ignore', invalid='ignore'):
                terms = bsquare / denoms
            f_val = mid - shifted_w + np.sum(terms)

            # -----------------------------------------------------------------
            # Compute error bound for Gu-Eisenstat stopping criterion
            # -----------------------------------------------------------------
            # Bound = eta * N * ( |mu| + |w - origin| + sum( |terms| ) )
            sum_abs_terms = np.sum(np.abs(terms))
            bound = eta * N * (abs(mid) + abs(shifted_w) + sum_abs_terms)
            if abs(f_val) <= bound:
                mu_roots[i] = mid
                converged = True
                break

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
    Solve the secular equation for standard rank-1 update in relative coordinate system using Gu-Eisenstat stopping criterion.

    Equation form: f_std(λ) = 1 + ρ Σ |z_i|² / (λ_i - λ) = 0

    Parameters:
    - mu_low, mu_high: Search intervals [low, high] for each eigenvalue (relative to origins).
    - origins: Reference origins (Shift) for each eigenvalue. True root = origins[i] + mu[i].
    - epsilon: Machine precision.
    - rho: Scalar coefficient of rank-1 update.
    - z_square: Squared interaction vector z (|z_i|²).
    - EA: Diagonal elements (λ_i).
    - seq: List of eigenvalue indices to compute.

    Returns:
    - mu_roots: Computed offsets mu.
    """

    # Initialize result array
    N_roots = len(mu_low)
    mu_roots = np.zeros(N_roots)

    # Set error bound parameter eta (tr932 suggests eta * N < 0.1)
    # Using conservative value for stability
    eta = epsilon
    N = len(EA)  # Total matrix dimension

    for i in seq:
        low = mu_low[i]
        high = mu_high[i]
        origin = origins[i]
        max_it = int(itermax_vector[i])

        # Precompute λ_i - origin (Shifted Diagonals)
        # This is a high-precision operation since λ_i and origin are original data
        shifted_EA = EA - origin

        it = 0
        converged = False
        mid = low

        while it < max_it:
            # Bisection midpoint (in mu space)
            mid = (low + high) / 2.0

            denoms = shifted_EA - mid

            with np.errstate(divide='ignore', invalid='ignore'):
                terms = z_square / denoms

            # Secular equation value
            f_val = 1.0 + rho * np.sum(terms)

            # -----------------------------------------------------------------
            # Compute error bound for Gu-Eisenstat stopping criterion
            # -----------------------------------------------------------------
            # Bound = eta * N * ( |rho| * sum( |terms| ) )
            sum_abs_terms = np.sum(np.abs(terms))
            bound = eta * N * (np.abs(rho) * sum_abs_terms)

            if abs(f_val) <= bound:
                mu_roots[i] = mid
                converged = True
                break

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


def bifunc_vector_gu(cpfunc, rt_left, rt_right, epsilon, w, bsquare, EA, itermax_vector, root, seq, config):
    """
    Vectorized bisection method for root finding with gu stopping criterion
    """
    N = len(root)
    origins = np.zeros(N)
    mu_low = np.zeros(N)
    mu_high = np.zeros(N)

    # Simple origin selection strategy: choose the pole closest to the interval midpoint
    # Note: This is not as good as the intelligent selection strategy in evd.py, but sufficient as a compatibility layer
    for i in seq:
        mid_abs = (rt_left[i] + rt_right[i]) / 2.0
        # Find the closest pole as origin
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
    # Return true roots
    for i in seq:
        root[i] = origins[i] + mus[i]

def bifunc(func, left, right, error, w, bsquare, eigenvalue, i, itermax):
    """
    Single-variable bisection method for root finding
    """
    mid = (left + right) / 2
    iter_ = 0
    # Evaluate function at the left endpoint
    f_left = func(left, w, bsquare, eigenvalue, left, right, i)

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
        f_mid = func(mid, w, bsquare, eigenvalue, left, right, i)
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


def bifunc_vector_std(func, left, right, error, w, bsquare, eigenvalue, itermax, root, seq):
    """
    Vectorized bisection method for root finding with standard stopping criterion
    """
    mid = (left + right) / 2
    N = left.size
    f_left = np.zeros(N)
    f_mid = np.zeros(N)
    tmp = np.zeros(N)
    # Evaluate function at the left endpoint
    for i in range(N):
        f_left[i] = func(left[i], w, bsquare, eigenvalue, left[i], right[i], seq[i])

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
                f_mid[i] = func(mid[i], w, bsquare, eigenvalue, left[i], right[i], seq[i])
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


def bifunc_vector_muti2(func, lleft, rright, error, w, bsquare, eigenvalue, itermax, root):
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