import numpy as np


def bifunc(func, left, right, error, w, bsquare, eigenvalue,i,itermax):
    """
    Single-variable bisection method for root finding
    """
    mid = (left + right) / 2
    iter_ = 0
    # Evaluate function at the left endpoint
    f_left = func(left, w, bsquare, eigenvalue, left, right,i)
    # Bisection loop
    while iter_ < itermax and abs((right - left) / 2) > error:
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



def bifunc_vector(func, left, right, error, w, bsquare, eigenvalue,itermax, root,seq):
    mid = (left + right) / 2
    N = left.size
    f_left = np.zeros(N)
    f_mid = np.zeros(N)
    tmp = np.zeros(N)
    # Evaluate function at the left endpoint
    for i in range(N):
        f_left[i] = func(left[i], w, bsquare, eigenvalue, left[i], right[i],seq[i])


    NB = N
    ci = int((N - 1) / NB)
    res = N % NB
    for k in range(ci + 1):
        begin = k * NB
        end = min(N, begin + NB)
        #print(begin, end)
        itmax = np.linalg.norm(x = itermax[begin:end], ord = np.inf)
        iter_ = 0
        while iter_ < itmax and np.linalg.norm(x = (right[begin:end] - left[begin:end]) / 2, ord = np.inf) > error:
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

def bifunc_vector_muti(func, lleft, rright, error, w, bsquare, eigenvalue,itermax, root):
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