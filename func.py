import numpy as np

def cpfunc(lmbda, w, bb, eigenvalue,left,right,i):
    if i % 2 == 1:
        if lmbda == left:
            return -1
        if lmbda == right:
            return 1
    if i % 2 == 0:
        if lmbda == left:
            return 1
        if lmbda == right:
            return -1

    ev_lambda = eigenvalue - lmbda
    SN = np.sign(ev_lambda)
    signal = np.sum(SN == -1)  # Count how many are -1
    if signal % 2 == 1:
        P1 = -1
    else:
        P1 = 1
    y = P1 * (w - lmbda - np.sum(bb / ev_lambda))
    return y

def cpfunc2(lmbda, w, bb, eigenvalue,left,right,i):
    return np.array([cpfunc(lmbda[j],w, bb, eigenvalue,left[j],right[j],i[j]) for j in range(len(lmbda))])
