from os import name
import random
import numpy as np
import evd
from evaluation import *
from tools import *
from matrix import *
from threading import *
from evd import *
from scipy.linalg import lapack
th = threading.Semaphore(24)
import pandas as pd

'''
    Test part
'''

def evdtest(N, seed1,config, mode=1):
    """
    Test incremental eigenvalue decomposition and standard eigenvalue decomposition.
    Compare the results with the results of dsyevd.
    """
    # Set random seed
    np.random.seed(seed1)
#     print('n=',N)
#     print('--- Generate Hermitian matrix A and incremental vector Alpha, w ---')
    #A = np.random.rand(N, N)
    #A = (A + A.T) / 2  # Ensure A is symmetric
    A = generate_matrix(N, mode, seed1)
#     print('A=',A)
    Alpha = np.random.randn(N, 1)
    w = np.random.randn(1, 1)[0]

#     print('--- matrix B ---')
    if config['target_type'] == 'incremental_rank1':
        B = np.block([[A, Alpha], [Alpha.T, w]])
    elif config['target_type'] == 'standard_rank1':
        w = abs(w)
        B = A + w * Alpha @ Alpha.T
    else:
        raise ValueError("Invalid target_type in config")
    # Prepare eigenvalues and eigenvectors of A
    EA, QA = np.linalg.eigh(A)
    QA, EA = sorted_eig(QA, np.diag(EA))

    # Incremental eigenvalue decomposition
    repeat = 10
    tEIG_cputime_list = []
    tEIG_time_list = []
    tIEVD_cputime_list = []
    tIEVD_time_list = []
    # Standard eigenvalue decomposition
#     print('--- EIGH ---')
    start_time = time.time()
    start_cpu_time = time.process_time()
    result = lapack.dsyevd(B)
    EB_,QB_ = result[0],result[1]
    EB_ = np.diag(EB_)
    tEIG_cputime = time.process_time() - start_cpu_time
    tEIG_time = time.time() - start_time

    start_time = time.time()
    start_cpu_time = time.process_time()
    start_perf_time = time.perf_counter()
    if config['target_type'] == 'incremental_rank1':
        QB, EB = evd(QA, EA, Alpha, w, N, config)
    elif config['target_type'] == 'standard_rank1':
        QB, EB = evd_standard(QA, EA, Alpha, w, config)
    else:
        raise ValueError("Invalid target_type in config")

    tIEVD_cputime = time.process_time() - start_cpu_time
    tIEVD_time = time.time() - start_time

    # Error analysis
#     print('--- ERROR ANALYSIS ---')
#     analyze_matrix_differences(EB, EB_, QB, QB_)
    if config['target_type'] == 'incremental_rank1':
        R, O = error_analysis(B, EB, QB,N+1)
    elif config['target_type'] == 'standard_rank1':
        R, O = error_analysis(B, EB, QB,N)
    else:
        raise ValueError("Invalid target_type in config")


    Eigenvalue_error = np.linalg.norm(EB-EB_) / N
#     R, O = error_analysis(B, np.diag(EB_), QB_,N)
#     print(f'Residual:        {R:.4e}')
#     print(f'Orthogonality:   {O:.4e}')

    return tIEVD_cputime, tIEVD_time, tEIG_cputime, tEIG_time, R, O, Eigenvalue_error

def batchtest(N,th,config,mode=1,batch=1):
    '''
    batch test
    '''
    with th:
        results = []
        tIEVD_cputime_list = []
        tIEVD_time_list = []
        tEIG_cputime_list = []
        tEIG_time_list = []
        R_list = []
        O_list = []
        Eigenvalue_error_list = []
        for i in range(batch):
            seed = 2
            tIEVD_cputime, tIEVD_time, tEIG_cputime, tEIG_time, R, O , Eigenvalue_error= evdtest(N, seed,config, mode=mode)
            tIEVD_cputime_list.append(tIEVD_cputime)
            tIEVD_time_list.append(tIEVD_time)
            tEIG_cputime_list.append(tEIG_cputime)
            tEIG_time_list.append(tEIG_time)
            R_list.append(R)
            O_list.append(O)
            Eigenvalue_error_list.append(Eigenvalue_error)
            result = {
                'N': N,
                'mode': mode,
                'seed': seed,
                'batch': i,
                'tIEVD_cputime': tIEVD_cputime,
                'tIEVD_time': tIEVD_time,
                'tEIG_cputime': tEIG_cputime,
                'tEIG_time': tEIG_time,
                'residual': R,
                'orthogonality': O,
                'eigenvalue_error': Eigenvalue_error,
                'target_type': config['target_type'],
                'stop_criterion': config['stop_criterion']
            }
            results.append(result)
        print('\n N = ',N)
#         print('\n IEVD mean cputime：',np.mean(tIEVD_cputime_list))
        print('\n IEVD mean time：',np.mean(tIEVD_time_list))
#         print('\n EIG mean cputime：',np.mean(tEIG_cputime_list))
        print('\n EIG mean time：',np.mean(tEIG_time_list))
        print('\n IEVD mean R：',np.mean(R_list))
        print('\n IEVD mean O：',np.mean(O_list))
        print('\n EIG mean Eigenvalue_error：',np.mean(Eigenvalue_error_list))

        return results


def batchNtest(th,config,mode=2):
    '''
    batch test for different N
    '''
    # N = [1000]
    all_results = []
    N = [500,5000,10000]
    for n in N:
        results = batchtest(n,th,config,mode=mode)
        all_results.extend(results)
    df = pd.DataFrame(all_results)
    return df
