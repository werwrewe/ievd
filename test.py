import random
import numpy as np
import evd
from evaluation import *
from tools import *
from matrix import *
from threading import *
from evd import *
from scipy.linalg import lapack
th = threading.Semaphore(1)

def evdtest(N, seed1,mode=1):
    """
    Test incremental eigenvalue decomposition and standard eigenvalue decomposition.
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

#     print('--- Incremental matrix B ---')
    B = np.block([[A, Alpha], [Alpha.T, w]])
    # Prepare eigenvalues and eigenvectors of A
    EA, QA = np.linalg.eigh(A)
    QA, EA = sorted_eig(QA, np.diag(EA))

    # Incremental eigenvalue decomposition

    # Standard eigenvalue decomposition
    #print('--- EIG ---')
    #start_time = time.time()
    #start_cpu_time = time.process_time()
    #start_perf_time = time.perf_counter()
    #EB_, QB_ = np.linalg.eig(B)
    #tEIG_cputime = time.process_time() - start_cpu_time
    #tEIG_time = time.time() - start_time
    #tPERF_time = time.perf_counter() - start_perf_time
    #print(f'EIG cputime: {tEIG_cputime:.4e}')
    #print(f'EIG time: {tEIG_time:.4e}')
    #print(f'EIG perf time: {tPERF_time:.4e}')
#     print('--- EIGH ---')
    start_time = time.time()
    start_perf_time = time.perf_counter()
    start_cpu_time = time.process_time()
    #print('B=\n',B)
    result = lapack.dsyevd(B)
    EB_,QB_ = result[0],result[1]
    EB_ = np.diag(EB_)
#     print('eb_,qb_',EB_,'\n', QB_)
    tEIG_cputime = time.process_time() - start_cpu_time
    tEIG_time = time.time() - start_time
    tPERF_time = time.perf_counter() - start_perf_time
#     print(f'EIGH cputime: {tEIG_cputime:.4e}')
#     print(f'EIGH time: {tEIG_time:.4e}')
#     print(f'EIGH perf time: {tPERF_time:.4e}')

#     print('--- IEVD ---')
    start_time = time.time()
    start_cpu_time = time.process_time()
    start_perf_time = time.perf_counter()
    QB, EB = evd(QA, EA, Alpha, w, N)
    # 替换原来的打印语句

    # print('eb,qb',EB,'\n', QB)
    # print('eb-eb',np.linalg.norm(EB-EB_))
    # print('qb-qb',np.linalg.norm(np.abs(QB)-np.abs(QB_)))
    tIEVD_cputime = time.process_time() - start_cpu_time
    tIEVD_time = time.time() - start_time
    tPERF_time = time.perf_counter() - start_perf_time
#     print(f'IEVD cputime: {tIEVD_cputime:.4e}')
#     print(f'IEVD time: {tIEVD_time:.4e}')
#     print(f'IEVD perf time: {tPERF_time:.4e}')
    # Error analysis
#     print('--- ERROR ANALYSIS ---')
#     analyze_matrix_differences(EB, EB_, QB, QB_)
    R, O = error_analysis(B, EB, QB,N)
#     R, O = error_analysis(B, np.diag(EB_), QB_,N)
#     print(f'Residual:        {R:.4e}')
#     print(f'Orthogonality:   {O:.4e}')

    return tIEVD_cputime, tIEVD_time, tEIG_cputime, tEIG_time, R, O

def batchtest(N,th,mode=1,batch=1):
    with th:
        tIEVD_cputime_list = []

        tIEVD_time_list = []
        tEIG_cputime_list = []
        tEIG_time_list = []
        R_list = []
        O_list = []
        for i in range(batch):
            seed = 1
            tIEVD_cputime, tIEVD_time, tEIG_cputime, tEIG_time, R, O = evdtest(N, seed, mode=mode)
            tIEVD_cputime_list.append(tIEVD_cputime)
            tIEVD_time_list.append(tIEVD_time)
            tEIG_cputime_list.append(tEIG_cputime)
            tEIG_time_list.append(tEIG_time)
            R_list.append(R)
            O_list.append(O)
        print('\n N = ',N)
#         print('\n IEVD mean cputime：',np.mean(tIEVD_cputime_list))
        print('\n IEVD mean time：',np.mean(tIEVD_time_list))
#         print('\n EIG mean cputime：',np.mean(tEIG_cputime_list))
        print('\n EIG mean time：',np.mean(tEIG_time_list))
        print('\n IEVD mean R：',np.mean(R_list))
        print('\n IEVD mean O：',np.mean(O_list))
        return np.mean(tIEVD_cputime_list),np.mean(tIEVD_time_list),np.mean(tEIG_cputime_list),np.mean(tEIG_time_list),np.mean(R_list),np.mean(O_list)

def batchNtest(th,mode=2):
    N = [1000,2500,5000,7500,10000,15000,20000]
#     N = [5,20,100,500,1000]
    for n in N:
        batchtest(n,th,mode=mode)

for mode in [0,1]:
    print('\n mode=',mode)
    batchNtest(th,mode=mode)