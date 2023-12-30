
import numpy
import cupy
import cupy as cp


from cupy import cublas
from cupy import cusparse
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse.linalg import _interface
import time

cupy.random.seed(seed = 0)

import time
import sys
sys.path.append('../InchingLiteInteger/Burn/')

import InchingLiteInteger.Burn.Orthogonalization.T3
import InchingLiteInteger.Burn.Orthogonalization.T2
import InchingLiteInteger.Burn.Krylov.T3
import tqdm 


# NOTE we will initiate a vector inside because this is only run once. 
def A_Adiag_EstimateSpectrumBound(A, A_diag, User_HalfMemMode = True ):

    #mempool = cp.get_default_memory_pool()
    #pinned_mempool = cp.get_default_pinned_memory_pool()


    N = A.shape[0]

    if User_HalfMemMode:
        KrylovAv = InchingLiteInteger.Burn.Krylov.T3.OOC2_HalfMemS_v_KrylovAv_VOID(A, A_diag)
    else:
        KrylovAv = InchingLiteInteger.Burn.Krylov.T3.OOC2_FullMemS_v_KrylovAv_VOID(A, A_diag)

    jmin = min(100, int(N / 2 / 2))
    #n_tridiag = max(jmin, int(N / 2))
    n_tridiag = 300 # NOTE 300 enough

    alpha_list = cp.zeros(n_tridiag + 1, dtype=A.dtype)
    beta_list = cp.zeros(n_tridiag, dtype=A.dtype)


    # NOTE f is the placeholder for Av product.
    f = cp.zeros(N, dtype=A.dtype)
    v0 = cp.zeros(N, dtype=A.dtype)

    v = cp.random.random(N) - 0.5
    v /= cp.linalg.norm(v, ord = 2)

    
    KrylovAv(A,v,f)
    #f = A @ v   # NOTE KrylovAv(A,v_,f_)
    alpha = cp.dot(v, f)

    f -= alpha * v
    alpha_list[0] = alpha

    for i in tqdm.tqdm(range(n_tridiag)): # NOTE Classic question here. what if you change n_tridiag? This is a no-restart lanczos
        beta = cp.linalg.norm(f, ord = 2)
        
        #v0 = v
        #v = f / beta
        cp.copyto(v0, v)
        cp.copyto(v, f / beta)

        #f = A @ v
        KrylovAv(A,v,f)
        #f = A @ v
        f -= beta * v0

        alpha = cp.dot(f, v)
        f -= alpha * v

        alpha_list[i + 1] += alpha
        beta_list[i] += beta




    #T_  = numpy.diag(cupy.asnumpy(alpha_list_),0) + numpy.diag(cupy.asnumpy(beta_list_),-1) + numpy.diag(cupy.asnumpy(beta_list_),1)
    T = numpy.diag(alpha_list, 0) + numpy.diag(beta_list, -1) + numpy.diag(beta_list, 1)
    w, q = numpy.linalg.eigh(T)
    print("Done Bound")
    f = None
    v = None
    v0 = None
    del f,v,v0, KrylovAv

    #mempool.free_all_blocks()
    #pinned_mempool.free_all_blocks()   


    return numpy.min(w), numpy.max(w)



