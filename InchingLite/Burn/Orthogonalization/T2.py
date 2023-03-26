
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg

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

sys.path.append('../InchingLite/Burn/')






# ==========================
# Orthogonalization
# ===========================


def T2_vnext_V_MGSvnext(u, V, k = None):

        # NOTE Assume V: [dim ,n_vector]

        # NOTE When output the u is not normalized as sometimes this magnitude can be useful
        if k == None:
            k = V.shape[1]
        else:
            k = int(k)


        uu = cupy.empty((k,), dtype=u.dtype)

        # NOTE Full reorthonorm. 
        # # https://github.com/cupy/cupy/blob/46a4833e355d9fcc885ef2113ea13bf7b4ad5c72/cupy/cublas.py#L428
        # gemv(transa, alpha, a, 
        #       x, beta, y)
        # Computes y = alpha * op(a) @ x + beta * y
        # uu = V[:k].T @ u
        cublas.gemv(_cublas.CUBLAS_OP_C, 1, V[:,:k], #,V[:k].T, 
                        u, 
                        0, # NOTE beta = 0, so uu is only a placeholder for the result.
                        uu)
        # u -= V[:k].T @ uu 
        cublas.gemv(_cublas.CUBLAS_OP_N, -1, V[:,:k], #V[:k].T, 
                        uu, 
                        1, 
                        u)
        uu = None
        del uu
        return u



def OOC2_qnext_Q_ICGSqnext(u,Q):


    # NOTE THis can be modified to the ICMGS when generalized eigproblem is needed. 
    #      Only the name hint is kept here.
    r_pre=cp.sqrt(cp.abs(u.T.dot(u)))
    # NOTE Full reorth 3 times...
    for i_FRO in range(3):
        u = u - Q.dot(Q.T.dot(u))
        r1 = cp.sqrt(cp.abs(u.T.dot(u)))

        if r1>r_pre/2:
            break
        r_pre = r1

    if r1 <= r_pre/2:
        print('WARNING. still a loss of orthogonality? Something wrong nan?')


    return u/r1




# ====================
# OBSOLETE
# =======================



# NOTE For JDM we need not put a stop at k, but for TRLM IRLM we need to 
def OOC2_qnext_Q_MGSqnext(u,Q):
    # NOTE THis can be easily modifed to QR algo.
    for i in range(Q.shape[1]):
        s = u.T.dot(Q[:,i:i+1])
        u = u - s*Q[:,i:i+1]

    return u

