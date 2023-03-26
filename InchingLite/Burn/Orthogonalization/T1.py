
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



def R_OrthogonalizeAllBefore(V, ix, beforeindex):
    # Modified Gram Schmidt
    # NOTE V:       A possibly rectangular matrix
    #      ix:      Index of the vector to be orthogonalised
    #      indices: Set of vectors (often culm from previous iterations) to be orthogoalised against
    s = cupy.matmul( V[:beforeindex, :],V[ix,:].T )
    V[ix,:] -= cupy.matmul(s.T, V[:beforeindex,:])
    V[ix,:] /= cupy.linalg.norm(V[ix,:], ord=2) 
    return V
