from collections import defaultdict
import tqdm
import sys
import itertools



import sys
import tqdm
import gc



import numpy as np
import numba as nb


import torch
from torch import jit


sys.path.append('..')
sys.path.append('../Script/Burn/')





import InchingLite.util



import numpy as np
import numba as nb

@nb.njit((nb.float64[:, :], nb.float64[:, :]), parallel=True)
def X_Y_CosineSimilarity(X, Y):



    # NOTE This is a numba accelerated calculation of cosine similarity. a,b need not carry the same number of vectors


    assert X.shape[1] == Y.shape[1], "ABORTED. Incompatible shape encounted in X_Y_CosineSimilarity. assume same shape[1]"
    common_number_element = X.shape[1]

    norm_X = np.empty((X.shape[0],), dtype=np.float64)
    norm_Y = np.empty((Y.shape[0],), dtype=np.float64)
    result = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float64)

    # ================
    # Calculate 2-norm
    #=================
    for i in nb.prange(X.shape[0]):
        sq_norm = 0.0
        for j in range(common_number_element):
            sq_norm += X[i][j] ** 2
        norm_X[i] = sq_norm ** 0.5
    
    for i in nb.prange(Y.shape[0]):
        sq_norm = 0.0
        for j in range(common_number_element):
            sq_norm += Y[i][j] ** 2
        norm_Y[i] = sq_norm ** 0.5
        
    # ===================
    # Calculate cosine
    # ===================
    for i in nb.prange(X.shape[0]):
        for j in range(Y.shape[0]):
            #if i < j:
            #    continue # NOTE The vectors provided may not be the same s.t. we need unsymmetric calculatino for all vecors present
            dot_ij = 0.0
            for k in range(common_number_element):
                dot_ij += X[i][k] * Y[j][k]
            dot_ij /= (norm_Y[j] * norm_X[i])
            result[i,j] = dot_ij
 
    return result 