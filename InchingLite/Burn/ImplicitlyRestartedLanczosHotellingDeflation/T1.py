# =======================================================================================
#   Copyright 2020-present Jordy Homing Lam, JHML, University of Southern California
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    *  Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#    *  Redistributions in binary form must reproduce the above copyright notice, 
#       this list of conditions and the following disclaimer in the documentation and/or 
#       other materials provided with the distribution.
#    *  Cite our work at Lam, J.H., Nakano, A., Katritch, V. REPLACE_WITH_INCHING_TITLE
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# =========================================================================================

# NOTE in use

import sys

sys.path.append('..')

import numpy
import cupy

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

sys.path.append('..')
sys.path.append('../InchingLite/Burn/')
import InchingLite.Burn.Orthogonalization.T3
import InchingLite.Burn.Orthogonalization.T2
import InchingLite.Burn.Orthogonalization.T1
import InchingLite.Burn.Krylov.T3



# =================
# Misc
# =================

def CheckConvergenceByBound(b, User_tol, k, iter_i, maxiter):
    # NOTE Fast check of convergence by bound

    # NOTE Instead of doing ||vAv/vv - \lambda||, We will check the convergence using 
    #      the guarenteed upper bound of error. See Demmel ANLA Ch7 theorem 7.2 for discussion p.365
    #      The error is bounded by the k-th element (indexed as k-1 in 0-indexed systems) of b 
    #      of the current tridiagonal

    # NOTE b:               The secondary diagonal of the tridiagonal
    #      User_tol:        User tolerance for the intermediate steps
    
    return (iter_i > maxiter) or (cupy.abs(b[k]) < User_tol)



# ===========================
# Lanczos with control flow for IRLM
# ==============================
# NOTE THis is more than just lanczos loop but the control flow is refined.
def OOC4_A_Trd0_Trd1_Qprevious_LanczosQ_LanczosTrd0_LanczosTrd1(A, a, b, V, k, p, 
                        User_tol, n, KrylovAv, User_Q_HotellingDeflation = None, User_HotellingShift = 10.0):

    # NOTE Lanczos Tridiagonalisation Single step.
    # NOTE  I/O explained
    #       A           : Matrix of interest or other relevant input n*n in size 
    #       (a,b)       : (Primary,Secondary) diagonal of the tridiagonal to be outputed
    #       V           : Ritz vectors
    #       k           : Number of Desired accurate extremal eigpair
    #       p           : Number of Undesired "buffering" eigpair . According to Paige's analysis there is a bound TODO
    #       User_tol    : Check on Convergence in the desired frontal region

    # NOTE This is equivalent to http://www.netlib.org/utk/people/JackDongarra/etemplates/node104.html
    #      or step (2) in IRLM   http://www.netlib.org/utk/people/JackDongarra/etemplates/node118.html
    #      It can be run w/o IRLM, but will be ineffective in handling the muddle water of non-extremal pairs
    #      It can also be followed by explicit restart of course.


    if User_Q_HotellingDeflation is None:
        #print("WARNING. Hotelling deflation not in use")
        _dohotelling = False
    else:
        _dohotelling = True



    u = cupy.zeros((n,),  dtype=cupy.float64)#V.dtype)
    for j in range(p):



        # NOTE because we will always initialise with k = 1 and we will assert k >1 for 
        #      the use of this code we can do it as 1+0-1==0
        if (k+j-1==0 or cupy.abs(b[k+j-2]) < User_tol):

            # ==================================
            # Random guess
            # ===================================


            V[k+j-1,:] = cupy.random.random((n,))

            # =========================
            # Full Reorthogonalise 1
            # =========================
            # NOTE Full Reorthog to assure stability and orthogoanlity
            #      http://www.netlib.org/utk/people/JackDongarra/etemplates/node108.html#select-ort
            #      Paige's work https://www.cs.mcgill.ca/~chris/pubClassic/PaigeThesis.pdf 
            if (k+j-1 > 1):

                V = InchingLite.Burn.Orthogonalization.T1.R_OrthogonalizeAllBefore(V, k+j-1, k+j-2)
                V = InchingLite.Burn.Orthogonalization.T1.R_OrthogonalizeAllBefore(V, k+j-1, k+j-2)

            else:
                # NOTE This is added because orthogonalise is not called when the if clause not fulfilled
                #      during initialisation

                V[k+j-1,:] /= cupy.sqrt(cupy.sum(V[k+j-1] *V[k+j-1] ))

        # ===========================
        # Single Matrix Vec
        # ===========================
        # NOTE This is the major difference between implementation in previous version of Inching
        #      Rather than doing MM multiplication in A*V[:,:k+j-1], we do it one by one
        #                                             A*V[:, k+j-1]
        #      s.t. we can fine monitor the convergence process one step every time!
        #      NOTE But this can still be done by batching A when mm=emory demand is too great.
        #      TODO insert the memory batching here.

        KrylovAv(A,V[k+j-1],u)



        # ==========================
        # Hotelling
        # ==========================
        # NOTE This can be done here./

        if _dohotelling:
            # TODO The kernel here may be memory unstable for unknown reason. Dig into this if necessary.
            # NOTE This is unexpectedly slower, likely because the matrix has to be interpreted.
            InchingLite.Burn.Orthogonalization.T3.T3_QHotelling_x_Ax_HotelledAx(User_Q_HotellingDeflation, V[k+j-1], u, HotellingShift=User_HotellingShift)

        # =====================================
        # Lanczos Ritz pair calculations
        # =====================================
        # NOTE http://www.netlib.org/utk/people/JackDongarra/etemplates/node118.html
        # TODO h is r in http://www.netlib.org/utk/people/JackDongarra/etemplates/node104.html

        if (k+j-1 > 0):
            # NOTE we are doing two steps together to take advantage of the parallelism
            #      This is equiv to retrieving two columns (indexed by k+j-2 and k+j-1) from V 
            #      similar to step (6)-(8) http://www.netlib.org/utk/people/JackDongarra/etemplates/node104.html 
            #      but because of the intitialisation else clause below we already did "half-leg" of these steps
            h = cupy.matmul(V[k+j-2:k+j,:], u[:,None])  #torch.mv(V[:,k+j-2:k+j].T, u)
            a[k+j-1] = h[1]                             # NOTE v_j-2 * r 
            b[k+j-1] = h[0]                             # NOTE v_j-1 * r 
            V[k+j,:] = u[None,:] - (cupy.matmul(h.T, V[k+j-2:k+j,:]))
        else:
            # NOTE intialisation only
            h = cupy.sum(V[0]*(u))
            a[0] = h
            V[1] = u - V[0]*h                   # NOTE the 'half-leg'

        beta = cupy.sqrt(cupy.sum(V[k+j]*V[k+j] )) #torch.linalg.vector_norm(V[:,k+j], ord=2, dtype=dtype_temp, out=None) 



        # ===========================
        # Check convergence 
        # ===========================
        if (beta > User_tol):
            V[k+j,:] /= beta
            b[k+j-1] = beta
        else:
            # NOTE Since we will purge converged value
            #      When converged we will reset the b s.t. abs(b[k+j-2]) < User_tol
            #      to trigger the ============Random guess===============
            #      s.t. this place can be reused
            b[k+j-1] = 0
            continue

        

        # =========================
        # Full Reorthogonalise 2
        # =========================
        V = InchingLite.Burn.Orthogonalization.T1.R_OrthogonalizeAllBefore(V, k+j, k+j-1)
        V = InchingLite.Burn.Orthogonalization.T1.R_OrthogonalizeAllBefore(V, k+j, k+j-1)

    return V, a, b




# =========================
# Implicit Shift
# =========================
# NOTE Make a tridiag
def OOC2_Trd0_Trd1_Tridiagonal(alpha, beta):
    #print(cupy.diag(beta[:], k=1).shape)
    T = cupy.diag(alpha)
    T = T + cupy.diag(beta[:], k=1)
    T = T + cupy.diag(beta[:], k=-1)
    return T

def OOC3_RitzQ_Trd0_Trd1_ImplicitShiftQ_Trd0_Trd1(V, a, b, k, p, User_ReturnIndiceLambda = None,
                                    ):


    # ============================================
    # determine shifts
    # ============================================
    # NOTE Implicit Shift Step for Lanczos
    T = OOC2_Trd0_Trd1_Tridiagonal(a, b[:k+p-1])
    eigvals = numpy.linalg.eigvalsh(cupy.asnumpy(T))

    # ===============================================
    # Filtering Function
    # ===============================================
    # NOTE Currently this is hard coded, but it can be made a flag for a lamda function
    if User_ReturnIndiceLambda is None:
        # NOTE np argsort default to ascend
        #indices = np.argsort(eigvals)[:p]          # NOTE This will select for the largest purge the small ones
        #indices = np.argsort(eigvals)[::-1][:p]     # NOTE This will select for the smallest purge the large ones
        indices = numpy.argsort(eigvals)[::-1][:p]
    else:
        indices = User_ReturnIndiceLambda(eigvals)



    # ==============================================
    # Purging and Update of Q and T 
    # ===============================================

    Q = cupy.eye(k+p) 
    for i in indices:

        T = OOC2_Trd0_Trd1_Tridiagonal(a, b[:k+p-1])

        T = cupy.array(T)
        lastb = b[k+p-1:k+p]

        Qj = cupy.linalg.qr(T -  eigvals[i]*cupy.eye(k+p), mode='complete')[0]

        T = cupy.matmul(T,Qj)
        T = cupy.matmul(Qj.T,T)

        # NOTE To avoid numerical residual due to float we will extract the tridiag and update a b
        a = cupy.diagonal(T, offset=0, axis1=0, axis2=1)
        b = cupy.concatenate(( cupy.diagonal(T, offset=1, axis1=0, axis2=1), lastb))

        Q = cupy.matmul(Q, Qj)


    # Transpose Q is easier than V
    Q = Q.T

    # =====================
    # Complete the shift
    # ========================
    v = cupy.matmul( Q[k,:],V[:k+p,:]) 
    V[:k,:] = cupy.matmul(Q[:k,:],V[:k+p,:])
    V[k,:] = b[k-1] * v + (b[k+p-1]*Q[k-1, k+p-1])*V[k+p,:]
    beta = cupy.sqrt(cupy.sum(V[k]*V[k])) #beta = cublas.nrm2(V[k,:]) # NOTE This produce a different incorrect number...??? TODO
    V[k,:] /= beta #
    b[k-1] = beta #
    
    V = InchingLite.Burn.Orthogonalization.T1.R_OrthogonalizeAllBefore(V, k ,k)
    V = InchingLite.Burn.Orthogonalization.T1.R_OrthogonalizeAllBefore(V, k ,k)


    
    return V, a, b





# =============
# Main
# ===================

def S_HeigvalIRLMHD_HeigvecIRLMHD(A,
                    k = 32, maxiter=15000, 
                    tol=1e-8,  # NOTE Usually 1e-8 for the paige bound is okay
                    User_HalfMemMode= True,
                    User_Q_HotellingDeflation = None, # NOTE This is not implemented as TRLM does not survived in speed
                    User_HotellingShift = 10.0):

    # NOTE Wrapper for the IRLM loop 
    # NOTE  I/O explained
    #       A           : Matrix of interest or other relevant input
    #       k           : Number of Desired accurate extremal eigpair
    #       maxiter     : Maximum number of iterations

    st = time.time()
    # ==============================
    # Memory management
    # ===============================
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()


    # =================
    # Bounding computation time
    # ===========================
    # NOTE THis is fixed so that we can calclaute block size easily.
    User_tol = tol
    PART00_Dimensions = True
    if PART00_Dimensions:
        n = A.shape[0]
        assert k < n, "ABORTED. k must be smaller than n"
        assert A.ndim == 2 , "ABORTED. It is a tensor not rank 2!"
        assert A.shape[0] == A.shape[1], "ABORTED. square"
        #assert (k%8 == 0), "ABORTED. Let's do it with a multiple of 8"
        assert k >= 32, "ABORTED. we did not test on less than 32 modes, as the number ritz vectors is too small."
        assert k > 1, "ABORTED. This implememntation only works with k > 1"

        # NOTE The workspace
        p = min(max(2 * k, k + 32), n - 1)

        # NOTE Raise one more for k. Unfortunately, we need to keep one more    
        k += 2

        # NOTE The total basis
        m = k+p
        assert n >= m, "ABORTED. You sure you want more basis than number of columns of A?"
        if maxiter is None:
            maxiter = 10 * n
        
        if tol == 0:
            tol = numpy.finfo(A.dtype).eps
        print("There are %s Ritz vectors, tol = %s"%(m, tol))




    # ===========================================
    # NOTE define protocol to be used. 
    # ============================================
    # NOTE Krylov
    if User_HalfMemMode:
        KrylovAv = InchingLite.Burn.Krylov.T3.OOC2_HalfMemS_v_KrylovAv_VOID(A)
    else:
        KrylovAv = InchingLite.Burn.Krylov.T3.OOC2_FullMemS_v_KrylovAv_VOID(A)

    """
    # NOTE This is not used as I need more control flow for IRLM...
    # NOTE Lanczos. 
    if User_HalfMemMode:
        Lanczos = OOC7_HalfMemS_RitzV_u_alpha_beta_kplus1_numRitz_VOID(A)
    else:
        Lanczos = OOC7_FullMemS_RitzV_u_alpha_beta_kplus1_numRitz_VOID(A)
    """

    # NOTE Hotelling
    if User_Q_HotellingDeflation is None:
        print("WARNING. Hotelling deflation not in use")
        _dohotelling = False
    else:
        _dohotelling = True



    # =========================
    # Initialise 
    # =========================
    Converged_ = False
    Converged_printer_ = True

    PART01_InitializeEmpties = True
    if PART01_InitializeEmpties:
        V = cupy.zeros(( m+1, n)) # NOTE the buffer +1
        alpha = cupy.zeros(m)
        beta = cupy.zeros(m)


    # NOTE beta_k == None. This is the a really-tridiag
    iter_i = 0
    # NOTE Ritz vector and tridiagonal
    V,alpha,beta = OOC4_A_Trd0_Trd1_Qprevious_LanczosQ_LanczosTrd0_LanczosTrd1(
                                A, alpha, beta, V, 1, k, User_tol, n, KrylovAv,
                                User_Q_HotellingDeflation = User_Q_HotellingDeflation, 
                                User_HotellingShift = User_HotellingShift) 

    for iter_i in range(maxiter):

        if Converged_:
            if Converged_printer_:
                #print("Converged")
                Converged_printer_ = False
            continue

        V, alpha, beta = OOC4_A_Trd0_Trd1_Qprevious_LanczosQ_LanczosTrd0_LanczosTrd1(
                                A, alpha, beta, V, k+1, p, User_tol, n, KrylovAv,
                                User_Q_HotellingDeflation = User_Q_HotellingDeflation, 
                                User_HotellingShift = User_HotellingShift) 
        
        # NOTE The k and p are fixed
        V, alpha ,beta = OOC3_RitzQ_Trd0_Trd1_ImplicitShiftQ_Trd0_Trd1(V, alpha, beta, k, p,  
        User_ReturnIndiceLambda = None)

        
        # NOTE b[k-1] b[k] are the "cliff", where the index boundary where undesired eigenpair start appearing
        #      We will use them as a trigger for random guess.
        if cupy.abs(beta[k-1]) < User_tol:
            beta[k-1] = 0.0
        if cupy.abs(beta[k]) < User_tol: 
            beta[k] = 0.0

        # NOTE Monitor convergence by paige
        if n > 2000000*3:
            printing_ = 1
        else:
            printing_ = 10

        if iter_i % printing_ == 0:
            print("User_tol %s < Current Estimate of Error" %(User_tol), beta[k-2:k-1]) 

        Converged_ = CheckConvergenceByBound(beta, User_tol, k-2, iter_i, maxiter)



    # ====================================
    # NOTE Final projectino
    # ====================================
    T = OOC2_Trd0_Trd1_Tridiagonal(alpha[:k], beta[:k-1])
    Teigval, Teigvec = numpy.linalg.eigh(cupy.asnumpy(T), UPLO='L')

    Teigval = cupy.array(Teigval)
    Teigvec = cupy.array(Teigvec)

    V[:k-1,:] = cupy.matmul(Teigvec[:k-1,:k-1].T, V[:k-1,:])




    # ===========================
    # Meory managemnt
    # =============================

    xx = V[:k-2].T
    V = None
    alpha = None
    beta = None

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()



    return Teigval[:k-2], xx




# =======================================================================================
#   Copyright 2020-present Jordy Homing Lam, JHML, University of Southern California
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    *  Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#    *  Redistributions in binary form must reproduce the above copyright notice, 
#       this list of conditions and the following disclaimer in the documentation and/or 
#       other materials provided with the distribution.
#    *  Cite our work at Lam, J.H., Nakano, A., Katritch, V. REPLACE_WITH_INCHING_TITLE
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# =========================================================================================