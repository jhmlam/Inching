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

import numpy
import cupy

from cupy import cublas
import cupy as cp
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
import InchingLiteInteger.Burn.PolynomialFilters.T2

# ====================================
# Thick Restart Lanczos
# ==================================
# NOTE REMARK. While the hotelling is correct, the calcualation is 6 times more in runtime.
#      if the hotelling is done at the Lanczos loop
#      At the end we do not do hotelling. Tradoff too large. though it is implemented,

def S_HeigvalTRLMHD_HeigvecTRLMHD(A, A_diag,
            k=32, 
            maxiter=10000, tol=0,
            User_HalfMemMode = True,
            User_WorkspaceSizeFactor = 2,
            User_Q_HotellingDeflation = None,
            User_HotellingShift = 10.0,
            User_PolynomialParams = None
            ):
    
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
    PART00_Dimensions = True
    if PART00_Dimensions:
        N = A.shape[0]
        assert k < N, "ABORTED. k must be smaller than n"
        assert A.ndim == 2 , "ABORTED. It is a tensor not rank 2!"
        assert A.shape[0] == A.shape[1], "ABORTED. square"


        #assert (k%8 == 0)
        #assert k >= 32, "ABORTED. we did not test on less than 32 modes, as the number ritz vectors is too small."

        # NOTE The workspace
        ncv = min(max(User_WorkspaceSizeFactor * k, k + 32), N - 1)
        
        
        if tol == 0:
            tol = numpy.finfo(A.dtype).eps
        print("There are %s Ritz vectors, tol = %s"%(ncv, tol))


    # ===================================
    # Initialise
    # ===================================
    PART01_InitializeEmpties = True
    if PART01_InitializeEmpties:


        # ===================
        # Initialize
        # ==================
        alpha = cupy.zeros((ncv,), dtype=A.dtype)
        beta = cupy.zeros((ncv,), dtype=A.dtype.char.lower())
        V = cupy.empty((ncv, N), dtype=A.dtype)

        # Set initial vector
        u = cupy.random.random((N,)).astype(A.dtype)
        V[0] = u / cublas.nrm2(u)

        # ===========================================
        # NOTE define krylov protocol to be used. 
        # ============================================
        if User_HalfMemMode:
            KrylovAv = InchingLiteInteger.Burn.Krylov.T3.OOC2_HalfMemS_v_KrylovAv_VOID(A, A_diag)
        else:
            KrylovAv = InchingLiteInteger.Burn.Krylov.T3.OOC2_FullMemS_v_KrylovAv_VOID(A, A_diag)

        if User_PolynomialParams is not None:
            ChebyshevAvC = InchingLiteInteger.Burn.PolynomialFilters.T2.OOC2_A_Adiag_ChebyshevAv(A, A_diag, 
                                                                    User_PolynomialParams = User_PolynomialParams, User_HalfMemMode = User_HalfMemMode)
            ChebyshevAv = ChebyshevAvC.ChebyshevAv

        # NOTE Lanczos. 
        LanczosAvC = OOC2_A_Adiag_LanczosAv(A, A_diag, 
                                            User_PolynomialParams = User_PolynomialParams, 
                                            User_HalfMemMode = User_HalfMemMode)

        # NOTE Hotelling
        if User_Q_HotellingDeflation is None:
            print("WARNING. Hotelling deflation not in use")
            _dohotelling = False
        else:
            _dohotelling = True



        # ================
        # Printing habit
        # =====================
        # NOTE Monitor convergence by res
        if N > 2000000*3:
            printing_ = 1
        else:
            printing_ = 100
        
        if User_PolynomialParams is not None:
            printing_ = 1






    # ======================================
    # Half-leg
    # ======================================
    # NOTE Lanczos iteration
    LanczosAvC.OOC7_A_RitzV_u_alpha_beta_istart_iend_VOID(A, V, u, alpha, beta, 0, ncv,
                                               User_Q_HotellingDeflation =User_Q_HotellingDeflation, 
                                               User_HotellingShift = User_HotellingShift)
    

    # NOTE beta_k == None. This is the a really-tridiag
    S, W = OOC4_alpha_beta_betak_k_TrdEigvalSel_TrdEigvecSel(alpha, beta, None, k, User_PolynomialParams= User_PolynomialParams)

    V[:k] = W.T @ V


    # NOTE Compute residual
    beta_k = beta[-1] * W[-1, :]
    res = cublas.nrm2(beta_k)

    # ====================
    # Main loop
    # =====================

    for coarse_iter in range(maxiter):

        beta[:k] = 0
        alpha[:k] = S

        # =======================
        # Single MGS here
        # =========================
        # NOTE only a single MGS is done. FRO does not help
        u = InchingLiteInteger.Burn.Orthogonalization.T2.T2_vnext_V_MGSvnext(u, V[:k].T, k=None)
        u /= cublas.nrm2(u)
        V[k] = u 

        # =============================
        # Krylov
        # ============================
        # NOTE reuse the last one to get u = A V[k]
        if User_PolynomialParams is None:
            KrylovAv(A,V[k],u)
        else:
            u = ChebyshevAv(A, V[k], User_ReturnRho = False)
            
        # =====================================
        # NOTE Hotelling 
        # ======================================

        if _dohotelling:
            # NOTE This is unexpectedly slower, likely because the matrix has to be interpreted. TODO Write out cuda kernel 
            u = InchingLiteInteger.Burn.Orthogonalization.T3.T3_QHotelling_x_Ax_HotelledAx(
                User_Q_HotellingDeflation,  V[k],u, HotellingShift=User_HotellingShift)




        # =====================
        # Lanczos v_next
        # ======================
        # NOTE This is neessary just because of the code structure
        cublas.dotc(V[k], u, out=alpha[k])
        u -= alpha[k] * V[k]
        u -= V[:k].T @ beta_k
        cublas.nrm2(u, out=beta[k])
        V[k+1] = u / beta[k]

        # NOTE FRO is done inside
        #Lanczos(A, V, u, alpha, beta, k + 1, ncv, 
        #        User_Q_HotellingDeflation, User_HotellingShift = User_HotellingShift)
        LanczosAvC.OOC7_A_RitzV_u_alpha_beta_istart_iend_VOID(A, V, u, alpha, beta, k + 1, ncv, 
                                               User_Q_HotellingDeflation =User_Q_HotellingDeflation, 
                                               User_HotellingShift = User_HotellingShift)

        # ==============================
        # Arrowhead RR
        # ==============================
        S, W = OOC4_alpha_beta_betak_k_TrdEigvalSel_TrdEigvecSel(alpha, beta, beta_k, k, User_PolynomialParams = User_PolynomialParams)

        


        V[:k] = W.T @ V


        # ========================================
        # Residual 
        # ======================================
        # NOTE Compute residual. 
        # NOTE That comparing tol with res a bound result, 
        #      it does not mean we need res==1e-15 to reach || eigval - rayleighquotient||_2 == 1e-15 

        beta_k = beta[-1] * W[-1, :]
        res = cublas.nrm2(beta_k)

        if  res <= tol: 
            break



        if coarse_iter % printing_ == 0:
            print("Coarse_iter %s Estimate at %s. Ritz values follows\n" %(coarse_iter, res))
            print(alpha, "\n")
            print(beta, "\n")







    print('Total number of iterations went through %s in %s seconds'%(coarse_iter, time.time() - st))

    if User_PolynomialParams is None:
        idx = cupy.argsort(S)
    else:
        # NOTE If we used cheb then we must rearrange unfortunately. 
        eigval_converged = []
        Av = cp.zeros(A.shape[0])
        for i in range(S.shape[0]):

            KrylovAv(A,cupy.ravel(V[i]),Av)
            vAv = Av.dot(cupy.ravel(V[i]))
            eigval_converged.append(vAv)

        # NOTE The true eigval recovered
        S = cp.ravel(cp.vstack(eigval_converged))
        idx = cp.argsort(S)


    # ===========================
    # Meory managemnt
    # =============================

    alpha = None
    beta = None
    beta_k = None
    res = None
    u = None
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()



    #return w[idx], x[:, idx]
    #return w[idx], x.T[:, idx]
    return S[idx], V[idx,:].T




# =========================================
# Construction of Tridiag
# ==========================================
# NOTE The minieigenproblem 
#      if beta_k is None we have the regular tridiag.
def OOC4_alpha_beta_betak_k_TrdEigvalSel_TrdEigvecSel(alpha, beta, beta_k, k, 
                                                      User_PolynomialParams = None):
    # Note: This is done on the CPU, because there is an issue in
    # cupy.linalg.eigh with CUDA 9.2, which can return NaNs. It will has little
    # impact on performance, since the matrix size processed here is not large.
    alpha = cupy.asnumpy(alpha)
    beta = cupy.asnumpy(beta)
    t = numpy.diag(alpha)
    t = t + numpy.diag(beta[:-1], k=1)
    t = t + numpy.diag(beta[:-1], k=-1)
    if beta_k is not None:
        beta_k = cupy.asnumpy(beta_k)
        t[k, :k] = beta_k
        t[:k, k] = beta_k

    # Solve it
    S, W = numpy.linalg.eigh(t)
    #print(w)
    #print(t)
    # Pick-up k ritz-values and ritz-vectors
    # NOTE numpy default ascending
    #print('alpha', alpha)
    if User_PolynomialParams is None:
        idx = numpy.argsort(S)[::-1]
        Sk = S[idx[-k:][::-1]]
        Wk = W[:, idx[-k:][::-1]]
    else:
        idx = numpy.argsort(S)
        Sk = S[idx[-k:][::-1]]
        Wk = W[:, idx[-k:][::-1]]

    return cupy.array(Sk), cupy.array(Wk)







# ==================================
# Lanczos
# ====================================
# NOTE normalize the ritz. Using the cupy elementwise kernel
OOC6_u_beta_i_n_v_V_vhat_Vhat = cupy.ElementwiseKernel(
    'T u, raw S beta, int32 j, int32 n', 
    'T v, raw T V',
    'v = u / beta[j]; V[i + (j+1) * n] = v;', 'cupy_eigsh_normalize'
)


# NOTE It uses pointer to update V, u, alpha, beta s.t. we return VOID.
class OOC2_A_Adiag_LanczosAv:

    def __init__(self, A, A_diag, User_PolynomialParams = None, User_HalfMemMode = True):

        self.User_PolynomialParams = User_PolynomialParams 
        self.temp_dtype = A.dtype
        self.N = A.shape[0]  # Assuming n is the length of v. v is a np array

        if User_HalfMemMode:
            self.KrylovAv = InchingLiteInteger.Burn.Krylov.T3.OOC2_HalfMemS_v_KrylovAv_VOID(A, A_diag)
        else:
            self.KrylovAv = InchingLiteInteger.Burn.Krylov.T3.OOC2_FullMemS_v_KrylovAv_VOID(A, A_diag)


        if self.User_PolynomialParams is not None:
            self.ChebyshevAvC = InchingLiteInteger.Burn.PolynomialFilters.T2.OOC2_A_Adiag_ChebyshevAv(A, A_diag, 
                                    User_PolynomialParams = User_PolynomialParams, User_HalfMemMode = User_HalfMemMode)
            self.ChebyshevAv = self.ChebyshevAvC.ChebyshevAv


        self.cublas_handle = device.get_cublas_handle()
        self.cublas_pointer_mode = _cublas.getPointerMode(self.cublas_handle)
        if A.dtype.char == 'f':
            self.dotc = _cublas.sdot
            self.nrm2 = _cublas.snrm2
            self.gemv = _cublas.sgemv
        elif A.dtype.char == 'd':
            self.dotc = _cublas.ddot
            self.nrm2 = _cublas.dnrm2
            self.gemv = _cublas.dgemv
        elif A.dtype.char == 'F':
            self.dotc = _cublas.cdotc
            self.nrm2 = _cublas.scnrm2
            self.gemv = _cublas.cgemv
        elif A.dtype.char == 'D':
            self.dotc = _cublas.zdotc
            self.nrm2 = _cublas.dznrm2
            self.gemv = _cublas.zgemv

        self.one = numpy.array(1.0, dtype=A.dtype)
        self.zero = numpy.array(0.0, dtype=A.dtype)
        self.mone = numpy.array(-1.0, dtype=A.dtype)


    def OOC7_A_RitzV_u_alpha_beta_istart_iend_VOID(self, A, V, u, alpha, beta, i_start, i_end, 
                                                        User_Q_HotellingDeflation = None, 
                                                        User_HotellingShift = 10.0,):
        

        ncv = V.shape[0]
        n = A.shape[0]

        # NOTE Hotelling
        if User_Q_HotellingDeflation is None:
            #print("WARNING. Hotelling deflation not in use")
            _dohotelling = False
        else:
            _dohotelling = True


        v = cupy.empty((n,), dtype=A.dtype)
        uu = cupy.empty((ncv,), dtype=A.dtype)


        v[...] = V[i_start]
        for i in range(i_start, i_end):

            # =========================
            # NOTE Krylov
            # =========================

            if self.User_PolynomialParams is None:
            #u[...] = A @ v
                self.KrylovAv(A,v,u)
            else:
                uu = self.ChebyshevAv(A, v, User_ReturnRho = False)
                cp.copyto(u,uu)


            # =====================================
            # NOTE Hotelling 
            # ======================================

            if _dohotelling:
                # TODO The kernel here may be memory unstable for unknown reason. Dig into this if necessary.
                # NOTE This is unexpectedly slower, likely because the matrix has to be interpreted.
                u = InchingLiteInteger.Burn.Orthogonalization.T3.T3_QHotelling_x_Ax_HotelledAx(
                    User_Q_HotellingDeflation, v , u, HotellingShift=User_HotellingShift)


            # ============================
            # Alpha
            # ============================
            # NOTE Get alpha
            _cublas.setPointerMode(
                self.cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                self.dotc(self.cublas_handle, n, v.data.ptr, 1, u.data.ptr, 1,
                        alpha.data.ptr + i * alpha.itemsize)
            finally:
                _cublas.setPointerMode(self.cublas_handle, self.cublas_pointer_mode)


            # ====================
            # Orthogonalize
            # ====================
            self.gemv(self.cublas_handle, _cublas.CUBLAS_OP_C,
                    n, i + 1,
                    self.one.ctypes.data, V.data.ptr, n,
                    u.data.ptr, 1,
                    self.zero.ctypes.data, uu.data.ptr, 1)
            self.gemv(self.cublas_handle, _cublas.CUBLAS_OP_N,
                    n, i + 1,
                    self.mone.ctypes.data, V.data.ptr, n,
                    uu.data.ptr, 1,
                    self.one.ctypes.data, u.data.ptr, 1)

            # =================
            # Beta
            # ====================
            _cublas.setPointerMode(
                self.cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                self.nrm2(self.cublas_handle, n, u.data.ptr, 1,
                        beta.data.ptr + i * beta.itemsize)
            finally:
                _cublas.setPointerMode(self.cublas_handle, self.cublas_pointer_mode)
            
            # =================
            # FRO
            # ====================
            # Orthogonalize
            self.gemv(self.cublas_handle, _cublas.CUBLAS_OP_C,
                 n, i + 1,
                 self.one.ctypes.data, V.data.ptr, n,
                 u.data.ptr, 1,
                 self.zero.ctypes.data, uu.data.ptr, 1)
            #print(uu)
            self.gemv(self.cublas_handle, _cublas.CUBLAS_OP_N,
                 n, i + 1,
                 self.mone.ctypes.data, V.data.ptr, n,
                 uu.data.ptr, 1,
                 self.one.ctypes.data, u.data.ptr, 1)

            # Break here as the normalization below touches V[i+1]
            if i >= i_end - 1:
                break

            # NOTE THis is the 
            OOC6_u_beta_i_n_v_V_vhat_Vhat(u, beta, i, n, v, V)

        return














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
