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

sys.path.append('../InchingLiteInteger/Burn/')

import InchingLiteInteger.Burn.JacobiDavidsonHotellingDeflation.IterativeSolvers
import InchingLiteInteger.Burn.Orthogonalization.T3
import InchingLiteInteger.Burn.Orthogonalization.T2
import InchingLiteInteger.Burn.Krylov.T3
import InchingLiteInteger.Burn.PolynomialFilters.T2
#import InchingLiteInteger.Fuel.CupysparseCsrInt32

# ================================
# Krylov Iteration
# ==================================

# NOTE What this does is that it package the system matrix for solver(system_matrix to work on
#      https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.LinearOperator.html?highlight=LinearOperator
# 
class OOC1_FullMemA_KrylovLinearOperator(cupyx.scipy.sparse.linalg.LinearOperator):

    def __init__(self, A, shift = 0, QHotelling = cp.zeros(3), HotellingShift = 10.0, _dohotelling = False):

        self._dohotelling = _dohotelling

        self.A = A
        spshape = A.shape
        ddtype  = A.dtype
        self.shift  = shift
        self.QHotelling = QHotelling
        self.HotellingShift = HotellingShift 
        self.cublas_handle = device.get_cublas_handle()
        self.cublas_pointer_mode = _cublas.getPointerMode(self.cublas_handle)

        if A.dtype.char == 'f':
            dotc = _cublas.sdot
            nrm2 = _cublas.snrm2
            gemv = _cublas.sgemv
        elif A.dtype.char == 'd':
            dotc = _cublas.ddot
            nrm2 = _cublas.dnrm2
            gemv = _cublas.dgemv
        elif A.dtype.char == 'F':
            dotc = _cublas.cdotc
            nrm2 = _cublas.scnrm2
            gemv = _cublas.cgemv
        elif A.dtype.char == 'D':
            dotc = _cublas.zdotc
            nrm2 = _cublas.dznrm2
            gemv = _cublas.zgemv
        else:
            raise TypeError('invalid dtype ({})'.format(A.dtype))

        self.cusparse_handle = None

        self.cusparse_handle = device.get_cusparse_handle()
        self.spmv_op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        self.spmv_alpha = numpy.array(1.0, A.dtype)
        self.spmv_beta = numpy.array(-1.0, A.dtype)
        self.spmv_cuda_dtype = _dtype.to_cuda_dtype(A.dtype)
        self.spmv_alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT

        n = A.shape[0]
        #v = cupy.empty((n,), dtype=A.dtype)
        #u = cupy.empty((n,), dtype=A.dtype)
        self.spmv_desc_A = cusparse.SpMatDescriptor.create(A)
        

        super(OOC1_FullMemA_KrylovLinearOperator,self).__init__(shape = spshape, dtype = ddtype)
    

    # NOTE This is 
    def _matvec(self,x):


        # Matrix-vector multiplication
        if self.cusparse_handle is None:
            res = self.A @ x 
            res -= self.shift * x
            
        else:
            spmv_desc_v = cusparse.DnVecDescriptor.create(x)
            res = self.shift * x
            spmv_desc_u = cusparse.DnVecDescriptor.create(res)
            buff_size = _cusparse.spMV_bufferSize(
                self.cusparse_handle, self.spmv_op_a, self.spmv_alpha.ctypes.data,
                self.spmv_desc_A.desc, spmv_desc_v.desc, self.spmv_beta.ctypes.data,
                spmv_desc_u.desc, self.spmv_cuda_dtype, self.spmv_alg)
            spmv_buff = cupy.empty(buff_size, cupy.int8)


            _cusparse.spMV(
                self.cusparse_handle, self.spmv_op_a, self.spmv_alpha.ctypes.data,
                self.spmv_desc_A.desc, 
                spmv_desc_v.desc,
                self.spmv_beta.ctypes.data, spmv_desc_u.desc,
                self.spmv_cuda_dtype, self.spmv_alg, 
                spmv_buff.data.ptr)
        _cublas.setPointerMode(self.cublas_handle, self.cublas_pointer_mode)
        if self._dohotelling:
            res += self.HotellingShift*((self.QHotelling@x)[None,:]@self.QHotelling).flatten()

        return res

class OOC1_HalfMemA_KrylovLinearOperator(cupyx.scipy.sparse.linalg.LinearOperator):

    def __init__(self, A, A_diag, shift = 0, QHotelling = cp.zeros(3),HotellingShift = 10.0, _dohotelling = False):


        self._dohotelling = _dohotelling
        self.A = A

        # NOTE We should do once when we initiate
        self.A_diag = A_diag
        spshape = A.shape
        ddtype  = A.dtype
        self.shift  = shift
        self.QHotelling = QHotelling
        self.HotellingShift = HotellingShift
        self.cublas_handle = device.get_cublas_handle()
        self.cublas_pointer_mode = _cublas.getPointerMode(self.cublas_handle)
        if A.dtype.char == 'f':
            dotc = _cublas.sdot
            nrm2 = _cublas.snrm2
            gemv = _cublas.sgemv
        elif A.dtype.char == 'd':
            dotc = _cublas.ddot
            nrm2 = _cublas.dnrm2
            gemv = _cublas.dgemv
        elif A.dtype.char == 'F':
            dotc = _cublas.cdotc
            nrm2 = _cublas.scnrm2
            gemv = _cublas.cgemv
        elif A.dtype.char == 'D':
            dotc = _cublas.zdotc
            nrm2 = _cublas.dznrm2
            gemv = _cublas.zgemv
        else:
            raise TypeError('invalid dtype ({})'.format(A.dtype))

        self.cusparse_handle = None

        self.cusparse_handle = device.get_cusparse_handle()
        self.spmv_op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        self.spmv_alpha = numpy.array(1.0, A.dtype)
        self.spmv_beta = numpy.array(-1.0, A.dtype)
        self.spmv_cuda_dtype = _dtype.to_cuda_dtype(A.dtype)
        self.spmv_alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT

        self.spmv_op_atriu = _cusparse.CUSPARSE_OPERATION_TRANSPOSE 
        self.spmv_betatriu = numpy.array(1.0, A.dtype)
        self.spmv_alphadiag = numpy.array(-1.0, A.dtype)
        
        n = A.shape[0]
        #v = cupy.empty((n,), dtype=A.dtype)
        #u = cupy.empty((n,), dtype=A.dtype)
        self.spmv_desc_A = cusparse.SpMatDescriptor.create(A)

        if self._dohotelling:

            self.v_hotelling1 = cupy.empty((QHotelling.shape[0],), dtype=QHotelling.dtype)
            if QHotelling.dtype.char == 'f':
                dotc = _cublas.sdot
                nrm2 = _cublas.snrm2
                gemv = _cublas.sgemv
            elif QHotelling.dtype.char == 'd':
                dotc = _cublas.ddot
                nrm2 = _cublas.dnrm2
                gemv = _cublas.dgemv
            elif QHotelling.dtype.char == 'F':
                dotc = _cublas.cdotc
                nrm2 = _cublas.scnrm2
                gemv = _cublas.cgemv
            elif QHotelling.dtype.char == 'D':
                dotc = _cublas.zdotc
                nrm2 = _cublas.dznrm2
                gemv = _cublas.zgemv
            else:
                raise TypeError('invalid dtype ({})'.format(QHotelling.dtype))

            self.cusparse_handle_Hotelling = None
            if _csr.isspmatrix_csr(QHotelling) and cusparse.check_availability('spmv'):
                self.cusparse_handle_Hotelling = device.get_cusparse_handle()
                self.spmv_op_a_Hotelling1 = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
                self.spmv_op_a_Hotelling2 = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
                self.spmv_alpha_Hotelling1 = numpy.array(self.HotellingShift, QHotelling.dtype)
                self.spmv_alpha_Hotelling2 = numpy.array(1.0, QHotelling.dtype)
                
                self.spmv_beta_Hotelling1 = numpy.array(0.0, QHotelling.dtype)
                self.spmv_beta_Hotelling2 = numpy.array(1.0, QHotelling.dtype)
                self.spmv_cuda_dtype_Hotelling = _dtype.to_cuda_dtype(QHotelling.dtype)
                self.spmv_alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT




            self.spmv_desc_QHotelling = cusparse.SpMatDescriptor.create(QHotelling)
            self.spmv_desc_vhotelling1 = cusparse.DnVecDescriptor.create(self.v_hotelling1)

        super(OOC1_HalfMemA_KrylovLinearOperator,self).__init__(shape = spshape, dtype = ddtype)
    

    # NOTE This is 
    def _matvec(self,x):


        # Matrix-vector multiplication
        if self.cusparse_handle is None:
            res = self.A @ x 
            res -= self.shift * x
        else:
            spmv_desc_v = cusparse.DnVecDescriptor.create(x)
            res = self.shift * x
            spmv_desc_u = cusparse.DnVecDescriptor.create(res)
            buff_size = _cusparse.spMV_bufferSize(
                self.cusparse_handle, self.spmv_op_a, self.spmv_alpha.ctypes.data,
                self.spmv_desc_A.desc, spmv_desc_v.desc, self.spmv_beta.ctypes.data,
                spmv_desc_u.desc, self.spmv_cuda_dtype, self.spmv_alg)
            spmv_buff = cupy.empty(buff_size, cupy.int8)
            spmv_bufftemptriu = cupy.empty(buff_size, cupy.int8)

            _cusparse.spMV(
                self.cusparse_handle, self.spmv_op_a, self.spmv_alpha.ctypes.data,
                self.spmv_desc_A.desc, 
                spmv_desc_v.desc,
                self.spmv_beta.ctypes.data, spmv_desc_u.desc,
                self.spmv_cuda_dtype, self.spmv_alg, 
                spmv_buff.data.ptr)
        _cublas.setPointerMode(self.cublas_handle, self.cublas_pointer_mode)


        if self.cusparse_handle is None:
            res += self.A.T @ x
        else:
            _cusparse.spMV(
                self.cusparse_handle, self.spmv_op_atriu, self.spmv_alpha.ctypes.data,
                self.spmv_desc_A.desc, 
                spmv_desc_v.desc,
                self.spmv_betatriu.ctypes.data, spmv_desc_u.desc,
                self.spmv_cuda_dtype, self.spmv_alg, 
                spmv_bufftemptriu.data.ptr)

        res -= cupy.multiply(self.A_diag ,x)
        #res += self.HotellingShift*((self.QHotelling@x)[None,:]@self.QHotelling).flatten()
        #self._dohotelling = False
        if self._dohotelling:

            if self.cusparse_handle_Hotelling is None:

                res += self.HotellingShift*((self.QHotelling@x)[None,:]@self.QHotelling).flatten()

            else:

                spmv_desc_v = cusparse.DnVecDescriptor.create(x)
                buff_size = _cusparse.spMV_bufferSize(
                    self.cusparse_handle_Hotelling, self.spmv_op_a_Hotelling1, 
                    self.spmv_alpha_Hotelling2.ctypes.data,
                    self.spmv_desc_QHotelling.desc, spmv_desc_v.desc, self.spmv_beta_Hotelling1.ctypes.data,
                    self.spmv_desc_vhotelling1.desc, self.spmv_cuda_dtype, self.spmv_alg)
                spmv_buff = cupy.empty(buff_size, cupy.int8)
                # NOTE self.hotellingshift * QX
                _cusparse.spMV(
                    self.cusparse_handle_Hotelling, self.spmv_op_a_Hotelling1, 
                    self.spmv_alpha_Hotelling1.ctypes.data,
                    self.spmv_desc_QHotelling.desc, 
                    spmv_desc_v.desc,
                    self.spmv_beta_Hotelling1.ctypes.data, self.spmv_desc_vhotelling1.desc,
                    self.spmv_cuda_dtype, self.spmv_alg, 
                    spmv_buff.data.ptr)


                # res = Q^T (shift * Q X) + res 
                spmv_desc_u = cusparse.DnVecDescriptor.create(res)
                _cusparse.spMV(
                    self.cusparse_handle_Hotelling, self.spmv_op_a_Hotelling2, 
                    self.spmv_alpha_Hotelling2.ctypes.data,
                    self.spmv_desc_QHotelling.desc, 
                    self.spmv_desc_vhotelling1.desc,
                    self.spmv_beta_Hotelling2.ctypes.data, spmv_desc_u.desc,
                    self.spmv_cuda_dtype, self.spmv_alg, 
                    spmv_buff.data.ptr)


        return res

# ================================
# Correction Equation Solver
# ================================


def OOC4_systemmatrix_Q_r_tol_uu_JdCorrectedZ(system_matrix, Q,
                r,
                tol, uu,
                User_HalfMemMode = True,
                maxiter=20):
    
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    N = r.shape[0]



    x0=cp.random.random(N)*(tol/N)



    
    precon=None
    right_hand_side = - r 
    # =======================================
    # Choice of solvers
    # ========================================
    # TODO A question here is should we use preconditioner e.g. jacobi == diag(A)^-1
    # NOTE As long as the matrix is kept positive definite it's okay to CG too.

    solver = InchingLiteInteger.Burn.JacobiDavidsonHotellingDeflation.IterativeSolvers.gmresDeflate
    #solver = cupyx.scipy.sparse.linalg.minres # 
    z, _ = solver(system_matrix,right_hand_side,uu= uu, tol = tol,
                                M = precon,
                                maxiter = maxiter, x0 = x0)
    x0 = None
    right_hand_side = None
    _ = None
    solver = None
    
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    return z

# ======================
# Main
# ========================


def S_HeigvalJDMHD_HeigvecJDMHD(A , A_diag,
        
        k=16,
        tol=1e-10,
        maxiter=1000,
        User_CorrectionSolverMaxiter=20,
        
        
        User_WorkspaceSizeFactor = 2,

        User_HalfMemMode= True,
        User_IntermediateConvergenceTol=1e-3,
        User_GapEstimate=0.1,
        User_FactoringToleranceOnCorrection = 1e-4,
        User_Q_HotellingDeflation = None,
        User_HotellingShift = 10.0,

        User_PolynomialParams = None,



        ):


    if (User_PolynomialParams is not None):
        print("")

    assert (User_PolynomialParams is None
            ), "ABORTED. WARNING. The J in JDM is targeting the smallest eigval due to implicit inversion which counteract the goal of Cheb to map to the largest eigval. You are advised to Use CDSI instead. Though it will still work it will take much longer. It will not even work if the dirac cheb is applied."
    assert User_WorkspaceSizeFactor > 1, "ABORTED. User_WorkspaceSizeFactor: int > 1"
    
    # NOTE  We have not included the linear solvers' preconditioner, 
    #       in most cases it does not really help much and you need to 
    #       store and calculate the likely denser preconditoner e.g. ILU1. of course ILU0 is trivial
    User_CorrectionSolverPreconditioner = False
    User_CorrectionSolver ='gmres'  # NOTE A natural choice is MINRES rather than GMRES for the symmtric matrix
    N=A.shape[0]
    
    jmax = k*User_WorkspaceSizeFactor 
    jmin = min(N, k)

    dtype_temp = A.dtype

    






    PART00_Initialization = True
    if PART00_Initialization:


        # ==============================
        # Memory management
        # ===============================
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()

        # Placegolders
        Av = cupy.empty((N,)).astype(A.dtype)   # NOTE A v =: Av
        r =  cupy.empty((N,1)).astype(A.dtype)   # NOTE Av - \theta v =: r
        
        eigval_converged = cp.zeros(k)


        # =======================
        # Ritz
        # =======================
        # NOTE First Memory Jump Point. The Workspace Ritz vectors to allow FRO on the z. 
        V = cupy.zeros([N,jmax+1]).astype(A.dtype) 



        # NOTE Second Memory Jump Point. The pool of Ritz vectors (to be) accepted as converged
        Q = cp.zeros([N,jmin]).astype(A.dtype) 


        # NOTE The first ritz vector v0
        v0 = cp.random.random((N,1)) - 0.5
        v0 = v0/cp.sqrt((cp.multiply(v0,v0)).sum(axis=0)) 

        # ===========================================
        # NOTE define krylov protocol to be used. 
        # ============================================
        if User_HalfMemMode:
            KrylovAv = InchingLiteInteger.Burn.Krylov.T3.OOC2_HalfMemS_v_KrylovAv_VOID(A, A_diag)
        else:
            KrylovAv = InchingLiteInteger.Burn.Krylov.T3.OOC2_FullMemS_v_KrylovAv_VOID(A, A_diag)

        if User_PolynomialParams is not None:
            ChebyshevAvC = InchingLiteInteger.Burn.PolynomialFilters.T2.OOC2_A_Adiag_ChebyshevAv(A, A_diag, User_PolynomialParams = User_PolynomialParams, User_HalfMemMode = User_HalfMemMode)
            ChebyshevAv = ChebyshevAvC.ChebyshevAv


        if User_Q_HotellingDeflation is None:
            print("WARNING. Hotelling deflation not in use")
            _dohotelling = False
        else:
            _dohotelling = True


        # ======================
        # Printing
        # =======================

        if N > 2000000*3:
            printing_ = 1
        else:
            printing_ = 100

        if User_PolynomialParams is None:
            pass
        else:
            printing_ = min(printing_, 10)
        # ======================
        # Indices
        # ===================
        jaccept_Q = 0
        



    # =======================
    # Half leg
    # =======================
    
    if User_PolynomialParams is None:
        KrylovAv(A,cupy.ravel(v0),Av)
    else:
        #Av = ChebyshevAv(A, A_diag, cupy.ravel(v0), User_PolynomialParams, User_ReturnRho = False, User_HalfMemMode = User_HalfMemMode)
        Av = ChebyshevAv(A, cupy.ravel(v0), User_ReturnRho = False)

    if _dohotelling:
        InchingLiteInteger.Burn.Orthogonalization.T3.T3_QHotelling_x_Ax_HotelledAx(
            User_Q_HotellingDeflation, v0, Av, HotellingShift=User_HotellingShift)


    V[:,0] = v0[:,0]
    u = v0
    G = v0.T.dot(Av[:, None])
    theta = G[0,0]


    print("Start JDM Coarse Iter")
    
    # ========================
    # Main loop
    # =========================

    n_RitzConverged = 0
    for i_iter in range(maxiter):

        # NOTE The cp eigh may fail to converge...
        S, W = numpy.linalg.eigh(cp.asnumpy(G), UPLO='L')
        S = cp.array(S)
        W = cp.array(W)

        if User_PolynomialParams is None:
            pass
        else:
            # NOTE Because now the target eigval is mapped to the [ thres, 1] we now want the largest eigval!
            # NOTE Also note that the S[0] is the one we always want to apply rational filter i.e. shift-invert on.
            S = S[::-1]
            W = W[:,::-1]


        while True:

            theta = S[0]
            u = V[:,:W.shape[0]].dot(W[:,:1]) 

            

            if User_PolynomialParams is None:
                KrylovAv(A,cupy.ravel(u),cupy.ravel(r))
            else:
                print("This should not be done. See Remark.")
                r = ChebyshevAv(A, cupy.ravel(u), User_ReturnRho = False)
                r = r[:,None]
            

                #print(r.shape, u.shape)
            # NOTE This is necessary unfortunately
            if _dohotelling:
                # TODO The kernel here may be memory unstable for unknown reason. Dig into this if necessary.
                # NOTE This is unexpectedly slower, likely because the matrix has to be interpreted.
                r = InchingLiteInteger.Burn.Orthogonalization.T3.T3_QHotelling_x_Ax_HotelledAx(
                    User_Q_HotellingDeflation, cupy.ravel(u) , cupy.ravel(r), HotellingShift=User_HotellingShift)
                r = r[:,None]
                
                """
                # NOTE THis is correct
                r += (User_HotellingShift*(
                            (User_HD_Eigvec@cupy.ravel(u))[None,:]
                            )@User_HD_Eigvec).T
                """


            r -= theta*u


            cur_tol = cublas.nrm2(cupy.ravel(r)) 

            if i_iter % printing_ == 0:
                print("%s, %s, %s, %s, %s" %(i_iter, cur_tol, theta, User_GapEstimate, n_RitzConverged))
            sigma = theta - User_GapEstimate

            


            # NOTE Second Memory jump point
            Q[:,jaccept_Q] = u[:,0]

            jcurrent_V = int(W.shape[0])
            # NOTE This is the not-converged break
            if cur_tol > tol or ( n_RitzConverged != k-1 and len(S) <= 1):
                break


            
            # ==================================
            # Compile the converged and postprocessing
            # ===================================================

            eigval_converged[n_RitzConverged] = theta
            n_RitzConverged += 1
            
            # TODO I.e. accepting Q_
            jaccept_Q += 1


            V[:,:(W.shape[1]-1)] = V[:,:W.shape[0]].dot(W[:,1:])
            jcurrent_V = int(W.shape[1]-1)

            S = S[1:]
            G, W = cp.diag(S), cp.identity(S.shape[0])
            
            #while True:
            #    pass
            # ==================================
            # All Done!
            # ==================================
            if n_RitzConverged == k:
                print("DONE. We went through %s coarse iter, %s eigval converged" %(i_iter, n_RitzConverged))

                if User_PolynomialParams is None:
                    idx = cupy.argsort(eigval_converged)
                else:
                    # NOTE If we used cheb then we must rearrange unfortunately. 
                    eigval_converged = []
                    Av = cp.zeros(A.shape[0])
                    for i in range(n_RitzConverged):

                        KrylovAv(A,cupy.ravel(Q[:,i]),Av)
                        vAv = Av.dot(cupy.ravel(Q[:,i]))
                        eigval_converged.append(vAv)

                    # NOTE The true eigval recovered
                    eigval_converged = cp.ravel(cp.vstack(eigval_converged))
                    idx = cp.argsort(eigval_converged)

                    # NOTE Remove numbers not in interval. Often overestimation when we do kpm 

                V = None
                S = None
                W = None
                #Q_ = None
                del V,S,W#,Q_
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()
                
                return eigval_converged[idx], Q[:,idx] # TODO return Q?
                # return eigval_converged, Q




        # NOTE restart
        if S.shape[0] == jmax:
            V[:,:jmin] = V[:,:W.shape[0]].dot(W[:,:jmin])
            jcurrent_V = jmin
            S = S[:jmin]
            G, W = cp.diag(S),cp.identity(S.shape[0])

        # NOTE compute the shift
        if cur_tol < User_IntermediateConvergenceTol:
            shift = theta
        else:
            shift = sigma

        
        # NOTE correction equation: solve approximately for z:
        #     (I-Q*Q.H)(A-theta*I)(I-Q*Q.H)z = -r, with z.T*u = 0

        if User_HalfMemMode:
            system_matrix = OOC1_HalfMemA_KrylovLinearOperator(A, A_diag,
                        shift  = shift, QHotelling = User_Q_HotellingDeflation, 
                                        HotellingShift = User_HotellingShift,
                                        _dohotelling = _dohotelling)
        else:
            system_matrix = OOC1_FullMemA_KrylovLinearOperator(A, 
                        shift  = shift, QHotelling = User_Q_HotellingDeflation, 
                                        HotellingShift = User_HotellingShift,
                                        _dohotelling = _dohotelling)
        
        
        z = OOC4_systemmatrix_Q_r_tol_uu_JdCorrectedZ(system_matrix,

                        Q = None, 
                        uu = cupy.ravel(u) / cublas.nrm2(cupy.ravel(u)) ,
                        r=r,
                        User_HalfMemMode= User_HalfMemMode,
                        tol = cur_tol*User_FactoringToleranceOnCorrection,
                        maxiter=User_CorrectionSolverMaxiter
                        )
        
        #z = cupy.copy(cupy.ravel(r)) # NOTE THis is overriding! with this we are doing SI!

        system_matrix = None
        del system_matrix

        # NOTE FRO on z
        z = z[:,cp.newaxis]

        z = InchingLiteInteger.Burn.Orthogonalization.T2.OOC2_qnext_Q_MGSqnext(z,Q[:,:jaccept_Q+1])
        z = InchingLiteInteger.Burn.Orthogonalization.T2.OOC2_qnext_Q_ICGSqnext(z,V[:,:jcurrent_V]) # NOTE Do not overdo this
        
        
            
        if User_PolynomialParams is None:
            KrylovAv(A,cupy.ravel(z),cupy.ravel(Av))
        else:
            print("This should not be done. See Remark.")
            Av = ChebyshevAv(A, cupy.ravel(z), User_ReturnRho = False)
            Av = cupy.ravel(Av)

        if len(Av.shape) == 1:
            Av = Av[:,None]

        if _dohotelling:
            Av += User_HotellingShift*((User_Q_HotellingDeflation@cupy.ravel(z))[None,:]@User_Q_HotellingDeflation).T


        # NOTE Construct small matrix G
        G = cp.vstack((cp.hstack((G, V[:,:jcurrent_V].T.dot(Av))),
                        cp.hstack((Av.T.dot(V[:,:jcurrent_V]), Av.T.dot(z)))))

        V[:,jcurrent_V]  = z[:,0]
    
        # NOTE This is very important otherwise mem leak
        z = None
        Q_ = None
        S = None

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        del z
        del Q_



    eigval_converged[n_RitzConverged] = theta
    n_RitzConverged += 1

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()



    idx = cupy.argsort(eigval_converged)

    print("ABORTED. It did not actually converged! We went through %s coarse iter and collected %s converged" %(
        i_iter, n_RitzConverged))
    return eigval_converged[idx], Q[:,idx]




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
