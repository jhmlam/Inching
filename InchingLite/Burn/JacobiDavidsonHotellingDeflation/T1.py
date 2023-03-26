# ====================
# ARCHIVED. This works
# ==============================


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
import InchingLite.Burn.JacobiDavidsonHotellingDeflation.IterativeSolvers
import InchingLite.Burn.Orthogonalization.T3
import InchingLite.Burn.Orthogonalization.T2
import InchingLite.Burn.Krylov.T3
# ==========================
# Orthogonalization
# ===========================


def OOC2_qnext_Q_MGSqnext(u,Q):
    # NOTE THis can be easily modifed to QR algo.
    for i in range(Q.shape[1]):
        s = u.T.dot(Q[:,i:i+1])
        u = u - s*Q[:,i:i+1]

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


# ================================
# Krylov Iteration
# ==================================

# NOTE What this does is that it package the system matrix for solver(system_matrix to work on
#      https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.LinearOperator.html?highlight=LinearOperator
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
        if _csr.isspmatrix_csr(A) and cusparse.check_availability('spmv'):
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

    def __init__(self, A, shift = 0, QHotelling = cp.zeros(3),HotellingShift = 10.0, _dohotelling = False):


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
        if _csr.isspmatrix_csr(A) and cusparse.check_availability('spmv'):
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

        res -= cupy.multiply(self.A.diagonal(k=0) ,x)
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


def OOC5_systemmatrix_Q_r_F_tol_JdCorrectedZ(system_matrix, Q,
                r,
                F,tol,
                User_HalfMemMode = True,
                maxiter=20):

    N = Q.shape[0]

    #print(F.shape)
    
    invF=cp.linalg.inv(F)
    x0=cp.random.random(N)*(tol/N)




    precon=None
    right_hand_side = - r + Q.dot(
                                invF).dot(
                                    Q.T.dot(
                                        r)
                                        )
    
    # =======================================
    # Choice of solvers
    # ========================================
    # TODO Should also try minres
    # NOTE As long as the matrix is kept positive definite it's okay to CG too.

    solver = InchingLite.Burn.JacobiDavidsonHotellingDeflation.IterativeSolvers.gmres
    z, _ = solver(system_matrix,right_hand_side,tol = tol,
                                M = precon,
                                maxiter = maxiter, x0 = x0)
    
    return z



# ======================
# Main
# ========================

def S_HeigvalJDMHD_HeigvecJDMHD(A,
        v0=None,
        k=1,
        tol=1e-10,
        maxiter=1000,
        linear_solver='gmres',
        linear_solver_maxiter=20,
        
        jmax=16,jmin=8,
        sigma=None,

        User_HalfMemMode= True,
        converge_bound=1e-3,
        gap_estimate=0.1,
        User_FactoringToleranceOnCorrection = 1e-4,

        User_Q_HotellingDeflation = None,
        User_HotellingShift = 10.0
        ):


    # NOTE We have not included the linear solvers' preconditioner, 
    #      in most cases it does not really help much and you need to store and calculate the likely denser preconditoner e.g. ILU1
    User_LinearSolverPreconditioner = False
    linear_solver='gmres'
    N=A.shape[0]

    assert User_HalfMemMode, "ABORTED. Only support half mem mode."
    
    if User_Q_HotellingDeflation is None:
        print("WARNING. Hotelling deflation not in use")
        _dohotelling = False
    else:
        _dohotelling = True

    PART00_Initialization = True
    if PART00_Initialization:


        # ==============================
        # Memory management
        # ===============================
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()


        # NOTE The first ritz vector v0
        if v0 is None:
            v0 = cp.random.random((N,1)) - 0.5
        elif cp.ndim(v0) == 1: 
            v0 = v0[:,cp.newaxis]
        v0 = v0/cp.sqrt((cp.multiply(v0,v0)).sum(axis=0)) 
        print(v0.shape)



        # Placegolders
        Av = cupy.empty((N,)).astype(A.dtype)   # NOTE A v =: Av
        r =  cupy.empty((N,1)).astype(A.dtype)   # NOTE Av - \theta v =: r
        #r_ =  cupy.zeros((N,)).astype(A.dtype)
        eigval_converged = cp.zeros(k)
        Q = cp.zeros([N,0]) # NOTE Q and F are variable length arrays build up upon successful convergence.
        F = cp.zeros([0,0])

        # ===========================================
        # NOTE define krylov protocol to be used. 
        # ============================================
        if User_HalfMemMode:
            KrylovAv = InchingLite.Burn.Krylov.T3.OOC2_HalfMemS_v_KrylovAv_VOID(A)
        else:
            KrylovAv = InchingLite.Burn.Krylov.T3.OOC2_FullMemS_v_KrylovAv_VOID(A)




        

    

    KrylovAv(A,cupy.ravel(v0),Av)

    if _dohotelling:
        InchingLite.Burn.Orthogonalization.T3.T3_QHotelling_x_Ax_HotelledAx(User_Q_HotellingDeflation, v0, Av, HotellingShift=User_HotellingShift)


    V = v0
    u = v0
    G = v0.T.dot(Av[:, None])
    theta = G[0,0]


    print("Start JDM Coarse Iter")
    

    n_RitzConverged = 0
    st = time.time()
    for i_iter in range(maxiter):

        S, W = cp.linalg.eigh(G, UPLO='L')

        while True:
            #Ritz approximation.
            theta,u = S[0],V.dot(W[:,:1])  
            #print(r.shape, cupy.ravel(r).shape)          
            KrylovAv(A,cupy.ravel(u),cupy.ravel(r))
            
            # NOTE This is necessary unfortunately
            if _dohotelling:
                # TODO The kernel here may be memory unstable for unknown reason. Dig into this if necessary.
                # NOTE This is unexpectedly slower, likely because the matrix has to be interpreted.
                r = InchingLite.Burn.Orthogonalization.T3.T3_QHotelling_x_Ax_HotelledAx(User_Q_HotellingDeflation, cupy.ravel(u) , cupy.ravel(r), HotellingShift=User_HotellingShift)
                #r = T3_QHotelling_x_Ax_HotelledAx(User_Q_HotellingDeflation, cupy.ravel(u) , cupy.ravel(r), HotellingShift=User_HotellingShift)
                r = r[:,None]
                
                """
                # NOTE THis is correct
                r += (User_HotellingShift*(
                            (User_Q_HotellingDeflation@cupy.ravel(u))[None,:]
                            )@User_Q_HotellingDeflation).T
                """


            r -= theta*u

            #print(r.shape)
            cur_tol = cublas.nrm2(cupy.ravel(r)) 
            #print(cur_tol)
            if N > 2000000*3:
                printing_ = 1
            else:
                printing_ = 100

            if i_iter % printing_ == 0:
                print("%s, %s, %s, %s, %s" %(i_iter, cur_tol, theta, gap_estimate, n_RitzConverged))
            sigma = theta - gap_estimate

            


            # NOTE This is a small matrix
            F_ = cp.vstack((cp.hstack((F , Q.T.dot(u))),
                            cp.hstack((u.T.dot(Q), u.T.dot(u)))))
            Q_= cp.concatenate([Q,u],axis=1)
            #print(Q_.shape, F_.shape) # NOTE It does not seem to be the build up of memoryrby Q and F!
            #print(S.shape)
            # NOTE This is the not-converged break
            if cur_tol > tol or ( n_RitzConverged != k-1 and len(S) <= 1):
                break



            # ==================================
            # Compile the converged and postprocessing
            # ===================================================

            eigval_converged[n_RitzConverged] = theta
            n_RitzConverged += 1
            
            F = F_
            Q = Q_


            V = V.dot(W[:,1:])
            S = S[1:]

            G, W = cp.diag(S), cp.identity(S.shape[0])



            if n_RitzConverged == k:
                print("DONE. We went through %s coarse iter, %s eigval converged" %(i_iter, n_RitzConverged))
                idx = cupy.argsort(eigval_converged)
                return eigval_converged[idx], Q[:,idx] # TODO return Q?
                # return eigval_converged, Q




        # NOTE Thick restart
        if S.shape[0] == jmax:
            
            #print("Maximum workspace reached")
            V = V.dot(W[:,:jmin])
            S = S[:jmin]
            G, W = cp.diag(S),cp.identity(S.shape[0])

        # NOTE compute the shift
        if cur_tol < converge_bound:
            shift = theta
        else:
            shift = sigma

        
        # NOTE correction equation: solve approximately for z:
        #     (I-Q*Q.H)(A-theta*I)(I-Q*Q.H)z = -r, with z.T*u = 0

        if User_HalfMemMode:
            system_matrix = OOC1_HalfMemA_KrylovLinearOperator(A, 
                        shift  = shift, QHotelling = User_Q_HotellingDeflation, 
                                        HotellingShift = User_HotellingShift,
                                        _dohotelling = _dohotelling)
        else:
            system_matrix = OOC1_FullMemA_KrylovLinearOperator(A, shift  = shift, 
                                        QHotelling = User_Q_HotellingDeflation, 
                                        HotellingShift = User_HotellingShift,
                                        _dohotelling = _dohotelling)

        z = OOC5_systemmatrix_Q_r_F_tol_JdCorrectedZ(system_matrix,
                        Q=Q_,
                        r=r,
                        F=F_,
                        User_HalfMemMode= User_HalfMemMode,
                        tol = cur_tol*User_FactoringToleranceOnCorrection,
                        maxiter=linear_solver_maxiter
                        )


        system_matrix = None
        del system_matrix

        # NOTE Addededed
        F_ = None
        del F_


        # NOTE FRO on Q_ and then V
        z = InchingLite.Burn.Orthogonalization.T2.T2_vnext_V_MGSvnext(z, Q_)
        z = z[:,cp.newaxis]
        
        #z = OOC2_qnext_Q_MGSqnext(z,Q_)
        z = OOC2_qnext_Q_ICGSqnext(z,V) 


        KrylovAv(A,cupy.ravel(z),cupy.ravel(Av))


        if len(Av.shape) == 1:
            Av = Av[:,None]

        if _dohotelling:
            Av += User_HotellingShift*((User_Q_HotellingDeflation@cupy.ravel(z))[None,:]@User_Q_HotellingDeflation).T


        # NOTE Construct small matrix G
        G = cp.vstack((cp.hstack((G, V.T.dot(Av))),
                        cp.hstack((Av.T.dot(V), Av.T.dot(z)))))


        # NOTE Include corrected z to search space
        V = cp.concatenate([V,z], axis=1)

    
        # NOTE This is very important otherwise mem leak
        z = None
        del z
        Q_ = None
        del Q_

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()




    eigval_converged[n_RitzConverged] = theta
    n_RitzConverged += 1

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()



    idx = cupy.argsort(eigval_converged)

    print("ABORTED. It did not actually converged! We went through %s coarse iter and collected %s converged" %(i_iter, n_RitzConverged))
    return eigval_converged[idx], Q[:,idx]




























# ==========================
# OBSOLETE
# ============================
def OBSOLETE_OOC6_A_Q_r_F_shift_tol_JdCorrectedZ(A,Q,
                r,
                F,shift,tol,
                #method='gmres',
                User_HalfMemMode = True,
                maxiter=20):

    N = Q.shape[0]

    #print(F.shape)
    
    invF=cp.linalg.inv(F)
    x0=cp.random.random(N)*(tol/N)




    if User_HalfMemMode:
        system_matrix = OOC1_HalfMemA_KrylovLinearOperator(A, shift  = shift)
    else:
        system_matrix = OOC1_FullMemA_KrylovLinearOperator(A, shift  = shift)


    precon=None
    right_hand_side = - r + Q.dot(
                                invF).dot(
                                    Q.T.dot(
                                        r)
                                        )
    
    # =======================================
    # Choice of solvers
    # ========================================
    # TODO Should also try minres 
    # NOTE As long as the matrix is kept positive definite it's okay to CG too.
    #      
    solver = InchingLite.Burn.JacobiDavidson.IterativeSolvers.gmres
    z, _ = solver(system_matrix,right_hand_side,tol = tol,
                                M = precon,
                                maxiter = maxiter, x0 = x0)
   
    return z

