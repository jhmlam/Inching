# ======================
# ARCHIVED. This is CORRECT
# ========================

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
sys.path.append('../InchingLite/Burn/')
import InchingLite.Burn.Orthogonalization.T3
import InchingLite.Burn.Orthogonalization.T2



# ====================================
# Thick Restart Lanczos
# ==================================
# NOTE REMARK. While the hotelling is correct, the calcualation is 6 times more in runtime.
#      if the hotelling is done at the Lanczos loop
#      At the end we do not do hotelling. Tradoff too large. though it is implemented,

def S_HeigvalTRLMHD_HeigvecTRLMHD(a, k=32, 
            maxiter=None, tol=0,
            User_HalfMemMode = True,

            User_Q_HotellingDeflation = None,
            User_HotellingShift = 10.0,
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
        n = a.shape[0]
        assert k < n, "ABORTED. k must be smaller than n"
        assert a.ndim == 2 , "ABORTED. It is a tensor not rank 2!"
        assert a.shape[0] == a.shape[1], "ABORTED. square"


        assert (k%8 == 0)
        assert k >= 32, "ABORTED. we did not test on less than 32 modes, as the number ritz vectors is too small."

        # NOTE The workspace
        ncv = min(max(2 * k, k + 32), n - 1)
        

        if maxiter is None:
            maxiter = 10 * n
        
        if tol == 0:
            tol = numpy.finfo(a.dtype).eps
        print("There are %s Ritz vectors, tol = %s"%(ncv, tol))


    # ===================================
    # Initialise
    # ===================================
    PART01_InitializeEmpties = True
    if PART01_InitializeEmpties:
        alpha = cupy.zeros((ncv,), dtype=a.dtype)
        beta = cupy.zeros((ncv,), dtype=a.dtype.char.lower())
        V = cupy.empty((ncv, n), dtype=a.dtype)

        # Set initial vector
        # NOTE we will use these u and uu for temporary storages of this size.
        u = cupy.random.random((n,)).astype(a.dtype)
        uu = cupy.empty((k,), dtype=a.dtype)

        # Normlaise
        V[0] = u / cublas.nrm2(u)





    # ===========================================
    # NOTE define protocol to be used. 
    # ============================================
    # NOTE Krylov
    if User_HalfMemMode:
        KrylovAv = InchingLite.Burn.Krylov.T3.OOC2_HalfMemS_v_KrylovAv_VOID(a)
    else:
        KrylovAv = InchingLite.Burn.Krylov.T3.OOC2_FullMemS_v_KrylovAv_VOID(a)


    # NOTE Lanczos. 
    if User_HalfMemMode:
        Lanczos = OOC7_HalfMemS_RitzV_u_alpha_beta_kplus1_numRitz_VOID(a)
    else:
        Lanczos = OOC7_FullMemS_RitzV_u_alpha_beta_kplus1_numRitz_VOID(a)


    # NOTE Hotelling
    if User_Q_HotellingDeflation is None:
        print("WARNING. Hotelling deflation not in use")
        _dohotelling = False
    else:
        _dohotelling = True


    # ======================================
    # Loop
    # ======================================
    # NOTE ARPACK style initilze
    # Lanczos iteration
    Lanczos(a, V, u, alpha, beta, 0, ncv, User_Q_HotellingDeflation, User_HotellingShift = User_HotellingShift)
    iter = ncv
    # NOTE beta_k == None. This is the a really-tridiag
    w, s = OOC4_alpha_beta_betak_k_TrdEigvalSel_TrdEigvecSel(alpha, beta, None, k)

    # NOTE 
    #      Cuda transpose is expensive
    #x = V.T @ s # NOTE This is a matrix of size (64, 3n).T (64,64) = (3n,64) and it's transpose is written into V[:k]
    #x = s.T @ V # NOTE all we need is (64,3n) = (64,64)^T (64,3n) 
    V[:k] = s.T @ V
    #print(x.shape,V.shape, s.shape)
    #sys.exit()


    # NOTE Compute residual
    beta_k = beta[-1] * s[-1, :]
    res = cublas.nrm2(beta_k)
    #print('init beta_k', beta_k)



    coarse_iter = 0
    for coarse_iter in range(maxiter):

        beta[:k] = 0
        alpha[:k] = w

        # =======================
        # Single MGS here
        # =========================
        # NOTE only a single MGS is done. FRO does not help
        u = InchingLite.Burn.Orthogonalization.T2.T2_vnext_V_MGSvnext(u, V[:k].T, k=None)
        u /= cublas.nrm2(u)
        V[k] = u 

        # =============================
        # Krylov
        # ============================
        # NOTE reuse the last one to get u = A V[k]
        KrylovAv(a,V[k],u)


        # =====================================
        # NOTE Hotelling 
        # ======================================

        if _dohotelling:
            # TODO The kernel here may be memory unstable for unknown reason. Dig into this if necessary.
            # NOTE This is unexpectedly slower, likely because the matrix has to be interpreted.
            u = InchingLite.Burn.Orthogonalization.T3.T3_QHotelling_x_Ax_HotelledAx(User_Q_HotellingDeflation,  V[k],u, HotellingShift=User_HotellingShift)




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
        Lanczos(a, V, u, alpha, beta, k + 1, ncv, User_Q_HotellingDeflation, User_HotellingShift = User_HotellingShift)


        # ==============================
        # Not-really-tridaig 
        # ==============================
        w, s = OOC4_alpha_beta_betak_k_TrdEigvalSel_TrdEigvecSel(alpha, beta, beta_k, k)
        # Store the approx eigenvector back to V[:k]
        V[:k] = s.T @ V


        # ========================================
        # Residual 
        # ======================================
        # NOTE Compute residual. 
        # NOTE That comparing tol with res a bound result, 
        #      it does not mean we need res==1e-15 to reach || eigval - rayleighquotient||_2 == 1e-15 
        #print('how beta_k goes?', beta_k)
        beta_k = beta[-1] * s[-1, :]
        res = cublas.nrm2(beta_k)

        if  res <= tol: 
            break
        
        iter += ncv - k
        coarse_iter += 1

        # NOTE Monitor convergence by res
        if n > 2000000*3:
            printing_ = 1
        else:
            printing_ = 100

        if coarse_iter % printing_ == 0:
            print('Coarse_iter %s Estimate at %s. Ritz values follows' %(coarse_iter, res))








    print('Total number of iterations went through %s in %s seconds'%(coarse_iter, time.time() - st))

    idx = cupy.argsort(w)




    # ===========================
    # Meory managemnt
    # =============================

    xx = V[idx,:].T
    V = None
    alpha = None
    beta = None
    beta_k = None
    res = None
    u = None
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()



    #return w[idx], x[:, idx]
    #return w[idx], x.T[:, idx]
    return w[idx], xx




# =========================================
# Construction of Tridiag
# ==========================================
# NOTE The minieigenproblem 
#      if beta_k is None we have the regular tridiag.
def OOC4_alpha_beta_betak_k_TrdEigvalSel_TrdEigvecSel(alpha, beta, beta_k, k):
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
    w, s = numpy.linalg.eigh(t)

    # Pick-up k ritz-values and ritz-vectors
    # NOTE numpy default ascending
    idx = numpy.argsort(w)[::-1]
    
    wk = w[idx[-k:][::-1]]
    sk = s[:, idx[-k:][::-1]]
    return cupy.array(wk), cupy.array(sk)







# ==================================
# Lanczos
# ====================================
# NOTE normalize the ritz. Using the cupy elementwise kernel
OOC6_u_beta_i_n_v_V_vhat_Vhat = cupy.ElementwiseKernel(
    'T u, raw S beta, int32 j, int32 n', 
    'T v, raw T V',
    'v = u / beta[j]; V[i + (j+1) * n] = v;', 'cupy_eigsh_normalize'
)



# NOTE This ios  the Lanczos loop
def OOC7_FullMemS_RitzV_u_alpha_beta_kplus1_numRitz_VOID(A):


    cublas_handle = device.get_cublas_handle()
    cublas_pointer_mode = _cublas.getPointerMode(cublas_handle)
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

    cusparse_handle = None
    if _csr.isspmatrix_csr(A) and cusparse.check_availability('spmv'):
        cusparse_handle = device.get_cusparse_handle()
        spmv_op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        spmv_alpha = numpy.array(1.0, A.dtype)
        spmv_beta = numpy.array(0.0, A.dtype)
        spmv_cuda_dtype = _dtype.to_cuda_dtype(A.dtype)
        spmv_alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT


    n = A.shape[0]
    v = cupy.empty((n,), dtype=A.dtype)
    #uu = cupy.empty((ncv,), dtype=A.dtype)
    one = numpy.array(1.0, dtype=A.dtype)
    zero = numpy.array(0.0, dtype=A.dtype)
    mone = numpy.array(-1.0, dtype=A.dtype)

    #outer_A = A

    def aux(A, V, u, alpha, beta, i_start, i_end, User_Q_HotellingDeflation = None, User_HotellingShift = 10.0):


        # NOTE Hotelling
        if User_Q_HotellingDeflation is None:
            #print("WARNING. Hotelling deflation not in use")
            _dohotelling = False
        else:
            _dohotelling = True



        #assert A is outer_A
        ncv = V.shape[0]
        uu = cupy.empty((ncv,), dtype=A.dtype)
        # Get ready for spmv if enabled
        if cusparse_handle is not None:
            # Note: I would like to reuse descriptors and working buffer
            # on the next update, but I gave it up because it sometimes
            # caused illegal memory access error.
            spmv_desc_A = cusparse.SpMatDescriptor.create(A)
            spmv_desc_v = cusparse.DnVecDescriptor.create(v)
            spmv_desc_u = cusparse.DnVecDescriptor.create(u)
            buff_size = _cusparse.spMV_bufferSize(
                cusparse_handle, spmv_op_a, spmv_alpha.ctypes.data,
                spmv_desc_A.desc, spmv_desc_v.desc, spmv_beta.ctypes.data,
                spmv_desc_u.desc, spmv_cuda_dtype, spmv_alg)
            spmv_buff = cupy.empty(buff_size, cupy.int8)
            #print("cusparse_handle not none")

        v[...] = V[i_start]
        for i in range(i_start, i_end):
            # NOTE Krylov
            if cusparse_handle is None:
                u[...] = A @ v
            else:
                _cusparse.spMV(
                    cusparse_handle, spmv_op_a, spmv_alpha.ctypes.data,
                    spmv_desc_A.desc, 
                    spmv_desc_v.desc,
                    spmv_beta.ctypes.data, spmv_desc_u.desc,
                    spmv_cuda_dtype, spmv_alg, 
                    spmv_buff.data.ptr)

            # NOTE Get alpha
            _cublas.setPointerMode(
                cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                dotc(cublas_handle, n, v.data.ptr, 1, u.data.ptr, 1,
                     alpha.data.ptr + i * alpha.itemsize)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)


            
            # =====================================
            # NOTE Hotelling 
            # ======================================

            if _dohotelling:
                # TODO The kernel here may be memory unstable for unknown reason. Dig into this if necessary.
                # NOTE This is unexpectedly slower, likely because the matrix has to be interpreted.
                u = InchingLite.Burn.Orthogonalization.T3.T3_QHotelling_x_Ax_HotelledAx(User_Q_HotellingDeflation, v , u, HotellingShift=User_HotellingShift)


            
            # =================
            # FRO
            # ====================
            # Orthogonalize
            gemv(cublas_handle, _cublas.CUBLAS_OP_C,
                 n, i + 1,
                 one.ctypes.data, V.data.ptr, n,
                 u.data.ptr, 1,
                 zero.ctypes.data, uu.data.ptr, 1)
            #print(uu)
            gemv(cublas_handle, _cublas.CUBLAS_OP_N,
                 n, i + 1,
                 mone.ctypes.data, V.data.ptr, n,
                 uu.data.ptr, 1,
                 one.ctypes.data, u.data.ptr, 1)

            #print(u.flags , V[:i+1].flags)
            #print('orth1??', V[:i+1]@u ) # YES
            #print(u.shape, V[:i+1].shape)
            #if i > 100 : 
            #    sys.exit()
            # Call nrm2
            _cublas.setPointerMode(
                cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                nrm2(cublas_handle, n, u.data.ptr, 1,
                     beta.data.ptr + i * beta.itemsize)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)



            # Orthogonalize
            gemv(cublas_handle, _cublas.CUBLAS_OP_C,
                 n, i + 1,
                 one.ctypes.data, V.data.ptr, n,
                 u.data.ptr, 1,
                 zero.ctypes.data, uu.data.ptr, 1)
            gemv(cublas_handle, _cublas.CUBLAS_OP_N,
                 n, i + 1,
                 mone.ctypes.data, V.data.ptr, n,
                 uu.data.ptr, 1,
                 one.ctypes.data, u.data.ptr, 1)


            #print('orth2??', V[:i+1]@u ) # YES
            #sys.exit()

            # Call nrm2
            _cublas.setPointerMode(
                cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                nrm2(cublas_handle, n, u.data.ptr, 1,
                     beta.data.ptr + i * beta.itemsize)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)







            # Break here as the normalization below touches V[i+1]
            if i >= i_end - 1:
                break

            # NOTE THis is the 
            OOC6_u_beta_i_n_v_V_vhat_Vhat(u, beta, i, n, v, V)
        #print('how beta progress?', beta) # NOTE never underflow. 
        #print('how alpha progress', alpha)
        uu = None
        del uu
    return aux



# NOTE This ios  the Lanczos loop
def OOC7_HalfMemS_RitzV_u_alpha_beta_kplus1_numRitz_VOID(A):
    cublas_handle = device.get_cublas_handle()
    cublas_pointer_mode = _cublas.getPointerMode(cublas_handle)
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

    cusparse_handle = None
    if _csr.isspmatrix_csr(A) and cusparse.check_availability('spmv'):
        cusparse_handle = device.get_cusparse_handle()
        spmv_op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        spmv_op_atriu = _cusparse.CUSPARSE_OPERATION_TRANSPOSE 

        spmv_alpha = numpy.array(1.0, A.dtype)
        spmv_beta = numpy.array(0.0, A.dtype)
        spmv_betatriu = numpy.array(1.0, A.dtype)
        spmv_alphadiag = numpy.array(-1.0, A.dtype)
        spmv_cuda_dtype = _dtype.to_cuda_dtype(A.dtype)
        spmv_alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT
        
    n = A.shape[0]
    v = cupy.empty((n,), dtype=A.dtype)
    utemptriu = cupy.empty((n,), dtype=A.dtype)
    utempdiag = cupy.empty((n,), dtype=A.dtype)
    #uu = cupy.empty((ncv,), dtype=A.dtype)
    one = numpy.array(1.0, dtype=A.dtype)
    zero = numpy.array(0.0, dtype=A.dtype)
    mone = numpy.array(-1.0, dtype=A.dtype)

    #outer_A = A

    def aux(A, V, u, alpha, beta, i_start, i_end, User_Q_HotellingDeflation = None, User_HotellingShift = 10.0):
        #assert A is outer_A
        ncv = V.shape[0]
        uu = cupy.empty((ncv,), dtype=A.dtype)


        # NOTE Hotelling
        if User_Q_HotellingDeflation is None:
            #print("WARNING. Hotelling deflation not in use")
            _dohotelling = False
        else:
            _dohotelling = True

        # Get ready for spmv if enabled
        if cusparse_handle is not None:
            # Note: I would like to reuse descriptors and working buffer
            # on the next update, but I gave it up because it sometimes
            # caused illegal memory access error.
            spmv_desc_A = cusparse.SpMatDescriptor.create(A)
            spmv_desc_v = cusparse.DnVecDescriptor.create(v)
            spmv_desc_u = cusparse.DnVecDescriptor.create(u)

            spmv_desc_utemptriu = cusparse.DnVecDescriptor.create(utemptriu)
            spmv_desc_utempdiag = cusparse.DnVecDescriptor.create(utempdiag)


            buff_size = _cusparse.spMV_bufferSize(
                cusparse_handle, spmv_op_a, spmv_alpha.ctypes.data,
                spmv_desc_A.desc, spmv_desc_v.desc, spmv_beta.ctypes.data,
                spmv_desc_u.desc, spmv_cuda_dtype, spmv_alg)
            spmv_buff = cupy.empty(buff_size, cupy.int8)
            spmv_bufftemptriu = cupy.empty(buff_size, cupy.int8)
            #spmv_bufftempdiag = cupy.empty(buff_size, cupy.int8)
            #print(spmv_desc_A)
            #print("cusparse_handle not none")

        v[...] = V[i_start]
        for i in range(i_start, i_end):
            # ===============================
            # NOTE Krylov
            # ==============================
            # Matrix-vector multiplication
            # u = [L+D]v
            # u += [D+U]v
            # u -= Dv
            if cusparse_handle is None:
                u[...] = A @ v
            else:
                _cusparse.spMV(
                    cusparse_handle, spmv_op_a, spmv_alpha.ctypes.data,
                    spmv_desc_A.desc, 
                    spmv_desc_v.desc,
                    spmv_beta.ctypes.data, spmv_desc_u.desc,
                    spmv_cuda_dtype, spmv_alg, 
                    spmv_buff.data.ptr)

            if cusparse_handle is None:
                u += A.T @ v
            else:
                _cusparse.spMV(
                    cusparse_handle, spmv_op_atriu, spmv_alpha.ctypes.data,
                    spmv_desc_A.desc, 
                    spmv_desc_v.desc,
                    spmv_betatriu.ctypes.data, spmv_desc_u.desc,
                    spmv_cuda_dtype, spmv_alg, 
                    spmv_bufftemptriu.data.ptr)


            u -= cupy.multiply(A.diagonal(k=0) ,v)
            
            # =====================================
            # NOTE Hotelling 
            # ======================================

            if _dohotelling:
                # TODO The kernel here may be memory unstable for unknown reason. Dig into this if necessary.
                # NOTE This is unexpectedly slower, likely because the matrix has to be interpreted.
                u = InchingLite.Burn.Orthogonalization.T3.T3_QHotelling_x_Ax_HotelledAx(User_Q_HotellingDeflation, v , u, HotellingShift=User_HotellingShift)



            # ====================================
            # Alpha
            # =====================================

            _cublas.setPointerMode(
                cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                dotc(cublas_handle, n, v.data.ptr, 1, u.data.ptr, 1,
                     alpha.data.ptr + i * alpha.itemsize)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)
            #gggg = (V[i ]@u )   
            #hhhh =   u - V[i ].T * gggg
            #print('baby test', V[:i+1]@hhhh)


            # =============================
            # FRO
            # ==================================
            # Orthogonalize
            gemv(cublas_handle, _cublas.CUBLAS_OP_C,
                 n, i + 1,
                 one.ctypes.data, V.data.ptr, n,
                 u.data.ptr, 1,
                 zero.ctypes.data, uu.data.ptr, 1)
            #print(uu)
            gemv(cublas_handle, _cublas.CUBLAS_OP_N,
                 n, i + 1,
                 mone.ctypes.data, V.data.ptr, n,
                 uu.data.ptr, 1,
                 one.ctypes.data, u.data.ptr, 1)

            #print(u.flags , V[:i+1].flags)
            #print('orth1??', V[:i+1]@u ) # YES
            #print(u.shape, V[:i+1].shape)
            #if i > 100 : 
            #    sys.exit()
            # Call nrm2
            _cublas.setPointerMode(
                cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                nrm2(cublas_handle, n, u.data.ptr, 1,
                     beta.data.ptr + i * beta.itemsize)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)


            # Orthogonalize
            gemv(cublas_handle, _cublas.CUBLAS_OP_C,
                 n, i + 1,
                 one.ctypes.data, V.data.ptr, n,
                 u.data.ptr, 1,
                 zero.ctypes.data, uu.data.ptr, 1)
            gemv(cublas_handle, _cublas.CUBLAS_OP_N,
                 n, i + 1,
                 mone.ctypes.data, V.data.ptr, n,
                 uu.data.ptr, 1,
                 one.ctypes.data, u.data.ptr, 1)


            #print('orth2??', V[:i+1]@u ) # YES
            #sys.exit()

            # Call nrm2
            _cublas.setPointerMode(
                cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                nrm2(cublas_handle, n, u.data.ptr, 1,
                     beta.data.ptr + i * beta.itemsize)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)


            # Break here as the normalization below touches V[i+1]
            if i >= i_end - 1:
                break

            OOC6_u_beta_i_n_v_V_vhat_Vhat(u, beta, i, n, v, V)
        #print('how beta progress?', beta) # NOTE never underflow. 
        #print('how alpha progress', alpha)
        uu = None
        del uu
        
    return aux


