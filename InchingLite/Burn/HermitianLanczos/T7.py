

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




# NOTE normalize the ritz. Using the cupy elementwise kernel
OOC6_u_beta_i_n_v_V_vhat_Vhat = cupy.ElementwiseKernel(
    'T u, raw S beta, int32 j, int32 n', 
    'T v, raw T V',
    'v = u / beta[j]; V[i + (j+1) * n] = v;', 'cupy_eigsh_normalize'
)

# =========================
# Lanczos
# =============================
# NOTE This is the lanczos loop with lots of boiler plates
#      n is the shape of A; ncv is the 
# NOTE Full 
def OOC7_FullMemS_RitzV_u_alpha_beta_kplus1_numRitz_VOID(A, n, ncv):
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

    v = cupy.empty((n,), dtype=A.dtype)
    uu = cupy.empty((ncv,), dtype=A.dtype)
    one = numpy.array(1.0, dtype=A.dtype)
    zero = numpy.array(0.0, dtype=A.dtype)
    mone = numpy.array(-1.0, dtype=A.dtype)

    #outer_A = A

    def aux(A, V, u, alpha, beta, i_start, i_end):
        #assert A is outer_A

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

    return aux



# NOTE This ios  the lanczos loop
def OOC7_HalfMemS_RitzV_u_alpha_beta_kplus1_numRitz_VOID(A, n, ncv):
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
        

    v = cupy.empty((n,), dtype=A.dtype)
    utemptriu = cupy.empty((n,), dtype=A.dtype)
    utempdiag = cupy.empty((n,), dtype=A.dtype)
    uu = cupy.empty((ncv,), dtype=A.dtype)
    one = numpy.array(1.0, dtype=A.dtype)
    zero = numpy.array(0.0, dtype=A.dtype)
    mone = numpy.array(-1.0, dtype=A.dtype)

    #outer_A = A

    def aux(A, V, u, alpha, beta, i_start, i_end):
        #assert A is outer_A

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
            


            # Call dotc
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

    return aux



