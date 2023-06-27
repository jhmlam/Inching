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


import InchingLiteInt64.Fuel.CupysparseCsrInt64

cupy.random.seed(seed = 0)

import time
import sys

# NOTE This ios  the lanczos loop
def OOC2_FullMemS_v_KrylovAv_VOID(A):
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
        #print("A is csr")

    n = A.shape[0]
    v = cupy.empty((n,), dtype=A.dtype)

    #print(v)
    #outer_A = A

    def aux(A, v, u):
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

        # Matrix-vector multiplication
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
        _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)

        #print(u)
    return aux
            


# NOTE This ios  the lanczos loop
def OOC2_HalfMemS_v_KrylovAv_VOID(A):
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


    #outer_A = A

    def aux(A, v, u):
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

        #print(u.shape, A.diagonal(k=0).shape, v.shape)
        u -= cupy.multiply(A.diagonal(k=0) ,v)

        _cublas.setPointerMode(
            cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
        _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)
    return aux



# NOTE Krylov
def OOC2_HalfMemS_v_KrylovAv_VOIDInt64(A, A_diag):
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
    if InchingLiteInt64.Fuel.CupysparseCsrInt64.isspmatrix_csr(A) and cusparse.check_availability('spmv'):
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
    """
    import tqdm
    A_diag = cupy.empty((n,), dtype=A.dtype)
    for i in tqdm.tqdm(range(n)):
        for j in range(int(A.indptr[i]), int(A.indptr[i+1])):
            if A.indices[j] == i:
                A_diag[i] = A.data[j]
                break
    #outer_A = A
    print(A_diag)
    """
    def aux(A, v, u):
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

        #print(u.shape, A.diagonal(k=0).shape, v.shape)
        #u -= cupy.multiply(A.diagonal(k=0) ,v)
        u -= cupy.multiply(A_diag, v)

        _cublas.setPointerMode(
            cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
        _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)
    return aux


# NOTE This ios  the lanczos loop
def OOC2_FullMemS_v_KrylovAv_VOIDInt64(A):
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
    if InchingLiteInt64.Fuel.CupysparseCsrInt64.isspmatrix_csr(A) and cusparse.check_availability('spmv'):
        cusparse_handle = device.get_cusparse_handle()
        spmv_op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        spmv_alpha = numpy.array(1.0, A.dtype)
        spmv_beta = numpy.array(0.0, A.dtype)
        spmv_cuda_dtype = _dtype.to_cuda_dtype(A.dtype)
        spmv_alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT
        #print("A is csr")

    n = A.shape[0]
    v = cupy.empty((n,), dtype=A.dtype)

    #print(v)
    #outer_A = A

    def aux(A, v, u):
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

        # Matrix-vector multiplication
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
        _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)

        #print(u)
    return aux
            

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
