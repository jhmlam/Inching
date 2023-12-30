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




# =========================
# Hotelling MV
# ==========================

# NOTE This is for Hotelling during Mv 

def T3_QHotelling_x_Ax_HotelledAx(QHotelling, x, res, HotellingShift = 10.0):

    # ====================
    # Explained
    # ====================
    # NOTE Say, we have a mv product res =: Ax
    #      we want to do a hotelling without storing the deflated matrix
    #      So, (A- shift qq^T) x == Ax - shift q q^T x
    # res += User_HotellingShift*((User_Q_HotellingDeflation@cupy.ravel(x))[None,:]@User_Q_HotellingDeflation).flatten()

    if _csr.isspmatrix_csr(QHotelling):



        cublas_handle = device.get_cublas_handle()
        cublas_pointer_mode = _cublas.getPointerMode(cublas_handle)
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

        v_hotelling1 = cupy.empty((QHotelling.shape[0],), dtype=QHotelling.dtype)
        cusparse_handle = None
        if _csr.isspmatrix_csr(QHotelling) and cusparse.check_availability('spmv'):

            cusparse_handle_Hotelling = device.get_cusparse_handle()
            spmv_op_a_Hotelling1 = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
            spmv_op_a_Hotelling2 = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
            spmv_alpha_Hotelling1 = numpy.array(HotellingShift, QHotelling.dtype)
            spmv_alpha_Hotelling2 = numpy.array(1.0, QHotelling.dtype)
            
            spmv_beta_Hotelling1 = numpy.array(0.0, QHotelling.dtype)
            spmv_beta_Hotelling2 = numpy.array(1.0, QHotelling.dtype)
            spmv_cuda_dtype_Hotelling = _dtype.to_cuda_dtype(QHotelling.dtype)
            spmv_alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT




            spmv_desc_QHotelling = cusparse.SpMatDescriptor.create(QHotelling)
            spmv_desc_vhotelling1 = cusparse.DnVecDescriptor.create(v_hotelling1)


            if cusparse_handle_Hotelling is None:

                res += HotellingShift*((QHotelling@x)[None,:]@QHotelling).flatten()

            else:
                spmv_desc_QHotelling = cusparse.SpMatDescriptor.create(QHotelling)
                spmv_desc_v = cusparse.DnVecDescriptor.create(x)
                spmv_desc_u = cusparse.DnVecDescriptor.create(res)
                buff_size = _cusparse.spMV_bufferSize(
                    cusparse_handle_Hotelling, spmv_op_a_Hotelling1, 
                    spmv_alpha_Hotelling2.ctypes.data,
                    spmv_desc_QHotelling.desc, spmv_desc_v.desc, spmv_beta_Hotelling1.ctypes.data,
                    spmv_desc_vhotelling1.desc, spmv_cuda_dtype_Hotelling, spmv_alg)
                spmv_buff = cupy.empty(buff_size, cupy.int8)
                # NOTE self.hotellingshift * QX
                _cusparse.spMV(
                    cusparse_handle_Hotelling, spmv_op_a_Hotelling1, 
                    spmv_alpha_Hotelling1.ctypes.data,
                    spmv_desc_QHotelling.desc, 
                    spmv_desc_v.desc,
                    spmv_beta_Hotelling1.ctypes.data, spmv_desc_vhotelling1.desc,
                    spmv_cuda_dtype_Hotelling, spmv_alg, 
                    spmv_buff.data.ptr)


                # res = Q^T (shift * Q X) + res 
                #spmv_desc_u = cusparse.DnVecDescriptor.create(res)
                _cusparse.spMV(
                    cusparse_handle_Hotelling, spmv_op_a_Hotelling2, 
                    spmv_alpha_Hotelling2.ctypes.data,
                    spmv_desc_QHotelling.desc, 
                    spmv_desc_vhotelling1.desc,
                    spmv_beta_Hotelling2.ctypes.data, spmv_desc_u.desc,
                    spmv_cuda_dtype_Hotelling, spmv_alg, 
                    spmv_buff.data.ptr)

            _cublas.setPointerMode(
                cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)
            
    else:
        # TODO You know a faster way
        for i_hotel in range(QHotelling.shape[0]):
                res += cupy.ravel( QHotelling[i_hotel][:,None]@(
                            HotellingShift *(QHotelling[i_hotel][None,:]@x[:,None])))
    return res





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