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

# NOTE I copied this from https://github.com/cupy/cupy/blob/main/cupyx/scipy/sparse/_csr.py
#      and removed utils that I don't call to isolate the problem. Only applies to the final big matrix

import operator
import warnings

import numpy

try:
    import scipy.sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False

import cupy
from cupy._core import _accelerator
from cupy.cuda import cub
from cupy.cuda import runtime
from cupy import cusparse
from cupyx.scipy.sparse import _base
from .CupysparseCompressInt64 import _compressed_sparse_matrix

from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import SparseEfficiencyWarning
from cupyx.scipy.sparse import _util


class csr_matrix(_compressed_sparse_matrix):

    """Compressed Sparse Row matrix.

    This can be instantiated in several ways.

    ``csr_matrix(D)``
        ``D`` is a rank-2 :class:`cupy.ndarray`.
    ``csr_matrix(S)``
        ``S`` is another sparse matrix. It is equivalent to ``S.tocsr()``.
    ``csr_matrix((M, N), [dtype])``
        It constructs an empty matrix whose shape is ``(M, N)``. Default dtype
        is float64.
    ``csr_matrix((data, (row, col)))``
        All ``data``, ``row`` and ``col`` are one-dimenaional
        :class:`cupy.ndarray`.
    ``csr_matrix((data, indices, indptr))``
        All ``data``, ``indices`` and ``indptr`` are one-dimenaional
        :class:`cupy.ndarray`.

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape of a matrix. Its length must be two.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        copy (bool): If ``True``, copies of given arrays are always used.

    .. seealso::
        :class:`scipy.sparse.csr_matrix`

    """

    format = 'csr'

    def get(self, stream=None):
        """Returns a copy of the array on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        Returns:
            scipy.sparse.csr_matrix: Copy of the array on host memory.

        """
        if not _scipy_available:
            raise RuntimeError('scipy is not available')
        data = self.data.get(stream)
        indices = self.indices.get(stream)
        indptr = self.indptr.get(stream)
        return scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)


    def _swap(self, x, y):
        return (x, y)

    def _add_sparse(self, other, alpha, beta):
        self.sum_duplicates()
        other = other.tocsr()
        other.sum_duplicates()
        if cusparse.check_availability('csrgeam2'):
            csrgeam = cusparse.csrgeam2
        elif cusparse.check_availability('csrgeam'):
            csrgeam = cusparse.csrgeam
        else:
            raise NotImplementedError
        return csrgeam(self, other, alpha, beta)


    def __mul__(self, other):
        if cupy.isscalar(other):
            self.sum_duplicates()
            return self._with_data(self.data * other)
        elif isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if cusparse.check_availability('spgemm'):
                return cusparse.spgemm(self, other)
            elif cusparse.check_availability('csrgemm2'):
                return cusparse.csrgemm2(self, other)
            elif cusparse.check_availability('csrgemm'):
                return cusparse.csrgemm(self, other)
            else:
                raise NotImplementedError
        elif _csc.isspmatrix_csc(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if cusparse.check_availability('csrgemm') and not runtime.is_hip:
                # trans=True is still buggy as of ROCm 4.2.0
                return cusparse.csrgemm(self, other.T, transb=True)
            elif cusparse.check_availability('spgemm'):
                b = other.tocsr()
                b.sum_duplicates()
                return cusparse.spgemm(self, b)
            elif cusparse.check_availability('csrgemm2'):
                b = other.tocsr()
                b.sum_duplicates()
                return cusparse.csrgemm2(self, b)
            else:
                raise NotImplementedError
        elif _base.isspmatrix(other):
            return self * other.tocsr()
        elif _base.isdense(other):
            if other.ndim == 0:
                self.sum_duplicates()
                return self._with_data(self.data * other)
            elif other.ndim == 1:
                self.sum_duplicates()
                other = cupy.asfortranarray(other)
                # need extra padding to ensure not stepping on the CUB bug,
                # see cupy/cupy#3679 for discussion
                is_cub_safe = (self.indptr.data.mem.size
                               > self.indptr.size * self.indptr.dtype.itemsize)
                # CUB spmv is buggy since CUDA 11.0, see
                # https://github.com/cupy/cupy/issues/3822#issuecomment-782607637
                is_cub_safe &= (cub._get_cuda_build_version() < 11000)
                for accelerator in _accelerator.get_routine_accelerators():
                    if (accelerator == _accelerator.ACCELERATOR_CUB
                            and not runtime.is_hip
                            and is_cub_safe and other.flags.c_contiguous):
                        return cub.device_csrmv(
                            self.shape[0], self.shape[1], self.nnz,
                            self.data, self.indptr, self.indices, other)
                if (cusparse.check_availability('csrmvEx') and self.nnz > 0 and
                        cusparse.csrmvExIsAligned(self, other)):
                    # csrmvEx does not work if nnz == 0
                    csrmv = cusparse.csrmvEx
                elif cusparse.check_availability('csrmv'):
                    csrmv = cusparse.csrmv
                elif cusparse.check_availability('spmv'):
                    csrmv = cusparse.spmv
                else:
                    raise NotImplementedError
                return csrmv(self, other)
            elif other.ndim == 2:
                self.sum_duplicates()
                if cusparse.check_availability('csrmm2'):
                    csrmm = cusparse.csrmm2
                elif cusparse.check_availability('spmm'):
                    csrmm = cusparse.spmm
                else:
                    raise NotImplementedError
                return csrmm(self, cupy.asfortranarray(other))
            else:
                raise ValueError('could not interpret dimensions')
        else:
            return NotImplemented

    def __truediv__(self, other):
        """Point-wise division by another matrix, vector or scalar"""
        if _util.isscalarlike(other):
            dtype = self.dtype
            if dtype == numpy.float32:
                # Note: This is a work-around to make the output dtype the same
                # as SciPy. It might be SciPy version dependent.
                dtype = numpy.float64
            dtype = cupy.result_type(dtype, other)
            d = cupy.reciprocal(other, dtype=dtype)
            return multiply_by_scalar(self, d)
        elif _util.isdense(other):
            other = cupy.atleast_2d(other)
            check_shape_for_pointwise_op(self.shape, other.shape)
            return self.todense() / other
        elif _base.isspmatrix(other):
            # Note: If broadcasting is needed, an exception is raised here for
            # compatibility with SciPy, as SciPy does not support broadcasting
            # in the "sparse / sparse" case.
            check_shape_for_pointwise_op(self.shape, other.shape,
                                         allow_broadcasting=False)
            dtype = numpy.promote_types(self.dtype, other.dtype)
            if dtype.char not in 'FD':
                dtype = numpy.promote_types(numpy.float64, dtype)
            # Note: The following implementation converts two sparse matrices
            # into dense matrices and then performs a point-wise division,
            # which can use lots of memory.
            self_dense = self.todense().astype(dtype, copy=False)
            return self_dense / other.todense()
        raise NotImplementedError

    # TODO(unno): Implement check_format

    def diagonal(self, k=0):
        assert k == 0, "ABORTED. Currently only supprt"
        rows, cols = self.shape
        ylen = min(rows + min(k, 0), cols - max(k, 0))
        if ylen <= 0:
            return cupy.empty(0, dtype=self.dtype)
        self.sum_duplicates()
        y = cupy.empty(ylen, dtype=self.dtype)
        _cupy_csr_diagonal()(k, rows, cols, self.data, self.indptr,
                             self.indices, y)
        
        return y

    def eliminate_zeros(self):
        """Removes zero entories in place."""
        compress = cusparse.csr2csr_compress(self, 0)
        self.data = compress.data
        self.indices = compress.indices
        self.indptr = compress.indptr

    def sort_indices(self):
        """Sorts the indices of this matrix *in place*.

        .. warning::
            Calling this function might synchronize the device.

        """
        if not self.has_sorted_indices:
            cusparse.csrsort(self)
            self.has_sorted_indices = True


    def transpose(self, axes=None, copy=False):
        """Returns a transpose matrix.

        Args:
            axes: This option is not supported.
            copy (bool): If ``True``, a returned matrix shares no data.
                Otherwise, it shared data arrays as much as possible.

        Returns:
            cupyx.scipy.sparse.spmatrix: Transpose matrix.

        """
        if axes is not None:
            raise ValueError(
                'Sparse matrices do not support an \'axes\' parameter because '
                'swapping dimensions is the only logical permutation.')

        shape = self.shape[1], self.shape[0]
        trans = _csc.csc_matrix(
            (self.data, self.indices, self.indptr), shape=shape, copy=copy)
        trans.has_canonical_format = self.has_canonical_format
        return trans

    def getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).

        Args:
            i (integer): Row

        Returns:
            cupyx.scipy.sparse.csr_matrix: Sparse matrix with single row
        """
        return self._major_slice(slice(i, i + 1), copy=True)

    def getcol(self, i):
        """Returns a copy of column i of the matrix, as a (m x 1)
        CSR matrix (column vector).

        Args:
            i (integer): Column

        Returns:
            cupyx.scipy.sparse.csr_matrix: Sparse matrix with single column
        """
        return self._minor_slice(slice(i, i + 1), copy=True)

    def _get_intXarray(self, row, col):
        row = slice(row, row + 1)
        return self._major_slice(row)._minor_index_fancy(col)

    def _get_intXslice(self, row, col):
        row = slice(row, row + 1)
        return self._major_slice(row)._minor_slice(col, copy=True)

    def _get_sliceXint(self, row, col):
        col = slice(col, col + 1)
        copy = row.step in (1, None)
        return self._major_slice(row)._minor_slice(col, copy=copy)

    def _get_sliceXarray(self, row, col):
        return self._major_slice(row)._minor_index_fancy(col)

    def _get_arrayXint(self, row, col):
        col = slice(col, col + 1)
        return self._major_index_fancy(row)._minor_slice(col)

    def _get_arrayXslice(self, row, col):
        if col.step not in (1, None):
            start, stop, step = col.indices(self.shape[1])
            cols = cupy.arange(start, stop, step, self.indices.dtype)
            return self._get_arrayXarray(row, cols)
        return self._major_index_fancy(row)._minor_slice(col)


def isspmatrix_csr(x):
    """Checks if a given matrix is of CSR format.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.csr_matrix`.

    """
    return isinstance(x, csr_matrix)


def check_shape_for_pointwise_op(a_shape, b_shape, allow_broadcasting=True):
    if allow_broadcasting:
        a_m, a_n = a_shape
        b_m, b_n = b_shape
        if not (a_m == b_m or a_m == 1 or b_m == 1):
            raise ValueError('inconsistent shape')
        if not (a_n == b_n or a_n == 1 or b_n == 1):
            raise ValueError('inconsistent shape')
    else:
        if a_shape != b_shape:
            raise ValueError('inconsistent shape')


def multiply_by_scalar(sp, a):
    data = sp.data * a
    indices = sp.indices.copy()
    indptr = sp.indptr.copy()
    return csr_matrix((data, indices, indptr), shape=sp.shape)


def multiply_by_dense(sp, dn):
    check_shape_for_pointwise_op(sp.shape, dn.shape)
    sp_m, sp_n = sp.shape
    dn_m, dn_n = dn.shape
    m, n = max(sp_m, dn_m), max(sp_n, dn_n)
    nnz = sp.nnz * (m // sp_m) * (n // sp_n)
    dtype = numpy.promote_types(sp.dtype, dn.dtype)
    data = cupy.empty(nnz, dtype=dtype)
    indices = cupy.empty(nnz, dtype=sp.indices.dtype)
    if m > sp_m:
        if n > sp_n:
            indptr = cupy.arange(0, nnz+1, n, dtype=sp.indptr.dtype)
        else:
            indptr = cupy.arange(0, nnz+1, sp.nnz, dtype=sp.indptr.dtype)
    else:
        indptr = sp.indptr.copy()
        if n > sp_n:
            indptr *= n

    # out = sp * dn
    cupy_multiply_by_dense()(sp.data, sp.indptr, sp.indices, sp_m, sp_n,
                             dn, dn_m, dn_n, indptr, m, n, data, indices)

    return csr_matrix((data, indices, indptr), shape=(m, n))


_GET_ROW_ID_ = '''
__device__ inline int get_row_id(int i, int min, int max, const int *indptr) {
    int row = (min + max) / 2;
    while (min < max) {
        if (i < indptr[row]) {
            max = row - 1;
        } else if (i >= indptr[row + 1]) {
            min = row + 1;
        } else {
            break;
        }
        row = (min + max) / 2;
    }
    return row;
}
'''

_FIND_INDEX_HOLDING_COL_IN_ROW_ = '''
__device__ inline int find_index_holding_col_in_row(
        int row, int col, const int64 *indptr, const int64 *indices) {
    int64 j_min = indptr[row];
    int64 j_max = indptr[row+1] - 1;
    while (j_min <= j_max) {
        int j = (j_min + j_max) / 2;
        int j_col = indices[j];
        if (j_col == col) {
            return j;
        } else if (j_col < col) {
            j_min = j + 1;
        } else {
            j_max = j - 1;
        }
    }
    return -1;
}
'''


@cupy._util.memoize(for_each_device=True)
def cupy_multiply_by_dense():
    return cupy.ElementwiseKernel(
        '''
        raw S SP_DATA, raw I SP_INDPTR, raw I SP_INDICES,
        int32 SP_M, int32 SP_N,
        raw D DN_DATA, int32 DN_M, int32 DN_N,
        raw I OUT_INDPTR, int32 OUT_M, int32 OUT_N
        ''',
        'O OUT_DATA, I OUT_INDICES',
        '''
        int i_out = i;
        int m_out = get_row_id(i_out, 0, OUT_M - 1, &(OUT_INDPTR[0]));
        int i_sp = i_out;
        if (OUT_M > SP_M && SP_M == 1) {
            i_sp -= OUT_INDPTR[m_out];
        }
        if (OUT_N > SP_N && SP_N == 1) {
            i_sp /= OUT_N;
        }
        int n_out = SP_INDICES[i_sp];
        if (OUT_N > SP_N && SP_N == 1) {
            n_out = i_out - OUT_INDPTR[m_out];
        }
        int m_dn = m_out;
        if (OUT_M > DN_M && DN_M == 1) {
            m_dn = 0;
        }
        int n_dn = n_out;
        if (OUT_N > DN_N && DN_N == 1) {
            n_dn = 0;
        }
        OUT_DATA = (O)(SP_DATA[i_sp] * DN_DATA[n_dn + (DN_N * m_dn)]);
        OUT_INDICES = n_out;
        ''',
        'cupyx_scipy_sparse_csr_multiply_by_dense',
        preamble=_GET_ROW_ID_
    )


def multiply_by_csr(a, b):
    check_shape_for_pointwise_op(a.shape, b.shape)
    a_m, a_n = a.shape
    b_m, b_n = b.shape
    m, n = max(a_m, b_m), max(a_n, b_n)
    a_nnz = a.nnz * (m // a_m) * (n // a_n)
    b_nnz = b.nnz * (m // b_m) * (n // b_n)
    if a_nnz > b_nnz:
        return multiply_by_csr(b, a)
    c_nnz = a_nnz
    dtype = numpy.promote_types(a.dtype, b.dtype)
    c_data = cupy.empty(c_nnz, dtype=dtype)
    c_indices = cupy.empty(c_nnz, dtype=a.indices.dtype)
    if m > a_m:
        if n > a_n:
            c_indptr = cupy.arange(0, c_nnz+1, n, dtype=a.indptr.dtype)
        else:
            c_indptr = cupy.arange(0, c_nnz+1, a.nnz, dtype=a.indptr.dtype)
    else:
        c_indptr = a.indptr.copy()
        if n > a_n:
            c_indptr *= n
    flags = cupy.zeros(c_nnz+1, dtype=a.indices.dtype)
    nnz_each_row = cupy.zeros(m+1, dtype=a.indptr.dtype)

    # compute c = a * b where necessary and get sparsity pattern of matrix d
    cupy_multiply_by_csr_step1()(
        a.data, a.indptr, a.indices, a_m, a_n,
        b.data, b.indptr, b.indices, b_m, b_n,
        c_indptr, m, n, c_data, c_indices, flags, nnz_each_row)

    flags = cupy.cumsum(flags, dtype=a.indptr.dtype)
    d_indptr = cupy.cumsum(nnz_each_row, dtype=a.indptr.dtype)
    d_nnz = int(d_indptr[-1])
    d_data = cupy.empty(d_nnz, dtype=dtype)
    d_indices = cupy.empty(d_nnz, dtype=a.indices.dtype)

    # remove zero elements in matric c
    cupy_multiply_by_csr_step2()(c_data, c_indices, flags, d_data, d_indices)

    return csr_matrix((d_data, d_indices, d_indptr), shape=(m, n))


@cupy._util.memoize(for_each_device=True)
def cupy_multiply_by_csr_step1():
    return cupy.ElementwiseKernel(
        '''
        raw A A_DATA, raw I A_INDPTR, raw I A_INDICES, int32 A_M, int32 A_N,
        raw B B_DATA, raw I B_INDPTR, raw I B_INDICES, int32 B_M, int32 B_N,
        raw I C_INDPTR, int32 C_M, int32 C_N
        ''',
        'C C_DATA, I C_INDICES, raw I FLAGS, raw I NNZ_EACH_ROW',
        '''
        int i_c = i;
        int m_c = get_row_id(i_c, 0, C_M - 1, &(C_INDPTR[0]));

        int i_a = i;
        if (C_M > A_M && A_M == 1) {
            i_a -= C_INDPTR[m_c];
        }
        if (C_N > A_N && A_N == 1) {
            i_a /= C_N;
        }
        int n_c = A_INDICES[i_a];
        if (C_N > A_N && A_N == 1) {
            n_c = i % C_N;
        }
        int m_b = m_c;
        if (C_M > B_M && B_M == 1) {
            m_b = 0;
        }
        int n_b = n_c;
        if (C_N > B_N && B_N == 1) {
            n_b = 0;
        }
        int i_b = find_index_holding_col_in_row(m_b, n_b,
            &(B_INDPTR[0]), &(B_INDICES[0]));
        if (i_b >= 0) {
            atomicAdd(&(NNZ_EACH_ROW[m_c+1]), 1);
            FLAGS[i+1] = 1;
            C_DATA = (C)(A_DATA[i_a] * B_DATA[i_b]);
            C_INDICES = n_c;
        }
        ''',
        'cupyx_scipy_sparse_csr_multiply_by_csr_step1',
        preamble=_GET_ROW_ID_ + _FIND_INDEX_HOLDING_COL_IN_ROW_
    )


@cupy._util.memoize(for_each_device=True)
def cupy_multiply_by_csr_step2():
    return cupy.ElementwiseKernel(
        'T C_DATA, I C_INDICES, raw I FLAGS',
        'raw D D_DATA, raw I D_INDICES',
        '''
        int j = FLAGS[i];
        if (j < FLAGS[i+1]) {
            D_DATA[j] = (D)(C_DATA);
            D_INDICES[j] = C_INDICES;
        }
        ''',
        'cupyx_scipy_sparse_csr_multiply_by_csr_step2'
    )



# NOTE This does not work and complain about pointer datatype? TODO
@cupy._util.memoize(for_each_device=True)
def _cupy_csr_diagonal():
    return cupy.ElementwiseKernel(
        'int64 k, int64 rows, int64 cols, '
        'raw T data, raw int64 indptr, raw int64 indices',
        'T y',
        '''
        int64 row = i;
        int64 col = i;
        if (k < 0) row -= k;
        if (k > 0) col += k;
        if (row >= rows || col >= cols) return;
        int j = find_index_holding_col_in_row(row, col,
            &(indptr[0]), &(indices[0]));
        if (j >= 0) {
            y = data[j];
        } else {
            y = static_cast<T>(0);
        }
        ''',
        'cupyx_scipy_sparse_csr_diagonal',
        preamble=_FIND_INDEX_HOLDING_COL_IN_ROW_, 
    )



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
