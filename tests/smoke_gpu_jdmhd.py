"""GPU smoke test: build the ENM Hessian of a small structure with the installed
`inching` package and solve a few modes with the JDMHD solver, mirroring
Notebook/Application/01_Inching_EigenproblemJDMHD.ipynb.

Usage: python tests/smoke_gpu_jdmhd.py [structure.pdb|.cif] [n_modes]
Defaults to tests/data/1ubq.pdb (ubiquitin, 602 atoms) and 8 modes.
Not collected by pytest (no test_ prefix): it needs a CUDA GPU and takes a while.
Machine rule: never saturate all cores -> cKDTree workers are clamped to 7 and
BLAS/OMP thread counts are set to 7 before anything heavy is imported.
"""
import os, sys, time, gc
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "7")
MAX_WORKERS = 7

import numpy as np
import scipy.spatial

_Base = scipy.spatial.cKDTree
class _ClampedKDTree(_Base):
    def _clamp(self, kw):
        w = kw.get("workers", 1)
        if w is None or w < 0 or w > MAX_WORKERS:
            kw["workers"] = MAX_WORKERS
        return kw
    def query_ball_point(self, *a, **kw): return super().query_ball_point(*a, **self._clamp(kw))
    def query(self, *a, **kw): return super().query(*a, **self._clamp(kw))
    def query_ball_tree(self, *a, **kw): return super().query_ball_tree(*a, **kw)
scipy.spatial.cKDTree = _ClampedKDTree  # must happen before InchingLiteInteger imports it

import torch, cupy, cupyx
import cupy as cp
from cupy import cublas
from cupyx.scipy import sparse as cupysparse

import InchingLiteInteger.util
import InchingLiteInteger.Fuel.Coordinate.T1
import InchingLiteInteger.Burn.Krylov.T3
from InchingLiteInteger.Fuel.T1 import Xnumpy_SparseCupyMatrixUngappped
from InchingLiteInteger.Burn.JacobiDavidsonHotellingDeflation.T1 import S_HeigvalJDMHD_HeigvecJDMHD

assert InchingLiteInteger.Fuel.Coordinate.T1.cKDTree is _ClampedKDTree, "cKDTree clamp not in effect"

pdbfn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "1ubq.pdb")
n_mode = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rc_Gamma, maxleafsize, PlusI = 8.0, 100, 1.0

print("torch", torch.__version__, "cuda", torch.version.cuda, "| cupy", cupy.__version__,
      "| device", torch.cuda.get_device_name(0))
torch.manual_seed(0); cupy.random.seed(seed=0)
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.cuda.DoubleTensor)

t0 = time.time()
X_df, X_top = InchingLiteInteger.util.BasicPdbCifLoading(pdbfn)
xyz = X_df[["x", "y", "z"]].to_numpy().astype(np.float64)
n_atoms = xyz.shape[0]
print(f"loaded {pdbfn}: {n_atoms} atoms in {time.time()-t0:.2f}s")

order, undo = InchingLiteInteger.Fuel.Coordinate.T1.X_KdCuthillMckeeOrder(xyz, rc_Gamma=rc_Gamma, Reverse=True)
xyz = xyz[order, :]
NnzMinMaxDict, HalfNnz = InchingLiteInteger.Fuel.Coordinate.T1.X_KdUngappedMinMaxNeighbor(
    xyz, rc_Gamma=rc_Gamma, maxleafsize=maxleafsize, CollectStat=False, User_ReturnHalfNnz=True, SliceForm=True)
H = Xnumpy_SparseCupyMatrixUngappped(xyz, batch_head=None, maxleafsize=maxleafsize, rc_Gamma=rc_Gamma,
                                     User_PlusI=PlusI, User_DictCharmmGuiPbc=None, NnzMinMaxDict=NnzMinMaxDict)
A, A_diag = H.ReturnCupyHLowerTriangleInt64(User_MaxHalfNnzBufferSize=HalfNnz)
print(f"Hessian lower triangle: shape {A.shape}, nnz {A.nnz}, index dtype {A.indices.dtype}, built at {time.time()-t0:.2f}s")

# Rigid-body (free) modes for Hotelling deflation, as in the notebook
Q = cp.zeros((6, 3 * n_atoms), dtype=cp.float64)
for i in range(3):
    q = cp.zeros((n_atoms, 3)); q[:, i] = 1 / np.sqrt(n_atoms); Q[i, :] = q.flatten()
Xc = cp.array(xyz)
for j, R in enumerate([[[0,0,0],[0,0,-1],[0,1,0]], [[0,0,1],[0,0,0],[-1,0,0]], [[0,-1,0],[1,0,0],[0,0,0]]]):
    g = (Xc @ cp.array(R, dtype=cp.float64).T).flatten(); Q[3 + j, :] = g / cp.linalg.norm(g)
for _ in range(2):
    V = Q.T
    for ix in range(1, 6):
        for _ in range(2):
            V[:, ix] -= V[:, :ix] @ (V[:, :ix].T @ V[:, ix]); V[:, ix] /= cp.sqrt(V[:, ix] @ V[:, ix])
    Q = V.T
Q = cupysparse.csr_matrix(Q, dtype=cp.float64)

t1 = time.time()
eigval, eigvec = S_HeigvalJDMHD_HeigvecJDMHD(A, A_diag, k=n_mode, tol=1e-12, maxiter=15000,
    User_CorrectionSolverMaxiter=20, User_HalfMemMode=True, User_IntermediateConvergenceTol=1e-3,
    User_GapEstimate=None, User_FactoringToleranceOnCorrection=1e-4,
    User_Q_HotellingDeflation=Q, User_HotellingShift=40)
print(f"JDMHD solved {n_mode} modes in {time.time()-t1:.2f}s")

# Residual check ||A v - lambda v||, exactly as the notebook's PART05
KrylovAv = InchingLiteInteger.Burn.Krylov.T3.OOC2_HalfMemS_v_KrylovAv_VOID(A, A_diag)
Av = cupy.empty((n_atoms * 3,)).astype(A.dtype)
worst = 0.0
for j in range(n_mode):
    KrylovAv(A, cupy.ravel(eigvec[:, j]), Av)
    r = float(cublas.nrm2(Av - eigval[j] * cupy.ravel(eigvec[:, j])))
    worst = max(worst, r)
    print(f"  mode {j}: lambda = {float(eigval[j]) - PlusI: .6e}   residual = {r:.2e}")
print(f"max residual {worst:.2e}; total {time.time()-t0:.2f}s")
assert worst < 1e-6, "eigenpair residual too large"
assert float(eigval[0]) - PlusI > 1e-8, "first non-rigid eigenvalue should be positive (free modes deflated)"
print("SMOKE TEST PASSED")
