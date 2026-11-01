import argparse
import contextlib
import io
import json
import os
import time
import numpy as np

SCALE = {("pdb", "pdb"): 1.0, ("pdb", "cif"): 0.1, ("cif", "pdb"): 10.0, ("cif", "cif"): 1.0}


def load(path):
    from inching.util import BasicPdbCifLoading
    df, _ = BasicPdbCifLoading(path)
    return df[["x", "y", "z"]].to_numpy().astype(np.float64)


def rigid_modes(xyz):
    import cupy as cp
    import cupyx.scipy.sparse as cupysparse
    n = xyz.shape[0]
    Q = cp.zeros((6, 3 * n))
    for i in range(3):
        q = cp.zeros((n, 3))
        q[:, i] = 1 / np.sqrt(n)
        Q[i] = q.ravel()
    X = cp.asarray(xyz)
    for i, R in enumerate([[[0, 0, 0], [0, 0, -1], [0, 1, 0]], [[0, 0, 1], [0, 0, 0], [-1, 0, 0]], [[0, -1, 0], [1, 0, 0], [0, 0, 0]]]):
        g = (X @ cp.asarray(R, dtype=cp.float64).T).ravel()
        Q[3 + i] = g / cp.linalg.norm(g)
    V = Q.T
    for _ in range(2):
        for ix in range(1, 6):
            for _ in range(2):
                V[:, ix] -= V[:, :ix] @ (V[:, :ix].T @ V[:, ix])
                V[:, ix] /= cp.sqrt(V[:, ix] @ V[:, ix])
    return cupysparse.csr_matrix(V.T)


def run_nma(xyz, k=64, solver="trlmhd", rc=8.0, maxleafsize=100, int64=False, deflation=True, tol=1e-12, maxiter=15000):
    import cupy as cp
    import torch
    from cupy import cublas
    import inching.Fuel.Coordinate.T1 as CT1
    import inching.Burn.Krylov.T3 as KT3
    from inching.Fuel.T1 import Xnumpy_SparseCupyMatrixUngappped
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cuda")
    cp.random.seed(0)
    xyz = xyz - xyz.mean(axis=0)
    n = xyz.shape[0]
    order, undo = CT1.X_KdCuthillMckeeOrder(xyz, rc_Gamma=rc, Reverse=True)
    xyz = xyz[order]
    nnz_dict, half_nnz = CT1.X_KdUngappedMinMaxNeighbor(xyz, rc_Gamma=rc, maxleafsize=maxleafsize, User_ReturnHalfNnz=True)
    H = Xnumpy_SparseCupyMatrixUngappped(xyz, maxleafsize=maxleafsize, rc_Gamma=rc, User_PlusI=1.0, NnzMinMaxDict=nnz_dict)
    build = H.ReturnCupyHLowerTriangleInt64 if int64 else H.ReturnCupyHLowerTriangleInt32
    A, A_diag = build(User_MaxHalfNnzBufferSize=half_nnz)
    Q = rigid_modes(xyz) if deflation else None
    if solver == "jdmhd":
        from inching.Burn.JacobiDavidsonHotellingDeflation.T1 import S_HeigvalJDMHD_HeigvecJDMHD
        eigval, eigvec = S_HeigvalJDMHD_HeigvecJDMHD(A, A_diag, k=k, tol=tol, maxiter=maxiter, User_GapEstimate=1e-6,
                                                     User_Q_HotellingDeflation=Q, User_HotellingShift=40)
    elif solver == "trlmhd":
        from inching.Burn.ThickRestartLanczosHotellingDeflation.T1 import S_HeigvalTRLMHD_HeigvecTRLMHD
        eigval, eigvec = S_HeigvalTRLMHD_HeigvecTRLMHD(A, A_diag, k=k, tol=tol, maxiter=maxiter,
                                                       User_Q_HotellingDeflation=Q, User_HotellingShift=10)
    else:
        from inching.Burn.ImplicitlyRestartedLanczosHotellingDeflation.T1 import S_HeigvalIRLMHD_HeigvecIRLMHD
        eigval, eigvec = S_HeigvalIRLMHD_HeigvecIRLMHD(A, A_diag, k=k, tol=tol, maxiter=maxiter,
                                                       User_Q_HotellingDeflation=Q, User_HotellingShift=10)
    Av = KT3.OOC2_HalfMemS_v_KrylovAv_VOID(A, A_diag)
    u = cp.empty((3 * n,), dtype=A.dtype)
    residual = []
    for j in range(eigvec.shape[1]):
        Av(A, cp.ravel(eigvec[:, j]), u)
        residual.append(float(cublas.nrm2(u - eigval[j] * cp.ravel(eigvec[:, j]))))
    m = eigvec.shape[1]
    return (cp.asnumpy(eigval) - 1.0, cp.asnumpy(eigvec).T.reshape(m, n, 3)[:, undo, :], residual,
            cp.get_default_memory_pool().total_bytes())


def animate(path, eigvec, modes, outdir, n_frames=7, amplitude=2.0, fmt="cif"):
    from inching.util import BasicPdbCifWriting
    from inching.Fuel.Coordinate.T1 import HeigvecOne_BoxCoxMagnitude
    ext = path.rsplit(".", 1)[-1]
    stem = os.path.basename(path).rsplit(".", 1)[0]
    xyz = load(path)
    t = np.linspace(-1.0, 1.0, n_frames)[:, None, None]
    out = []
    for m in modes:
        v = eigvec[m]
        mag = np.clip(HeigvecOne_BoxCoxMagnitude(v), 0.1, 1.0)
        nrm = np.linalg.norm(v, axis=1)[:, None]
        d = mag[:, None] * np.divide(v, nrm, out=np.zeros_like(v), where=nrm > 0) * (amplitude / 10.0)
        fn = os.path.join(outdir, "%s_mode%d.%s" % (stem, m, fmt))
        BasicPdbCifWriting(ref_structure_dir=path, save_structure_dir=fn, position=(xyz + t * d) * SCALE[(ext, fmt)],
                           keepIds=True, SaveFormat=fmt, User_Bfactor=mag)
        out.append(fn)
    return out


def info(xyz, rc=8.0, maxleafsize=100, k=64):
    from scipy.spatial import cKDTree
    import inching.Fuel.Coordinate.T1 as CT1
    n = xyz.shape[0]
    N = 3 * n
    degree = cKDTree(xyz).query_ball_point(xyz, rc / 10.0, workers=-1, return_length=True)
    Nij = int(degree.sum())
    buf = int(((Nij - n) / 2 + n) * 9)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        order, _ = CT1.X_KdCuthillMckeeOrder(xyz, rc_Gamma=rc, Reverse=True)
        bands = CT1.X_KdUngappedMinMaxNeighbor(xyz[order], rc_Gamma=rc, maxleafsize=maxleafsize)
    bw = max(e - s for v in bands.values() for s, e in v)
    gb = 1024 ** 3
    ncv = min(max(2 * k, k + 32), N - 1)
    return {
        "n_atoms": n, "dof": N,
        "neighbours_mean": float(degree.mean()), "neighbours_std": float(degree.std()),
        "nnz_lower_est": 9 * (Nij - n) // 2 + 6 * n, "nnz_lower_buffer": buf, "nnz_full_est": 9 * Nij,
        "int32_ok": buf < 2**31 - 1,
        "bandwidth_atoms_max": int(bw), "n_batches": len(bands),
        "gb_hessian_int32": (12 * buf + 12 * N + 24 * n) / gb, "gb_hessian_int64": (16 * buf + 16 * N + 24 * n) / gb,
        "gb_modes_jdmhd": 8 * N * (3 * k + 40) / gb, "gb_modes_trlmhd": 8 * N * (ncv + 4) / gb,
    }


def parse_modes(s, n):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return [m for m in out if m < n]


def main(argv=None):
    p = argparse.ArgumentParser(prog="inching", description="GPU normal mode analysis of large macromolecular assemblies")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("nma", help="compute the lowest vibrational modes")
    q.add_argument("structure", help="pdb or cif file")
    q.add_argument("-k", type=int, default=64, help="number of modes (irlmhd needs >= 32)")
    q.add_argument("--solver", choices=["trlmhd", "jdmhd", "irlmhd"], default="trlmhd")
    q.add_argument("--rc", type=float, default=8.0, help="cutoff in angstrom")
    q.add_argument("--maxleafsize", type=int, default=100)
    q.add_argument("--int64", action="store_true", help="int64 indexing (nnz > 2^31)")
    q.add_argument("--no-deflation", action="store_true", help="keep the 6 rigid-body modes")
    q.add_argument("--tol", type=float, default=1e-12)
    q.add_argument("--maxiter", type=int, default=15000)
    q.add_argument("-o", "--outdir", default=".")
    q = sub.add_parser("animate", help="write per-mode multi-model trajectories from an nma result")
    q.add_argument("structure")
    q.add_argument("eigvec", help="<id>_eigvec.npy from nma")
    q.add_argument("--modes", default="0-5", help="e.g. 0-5 or 0,3,7")
    q.add_argument("--frames", type=int, default=7)
    q.add_argument("--amplitude", type=float, default=2.0, help="max displacement in angstrom")
    q.add_argument("--fmt", choices=["cif", "pdb"], default="cif")
    q.add_argument("-o", "--outdir", default=".")
    q = sub.add_parser("info", help="size, sparsity and memory estimate without solving (CPU only)")
    q.add_argument("structure")
    q.add_argument("--rc", type=float, default=8.0)
    q.add_argument("--maxleafsize", type=int, default=100)
    q.add_argument("-k", type=int, default=64)
    a = p.parse_args(argv)

    if a.cmd == "info":
        print(json.dumps(info(load(a.structure), a.rc, a.maxleafsize, a.k), indent=1))
        return
    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.basename(a.structure).rsplit(".", 1)[0]
    if a.cmd == "nma":
        t0 = time.time()
        eigval, eigvec, residual, gpu_bytes = run_nma(load(a.structure), a.k, a.solver, a.rc, a.maxleafsize, a.int64,
                                                      not a.no_deflation, a.tol, a.maxiter)
        np.save(os.path.join(a.outdir, stem + "_eigval.npy"), eigval)
        np.save(os.path.join(a.outdir, stem + "_eigvec.npy"), eigvec)
        meta = dict(vars(a), n_atoms=eigvec.shape[1], n_modes=len(eigval), eigval=eigval.tolist(), residual=residual,
                    runtime_s=time.time() - t0, gpu_pool_gb=gpu_bytes / 1024**3)
        with open(os.path.join(a.outdir, stem + "_nma.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print("%d modes, max residual %.1e, %.1f s" % (len(eigval), max(residual), meta["runtime_s"]))
    else:
        eigvec = np.load(a.eigvec)
        for fn in animate(a.structure, eigvec, parse_modes(a.modes, eigvec.shape[0]), a.outdir, a.frames, a.amplitude, a.fmt):
            print(fn)


if __name__ == "__main__":
    main()
