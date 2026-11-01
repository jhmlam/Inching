# Inching - Scalable Computation for Anisotropic Vibrations of Large Macromolecular Assemblies
[![Click to try on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhmlam/Inching/blob/main/GoogleColab/GoogleColab_Inching_v1.1.ipynb)
## Abstract
The Normal Mode Analysis (NMA) is a standard approach to elucidate the anisotropic vibrations of macromolecules at their folded states, where low-frequency collective motions can reveal rearrangements of domains and changes in the exposed surface of macromolecules. Recent advances in structural biology have enabled the resolution of megascale macromolecules with millions of atoms. However, the calculation of their vibrational modes remains elusive due to the prohibitive cost associated with constructing and diagonalizing the underlying eigenproblem and the current approaches to NMA are not readily adaptable for efficient parallel computing on graphic processing unit (GPU). Here, we present eigenproblem construction and diagonalization approach that implements level-structure bandwidth-reducing algorithms to transform the sparse computation in NMA to a globally-sparse-yet-locally-dense computation, allowing batched tensor products to be most efficiently executed on GPU. We mapped, optimized, and compared several low-complexity Krylov-subspace eigensolvers, supplemented by techniques such as Chebyshev filtering, sum decomposition, external explicit deflation and shift-and-inverse, to allow fast GPU-resident calculations. The method allows accurate calculation of the first 1000 vibrational modes of some largest structures in PDB (> 2.4 million atoms) at least 250 times faster than existing methods.


## How to cite us?
This is a codebase for the INCHING algorithm used in the following paper

> Lam, J.H., Nakano, A. & Katritch, V. Scalable computation of anisotropic vibrations for large macromolecular assemblies. Nat Commun 15, 3479 (2024). https://doi.org/10.1038/s41467-024-47685-8

Here's a Google Colab notebook running on a T4 GPU for free. For performant computing, please go to `Notebook`, install the software locally and use it on a fair-standing GPU e.g. RTX4090 NVIDIA with 24 GB.

[![Click to try on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhmlam/Inching/blob/main/GoogleColab/GoogleColab_Inching_v1.1.ipynb)





## Install and use (v1.1+)

Requirements: Linux, an NVIDIA GPU, Python >= 3.11.

Option A, conda (recommended; everything incl. CUDA libraries comes from conda-forge, `inching` itself from the `jhmlam` channel):
```
conda create -n inching -c conda-forge -c jhmlam inching
conda activate inching
```
Activate the environment before running (CuPy compiles its kernels with the CUDA headers of the environment).

Option B, pip, on top of an existing CUDA 12 PyTorch install (what Google Colab does):
```
pip install openmm mdtraj cupy-cuda12x torch numpy scipy pandas matplotlib seaborn plotly tqdm
pip install --no-deps git+https://github.com/jhmlam/Inching.git@v1.1
```

Example on ubiquitin (PDB 1UBQ, 660 atoms):
```
wget https://files.rcsb.org/download/1UBQ.pdb
inching info 1UBQ.pdb -k 16            # CPU only: atoms, neighbours, Hessian nnz, int32 ok?, GPU memory estimate
inching nma  1UBQ.pdb -k 16 -o out     # lowest 16 non-rigid modes (default solver trlmhd; --solver jdmhd|irlmhd)
inching animate 1UBQ.pdb out/1UBQ_eigvec.npy --modes 0-5 --amplitude 2 -o out
```
`inching nma` writes `out/1UBQ_eigval.npy` (k,), `out/1UBQ_eigvec.npy` (k, n_atoms, 3) in file atom order and
`out/1UBQ_nma.json` (parameters, eigenvalues, residuals, runtime, GPU memory). `inching animate` writes one
multi-model `1UBQ_mode<m>.cif` per mode (7 frames, max displacement 2 Å, B-factor = displacement magnitude);
open in PyMOL/ChimeraX as a trajectory and `spectrum b`. Options: `--rc` cutoff in Å (8), `--int64` for
> 2^31 nonzeros, `--no-deflation` to keep the six rigid-body modes, `--tol`, `--maxiter`, `--fmt pdb`.

The same functions from Python:
```python
from inching.cli import load, info, run_nma, animate
xyz = load("1UBQ.pdb")
eigval, eigvec, residual, gpu_bytes = run_nma(xyz, k=16, solver="trlmhd")
animate("1UBQ.pdb", eigvec, modes=range(6), outdir="out")
```
Google Colab (free T4): `GoogleColab/GoogleColab_Inching_v1.1.ipynb`. Notebooks for the paper workflow: `Notebook/Application`.

**Published version.** The code used in the paper is tag [`v1.0`](https://github.com/jhmlam/Inching/tree/v1.0)
(Zenodo [10.5281/zenodo.10645601](https://doi.org/10.5281/zenodo.10645601), package name `InchingLiteInteger`).
v1.1 renames the package to `inching`, adds the CLI and runs on current CuPy 14 / PyTorch 2 / CUDA 12-13;
the algorithms are unchanged. The v1.0 Colab notebook (`GoogleColab_Inching_v023_ReleaseOkay.ipynb`) needs CUDA 11 and no longer runs; use `GoogleColab_Inching_v1.1.ipynb`.

## Our Contributions
- [x] Developed a Globally-Sparse-Yet-Locally-Dense computational approach applicable to both small- and large- macromolecules.
- [x] Implemented several GPU-resident Eigensolvers for Sparse Symmetric Matrices (Jacobi Davidson Method, Thick Restart Lanczos Method, Implicitly Restarted Lanczos Method)
- [x] Modularized support for seamless integration of Chebyshev filter diagonalization, sum decomposition, and external explicit deflation, among other techniques.
- [x] Achieved Linear Scaling in run time, with constant memory use, relative to the number of modes. See CTRLM.



![Alt Text](/assets/Animation_Inching_3j3q_06.gif)







