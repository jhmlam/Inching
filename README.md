# Inching - Scalable Computation for Anisotropic Vibrations of Large Macromolecular Assemblies
[![Click to try on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhmlam/Inching/blob/main/GoogleColab/GoogleColab_Inching_v023_ReleaseOkay.ipynb)
## Abstract
The Normal Mode Analysis (NMA) is a standard approach to elucidate the anisotropic vibrations of macromolecules at their folded states, where low-frequency collective motions can reveal rearrangements of domains and changes in the exposed surface of macromolecules. Recent advances in structural biology have enabled the resolution of megascale macromolecules with millions of atoms. However, the calculation of their vibrational modes remains elusive due to the prohibitive cost associated with constructing and diagonalizing the underlying eigenproblem and the current approaches to NMA are not readily adaptable for efficient parallel computing on graphic processing unit (GPU). Here, we present eigenproblem construction and diagonalization approach that implements level-structure bandwidth-reducing algorithms to transform the sparse computation in NMA to a globally-sparse-yet-locally-dense computation, allowing batched tensor products to be most efficiently executed on GPU. We mapped, optimized, and compared several low-complexity Krylov-subspace eigensolvers, supplemented by techniques such as Chebyshev filtering, sum decomposition, external explicit deflation and shift-and-inverse, to allow fast GPU-resident calculations. The method allows accurate calculation of the first 1000 vibrational modes of some largest structures in PDB (> 2.4 million atoms) at least 250 times faster than existing methods.

## Contributions
- [x] Developed a Globally-Sparse-Yet-Locally-Dense computational approach applicable to both small- and large- macromolecules.
- [x] Implemented several GPU-resident Eigensolvers for Sparse Symmetric Matrices (Jacobi Davidson Method, Thick Restart Lanczos Method, Implicitly Restarted Lanczos Method)
- [x] Modularized support for seamless integration of Chebyshev filter diagonalization, sum decomposition, and external explicit deflation, among other techniques.
- [x] Achieved Linear Scaling in run time, with constant memory use, relative to the number of modes. See CTRLM



![Alt Text](/assets/Animation_Inching_3j3q_06.gif)



## Summary
This is a codebase for `XYZZ` used in the following paper

> "VDFVSDVV" by ABBCBC [[arXiv](https://arxiv.org/abs/2106.dasd342342354)]

Here's a Colab notebook which contains an example for tCCCCCC

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/XYSSDFFSDFS)  






## How to contribute

We have a detailed account of the contribution procedure and guidelines in the corresponding file: [CONTRIBUTING](CONTRIBUTING.md)





