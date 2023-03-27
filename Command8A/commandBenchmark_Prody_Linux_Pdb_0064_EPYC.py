# ===================== Basic Imports and Settings================================
import glob
import platform

# A list of pdb available at different sizes

Benchmarking_folder = "../BenchmarkLinuxPrody0064_EPYC/"
pdbavail = sorted(glob.glob('../DataRepo/PdbByAtomCount/*.pdb'))
User_Platform = platform.system() # Windows Darwin Linux
User_Device = "AMD-EPYC-7513-32Core"

User_rc_Gamma = 8.0
User_maxleafsize = 100
User_n_mode = 64
User_tol = 1e-15
User_PlusI = 0 # NOTE Prody do not have +1
PDBCIF = "Pdb"
User_MaxIter = 15000




PART00_Import = True
if PART00_Import:
   import os
   import gc
   import sys
   import pickle
   import prody
   import numpy as np
   import time
   import tqdm
   import tracemalloc
   import scipy
   import torch 


   import platform


   import cupy as cp
   import cupyx
   from cupyx.scipy.sparse import linalg as cupylinalg
   from cupyx.scipy import sparse as cupysparse
   import scipy.sparse.linalg as scipylinalg
   #import lanczosrawtidySmallFRO2

   import time
   import scipy.sparse

   import cupy
   from cupy import cublas



   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   from scipy.spatial import cKDTree



   sys.path.append('..')
   sys.path.append('../InchingLite/Burn/')


   import torch





   sys.path.append('..')
   sys.path.append('../InchingLite/Burn/')
   import InchingLite.util
   #import InchingLite.Fuel.Coordinate.T1
   #import InchingLite.Fuel.Coordinate.T2
   #import InchingLite.Burn.Coordinate.T1
   #import InchingLite.Burn.Coordinate.T3

   #from InchingLite.Fuel.T1 import X_SparseCupyMatrix, Xnumpy_SparseCupyMatrixUngappped

   #import InchingLite.Burn.Visualisation.T1
   #import InchingLite.Burn.Visualisation.T2

   # ============================
   # Some torch speed up tips
   # =============================

   # Turn on cuda optimizer
   torch.backends.cudnn.is_available()
   torch.backends.cudnn.enabled = True
   torch.backends.cudnn.benchmark = True
   # disable debugs NOTE use only after debugging
   torch.autograd.set_detect_anomaly(False)
   torch.autograd.profiler.profile(False)
   torch.autograd.profiler.emit_nvtx(False)
   # Disable gradient tracking
   torch.no_grad()
   torch.inference_mode()
   torch.manual_seed(0)
   cupy.random.seed(seed = 0)
   os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # NOTE In case any error showup
   # Reset Cuda and Torch
   device = torch.device(0)
   torch.set_default_dtype(torch.float64)
   torch.set_default_tensor_type(torch.cuda.DoubleTensor)
   try:
      InchingLite.util.TorchEmptyCache()
   except RuntimeError:
      print("The GPU is free to use. THere is no existing occupant")
   try:
      print(torch.cuda.memory_summary(device = 0, abbreviated=True))
   except KeyError:
      print("The GPU is free to use. THere is no existing occupant")


# =======================
# Determine N_atoms
# ==============================
# NOTE 2 minutes
PART01_ListOfPDB = True
if PART01_ListOfPDB:
   if os.path.exists("%s/%sSize.pkl" %(Benchmarking_folder, PDBCIF)):
      with open("%s/%sSize.pkl" %( Benchmarking_folder, PDBCIF),"rb") as fn:
         pdbavail, sizedict = pickle.load(fn)
   else:

      pdbavail = [InchingLite.util.WinFileDirLinux(i) for i in pdbavail]
      size = []
      for pdbfn in tqdm.tqdm(pdbavail):

         X_df, X_top = InchingLite.util.BasicPdbCifLoading(pdbfn)
         protein_xyz = X_df[['x','y','z']].to_numpy().astype(np.float64)
         size.append(protein_xyz.shape[0])
         del X_df, protein_xyz
         gc.collect()



      pdbavail = [pdbavail[i] for i in np.argsort(size).tolist()]
      print("Ranked file size in atom number")
      #print([os.path.getsize(i) for i in pdbavail])
      sizedict = dict(zip([i.split("/")[-1].split(".")[0] for i in pdbavail],sorted(size)))
      print(dict(zip([i.split("/")[-1].split(".")[0] for i in pdbavail],sorted(size))))

      with open("%s/%sSize.pkl" %(Benchmarking_folder, PDBCIF),"wb") as fn:
         pickle.dump((pdbavail, dict(zip([i.split("/")[-1].split(".")[0] for i in pdbavail],sorted(size)))),fn , protocol=4)







print(len(pdbavail))
pdbavail = ["../DataRepo/PdbByAtomCount/1a1x.pdb"] + pdbavail#[::-1]
for pdbfn in pdbavail:
    


    pdbid = pdbfn.split("/")[-1].split(".")[0]

    if os.path.exists("%s/PerformanceList_Prody_%s_%s_%s.pkl" %(Benchmarking_folder, pdbid, User_Platform, User_Device.replace(" ","") )):
        #if '1a1x' not in pdbid:
            continue
    print(pdbfn)
    st = time.time()

    X_df, X_top = InchingLite.util.BasicPdbCifLoading(pdbfn)
    protein_xyz = X_df[['x','y','z']].to_numpy().astype(np.float64)
    # NOTE PDB format digit decimal do no destroy collinearity!
    protein_xyz -= np.around(protein_xyz.mean(axis= 0), decimals=4)
    n_atoms = protein_xyz.shape[0]



    print(pdbid, n_atoms)
    #if protein_xyz.shape[0] > 50000: # NOTE We will refrain from testing > 60k system which will takes > 3 hours to complete in general
    #    continue





    # =======================
    # ANM Calcilatin
    # =======================
    st = time.time()
    tracemalloc.start()
    # NOTE Peak size not current after everything closed! 
    #      Also note that there can be a memory copying procedure s.t. the dense matrix is doubled in size
    anm = prody.ANM()
    anm.buildHessian(protein_xyz.astype(np.float64)*10.0, cutoff=8.0, gamma=1.0)# , sparse = True)
    anm.calcModes(n_modes = User_n_mode, zeros = True)

    peak_mem = tracemalloc.get_traced_memory()
    peak_mem = peak_mem[1] / 1024 / 1024

    tracemalloc.stop()
    runtime = time.time() - st


    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()



    # =================
    # Save 
    # ========================

    eigval = anm.getEigvals()
    eigval = eigval.astype(np.float64)
    eigvec = anm.getEigvecs()
    eigvec = eigvec.astype(np.float64)


    with open("%s/Eigval_Prody_%s_%s_%s.pkl" %(Benchmarking_folder, pdbid, User_Platform, User_Device.replace(" ", "")),"wb") as fn:
        pickle.dump(np.array(eigval),fn, protocol=4)
    with open("%s/Eigvec_Prody_%s_%s_%s.pkl" %(Benchmarking_folder, pdbid, User_Platform, User_Device.replace(" ", "") ),"wb") as fn:
        pickle.dump(np.array(eigvec.T.reshape(User_n_mode,int(n_atoms),int(3))),fn, protocol=4)



    





    PART05_Performance = True
    if PART05_Performance:

        # ===============
        # Performance
        # ===============
        delta_lambda_list = []
        for jj in range(User_n_mode):
            B = anm._hessian@eigvec[:,jj].T - eigval[jj] * eigvec[:,jj].T

            delta_lambda_list.append(np.linalg.norm(B, ord=2))
            print(eigval[jj], np.linalg.norm(B, ord=2), jj)

        n_atoms = protein_xyz.shape[0]


        GPU = "%s %s" %(User_Platform, User_Device.replace(" GPU", ""))





        performance = ["ProDy 2.4 (ARPACK %s)" %(GPU), pdbfn, n_atoms, 
                        runtime, peak_mem, 
                        User_Platform, User_Device, 
                        User_maxleafsize]



        longperformance = []
        for i in range(len(delta_lambda_list)):
            longperformance.append(performance + [i ,delta_lambda_list[i], eigval[i] - User_PlusI])
        
        with open("%s/PerformanceList_Prody_%s_%s_%s.pkl" %(Benchmarking_folder, 
            pdbid, User_Platform, User_Device.replace(" ","")),"wb") as fn:   
            pickle.dump(longperformance,fn, protocol=4)


        del X_df, protein_xyz
        gc.collect()






        del anm

        B = None
        del B
        eigvec, eigval = None, None
        del eigvec, eigval

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()


        gc.collect()
