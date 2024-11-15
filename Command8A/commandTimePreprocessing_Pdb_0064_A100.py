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
# ===================== Basic Imports and Settings================================
import glob
import platform

# A list of pdb available at different sizes

Benchmarking_folder = "../BenchmarkLinuxTimePreprocessing0064_A100/"
pdbavail = sorted(glob.glob('../DataRepo/PdbByAtomCount/*.pdb'))
User_Platform = platform.system() # Windows Darwin Linux
User_rc_Gamma = 8.0
User_maxleafsize = 100
User_n_mode = 64
User_tol = 1e-15

User_PlusI = 1 # NOTE we do not heal the condition number to mimick a less careful implementation
PDBCIF = "Pdb"


PART00_Import = True
if PART00_Import:
   import os
   import gc
   import sys
   import pickle
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
   import InchingLite.Fuel.Coordinate.T1
   import InchingLite.Fuel.Coordinate.T2
   import InchingLite.Burn.Coordinate.T1
   import InchingLite.Burn.Coordinate.T3

   from InchingLite.Fuel.T1 import X_SparseCupyMatrix, Xnumpy_SparseCupyMatrixUngappped

   import InchingLite.Burn.Visualisation.T1
   import InchingLite.Burn.Visualisation.T2

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







benchmark_inching = []
print(len(pdbavail))
pdbavail = ["../DataRepo/PdbByAtomCount/1a1x.pdb"] + pdbavail#[::-1]
for pdbfn in pdbavail[:]:
    

    #if '7bii' not in pdbfn:
    #    continue

    devices_ = [d for d in range(torch.cuda.device_count())]
    device_names_  = [torch.cuda.get_device_name(d) for d in devices_]
    User_Device =  device_names_[0]

    pdbid = pdbfn.split("/")[-1].split(".")[0]
    #if '1a1x' not in pdbid:
    #    #if '4v4k' not in pdbid: 
    #        continue
    #else:
    #    pass
    if os.path.exists("%s/PerformanceList_TimePreprocessing_%s_%s_%s.pkl" %(Benchmarking_folder, pdbid, User_Platform, User_Device.replace(" ","") )):
            continue






    print(pdbfn)

    X_df, X_top = InchingLite.util.BasicPdbCifLoading(pdbfn)
    protein_xyz = X_df[['x','y','z']].to_numpy().astype(np.float64)
    # NOTE PDB format digit decimal do no destroy collinearity!
    protein_xyz -= np.around(protein_xyz.mean(axis= 0), decimals=4)
    #protein_xyz -= protein_xyz.mean(axis= 0)
    #print(protein_xyz[:30,:]*100000)
    #sys.exit()
    #print(protein_xyz.shape)
    n_atoms = protein_xyz.shape[0]



    print(pdbid, n_atoms)
    #if protein_xyz.shape[0] > 50000: # NOTE We will refrain from testing > 60k system which will takes > 3 hours to complete in general
    #    continue

    # ===============================================
    # K-d Cuthill (NOTE CPU np array)
    # ===================================
    # NOTE Cuthill Order and Undo
    st = time.time()
    tracemalloc.start()
    cuthill_order, cuthill_undoorder = InchingLite.Fuel.Coordinate.T1.X_KdCuthillMckeeOrder(protein_xyz,  
                                rc_Gamma = User_rc_Gamma, Reverse = True,
                                )

    Measured_CuthillTime = time.time() - st

    peak_mem_Cuthill = tracemalloc.get_traced_memory()
    peak_mem_Cuthill = peak_mem_Cuthill[1] / 1024 / 1024
    tracemalloc.stop()
    #protein_xyz = protein_xyz[cuthill_order,:]
    protein_tree = cKDTree(protein_xyz, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)

    
    
    #print(A)
    print('start eigsh cupy')







    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()




    if os.path.exists("%s/Matrix_TimePreprocessing_%s.pkl" %(Benchmarking_folder, pdbid)):
        print("THIS SHOULD NOT HAPPEN!")
    else:


        # ==================
        # Cupy hessian
        # =====================
        PART03_MakeCupyHessian = True
        if PART03_MakeCupyHessian:
            st = time.time()
            NnzMinMaxDict, HalfNnz  = InchingLite.Fuel.Coordinate.T1.X_KdUngappedMinMaxNeighbor(protein_xyz, 
                                    rc_Gamma = User_rc_Gamma, 
                                    maxleafsize = User_maxleafsize,
                                    CollectStat = False,
                                    User_ReturnHalfNnz = True,
                                    SliceForm= True)


            # NOTE Pyotch tensor spend textra memory when dlpack has to be called and there are mmeleak
            #X = torch.tensor(protein_xyz, device=device, requires_grad= False)
            X = protein_xyz
            Xnumpy_SparseCupyMatrixUngapppedC = Xnumpy_SparseCupyMatrixUngappped(X, batch_head = None, 
                maxleafsize = User_maxleafsize, rc_Gamma = User_rc_Gamma,
                device  = torch.device(0), 
                User_PlusI = User_PlusI, 
                dtype_temp = torch.float64, 
                X_precision = torch.cuda.DoubleTensor,
                NnzMinMaxDict = NnzMinMaxDict)

            A = Xnumpy_SparseCupyMatrixUngapppedC.ReturnCupyHLowerTriangle(
                        User_MaxHalfNnzBufferSize = HalfNnz)
            Measured_HessianTime = time.time() - st
            peak_mem = cupy.get_default_memory_pool().used_bytes() / 1024 / 1024
            print(A.data.shape)






            B = None
            A.data = None
            A.indices = None
            A.indptr = None
            del A.data, A.indices, A.indptr
            del A, B

            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
            del X
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.memory_allocated(0)
            torch.cuda.max_memory_allocated(0)
   


            import pickle





    # ===============
    # Performance
    # ===============

    n_atoms = protein_xyz.shape[0]

    
    GPU = "%s %s" %(User_Platform, User_Device.replace(" GPU", ""))

    performance = [["3DRCM" , pdbfn, n_atoms,
                    Measured_CuthillTime , peak_mem_Cuthill,
                    User_Platform, User_Device,
                    User_maxleafsize],
                   ["Hessian", pdbfn, n_atoms,
                    Measured_HessianTime , peak_mem,
                    User_Platform, User_Device,
                    User_maxleafsize],
                  ]



    with open("%s/PerformanceList_TimePreprocessing_%s_%s_%s.pkl" %(Benchmarking_folder,
        pdbid, User_Platform, User_Device.replace(" ","")),"wb") as fn:
        pickle.dump(performance,fn, protocol=4)





    PART05_cleanup = True
    if PART05_cleanup:

        del X_df, protein_xyz
        gc.collect()



        #Xnumpy_SparseCupyMatrixUngapppedC.X, Xnumpy_SparseCupyMatrixUngapppedC.X_unsqueezed = None, None
        #del Xnumpy_SparseCupyMatrixUngapppedC.X, Xnumpy_SparseCupyMatrixUngapppedC.X_unsqueezed
        Xnumpy_SparseCupyMatrixUngapppedC = None
        del Xnumpy_SparseCupyMatrixUngapppedC
        eigvec, eigval = None, None
        del eigvec, eigval


