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

import glob
import platform

# A list of pdb available at different sizes

Benchmarking_folder = "../ShowcaseLinuxInchingJDM0064_A100/"
pdbavail = sorted(glob.glob('../DataRepo/CifShowcase/*.cif'))
User_Platform = platform.system() # Windows Darwin Linux


User_Device_Overide = "NVIDIAA100-PCIE-40GB"
User_rc_Gamma = 8.0
User_maxleafsize = 100
User_n_mode = 64
User_tol = 1e-15
User_Eigsolver="JDM"
PDBCIF = "Cif"


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
   from InchingLite.Fuel.Coordinate.T1 import HeigvecOne_BoxCoxMagnitude

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
pdbavail = pdbavail[:]
for pdbfn in pdbavail[:]:
    

    #if '7bii' not in pdbfn:
    #    continue

    devices_ = [d for d in range(torch.cuda.device_count())]
    device_names_  = [torch.cuda.get_device_name(d) for d in devices_]
    User_Device =  device_names_[0]
    User_Device = User_Device_Overide # "NVIDIAA100-PCIE-40GB"

    pdbid = pdbfn.split("/")[-1].split(".")[0]
    #if '1a1x' not in pdbid:
    #    #if '4v4k' not in pdbid: 
    #        continue
    #else:
    #    pass
    #if not os.path.exists("%s/PerformanceList_Inching_%s_%s_%s.pkl" %(Benchmarking_folder, pdbid, User_Platform, User_Device.replace(" ","") )):
    #        continue






    print(pdbfn)
    st = time.time()

    X_df, X_top = InchingLite.util.BasicPdbCifLoading(pdbfn)
    protein_xyz = X_df[['x','y','z']].to_numpy().astype(np.float64)
    # NOTE PDB format digit decimal do no destroy collinearity!
    protein_xyz -= np.around(protein_xyz.mean(axis= 0), decimals=4)
    n_atoms = protein_xyz.shape[0]



    print(pdbid, n_atoms)

    # ===============================================
    # K-d Cuthill (NOTE CPU np array)
    # ===================================
    # NOTE Cuthill Order and Undo


    
    st = time.time()





    PART04_GetEigvecEigvalSaved = True
    if PART04_GetEigvecEigvalSaved:
        
        with open("%s/Eigval_Inching%s_%s_%s_%s.pkl" %(
                    Benchmarking_folder, User_Eigsolver, pdbid, User_Platform,
                    User_Device.replace(" ","")),"rb") as fn:
            eigval = pickle.load(fn)

        with open("%s/Eigvec_Inching%s_%s_%s_%s.pkl" %(
                    Benchmarking_folder, User_Eigsolver, pdbid, User_Platform,
                    User_Device.replace(" ","")),"rb") as fn:
            eigvec = pickle.load(fn)
            

    PART05_Animate = True
    if PART05_Animate:
        for User_TheModeToShow in range(15):

            if User_TheModeToShow <=5:
                continue

            if pdbfn.split(".")[-1] == 'pdb':
                nmfactor = 0.1
            else:
                nmfactor = 1
            PART05a_Raw = False
            if PART05a_Raw:
                if '3j3q' in pdbid:
                    InchingLite.util.SaveOneModeLinearisedAnime(
                            eigvec[User_TheModeToShow,:,:],
                            protein_xyz*nmfactor,
                            n_timestep = 5,
                            DIR_ReferenceStructure = pdbfn,#[:-4] + "trial.cif",
                            DIR_SaveFolder = Benchmarking_folder,
                            SaveFormat = 'cif',
                            outputlabel = 'Propagate_%s_%s'%(pdbid, User_TheModeToShow),
                            max_abs_deviation = 2.0*nmfactor,
                            stepsize = 1.0*nmfactor,
                            max_n_output = 5,
                            SaveSeparate = False,
                            RemoveOrig = True, # NOTE This flag remove the unmoved structure from the trajectory produce
                            )
                else:
                    InchingLite.util.SaveOneModeLinearisedAnime(
                            eigvec[User_TheModeToShow,:,:],
                            protein_xyz*nmfactor,
                            n_timestep = 16,
                            DIR_ReferenceStructure = pdbfn,#[:-4] + "trial.cif",
                            DIR_SaveFolder = Benchmarking_folder,
                            SaveFormat = 'cif',
                            outputlabel = 'Propagate_%s_%s'%(pdbid, User_TheModeToShow),
                            max_abs_deviation = 3.0*nmfactor,
                            stepsize = 1.0*nmfactor,
                            max_n_output = 32,
                            SaveSeparate = False,
                            RemoveOrig = True, # NOTE This flag remove the unmoved structure from the trajectory produce
                            )




            PART05b_BoxCox = True
            if PART05b_BoxCox:

                if os.path.exists("%s/%s_BoxCox_%s_%s.cif" %(Benchmarking_folder, pdbid, pdbid, User_TheModeToShow)):
                    continue

                # NOTE Kerneled eigvec
                deltaX_magnitude = HeigvecOne_BoxCoxMagnitude( eigvec[User_TheModeToShow,:,:],
                        User_WinsorizingWindow = (0.025, 0.975),
                        User_LogisticParam = (0.05, 1.0),
                        )

                deltaX_magnitude = np.clip(deltaX_magnitude, 0.1, 1.0)
                eigvec_unit = eigvec[User_TheModeToShow] / np.linalg.norm(eigvec[User_TheModeToShow], axis=1)[:,None]
                deltaX = deltaX_magnitude[:,None] * eigvec_unit


                if '3j3q' in pdbid:
                    InchingLite.util.SaveOneModeLinearisedAnime(
                            deltaX, 
                            protein_xyz*nmfactor,                
                            n_timestep = 5, 
                            DIR_ReferenceStructure = pdbfn,#[:-4] + "trial.cif",
                            DIR_SaveFolder = Benchmarking_folder,
                            SaveFormat = 'cif',
                            outputlabel = 'BoxCox_%s_%s'%(pdbid, User_TheModeToShow),
                            max_abs_deviation = 2.0*nmfactor,
                            stepsize = 1.0*nmfactor,
                            UnitMovement = False,
                            max_n_output = 5,
                            SaveSeparate = False,
                            RemoveOrig = True, # NOTE This flag remove the unmoved structure from the trajectory produce
                            User_Bfactor=deltaX_magnitude
                            )
                else:

                    InchingLite.util.SaveOneModeLinearisedAnime(
                            deltaX, 
                            protein_xyz*nmfactor, 
                            n_timestep = 16, 
                            DIR_ReferenceStructure = pdbfn,#[:-4] + "trial.cif",
                            DIR_SaveFolder = Benchmarking_folder, 
                            SaveFormat = 'cif',
                            outputlabel = 'BoxCox_%s_%s'%(pdbid, User_TheModeToShow),
                            max_abs_deviation = 3.0*nmfactor,
                            stepsize = 1.0*nmfactor,
                            UnitMovement = False,
                            max_n_output = 32,
                            SaveSeparate = False,
                            RemoveOrig = True, # NOTE This flag remove the unmoved structure from the trajectory produce
                            User_Bfactor=deltaX_magnitude
                            )





        del eigvec
        gc.collect()




    





    PART05_Performance = True
    if PART05_Performance:


        del X_df, protein_xyz
        gc.collect()


        """
        B = None
        A.data = None
        A.indices = None
        A.indptr = None
        del A.data, A.indices, A.indptr 
        del A, B
        Xnumpy_SparseCupyMatrixUngapppedC.X, Xnumpy_SparseCupyMatrixUngapppedC.X_unsqueezed = None, None
        del Xnumpy_SparseCupyMatrixUngapppedC.X, Xnumpy_SparseCupyMatrixUngapppedC.X_unsqueezed
        Xnumpy_SparseCupyMatrixUngapppedC = None
        del Xnumpy_SparseCupyMatrixUngapppedC
        eigvec, eigval = None, None
        del eigvec, eigval


        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        del X 
        """
        #torch.cuda.empty_cache()    
        #torch.cuda.reset_peak_memory_stats(0)
        #torch.cuda.memory_allocated(0)
        #torch.cuda.max_memory_allocated(0)
