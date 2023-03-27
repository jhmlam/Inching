# ===================== Basic Imports and Settings================================
import glob
import platform

# A list of pdb available at different sizes

Benchmarking_folder = "../BenchmarkLinuxInchingIRLM0064_A100/"
pdbavail = sorted(glob.glob('../DataRepo/CifByAtomCount/*.cif'))
User_Platform = platform.system() # Windows Darwin Linux

User_rc_Gamma = 8.0
User_maxleafsize = 100
User_n_mode = 64 
User_tol = 1e-15
User_PlusI = 1.0
PDBCIF = "Cif"
User_MaxIter = 15000

# IRLMHD Params
User_GapEstimate = 1e-6
User_SolverName = 'gmres'
User_SolverMaxIter = 20
User_EigTolerance = 1e-12



PART00_Import = True
if PART00_Import:
   import os
   import gc
   import sys
   import pickle
   #import prody
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







print(len(pdbavail))
pdbavail = ["../DataRepo/PdbByAtomCount/1a1x.pdb"] + pdbavail#[::-1]
for pdbfn in pdbavail:
    

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
    if os.path.exists("%s/PerformanceList_InchingIRLMHD_%s_%s_%s.pkl" %(Benchmarking_folder, pdbid, User_Platform, User_Device.replace(" ","") )):
        #if '1a1x' not in pdbid:
            continue
    print(pdbfn)
    st = time.time()

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
    cuthill_order, cuthill_undoorder = InchingLite.Fuel.Coordinate.T1.X_KdCuthillMckeeOrder(protein_xyz,  
                                rc_Gamma = User_rc_Gamma, Reverse = True,
                                )
    protein_xyz = protein_xyz[cuthill_order,:]
    protein_tree = cKDTree(protein_xyz, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)


    
    from InchingLite.Burn.ImplicitlyRestartedLanczosHotellingDeflation.T1 import S_HeigvalIRLMHD_HeigvecIRLMHD
    print('start eigsh cupy')







    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()





    # ==================
    # Cupy hessian
    # =====================
    PART03_MakeCupyHessian = True
    if PART03_MakeCupyHessian:
        # NOTE Nnz neighborhood after cuthill
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
        print(A.data.shape)
        PARTZZZ_CheckCorrect = False
        if PARTZZZ_CheckCorrect:
            print(cupy.allclose(A.data, A.T.data, rtol=1e-10, atol = 1e-10, equal_nan = False))

            for iiiii in range(3):
              tx_matrix = cupy.zeros((int(A.shape[0]/3), 3), dtype= A.dtype)
              tx_matrix[:,iiiii] += 1
              tx_matrix = tx_matrix.reshape(A.shape[0])
              tx_matrix /= cublas.nrm2(tx_matrix)

              print(tx_matrix)

              bb = A@tx_matrix + A.T @ tx_matrix - cupy.multiply(A.diagonal(k=0), tx_matrix)

              cc = bb - tx_matrix
              print(cupy.sum(cc))
              print(A[:20,:20])
              print(cupy.abs(cc))
              print(cupy.where(cupy.abs(cc) > 1e-10))
            sys.exit()

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()




    PART03b_MakeFreeModes = False
    if PART03b_MakeFreeModes:

        Q_HotellingDeflation = cp.zeros((6,3*n_atoms), dtype = cp.float64)
        # NOTE Translation
        for i in range(3):
            q1 = cp.zeros((n_atoms,3))
            q1[:,i] = 1/np.sqrt(n_atoms)
            Q_HotellingDeflation[i,:] = q1.flatten()
            q1 = None
            del q1
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()


        
        # NOTE Rotation
        R_x = cp.array([        [0,0,0],
                                [0,0,-1],
                                [0,1,0]], dtype=cp.float64).T
        R_y = cp.array([        [0,0,1],
                                [0,0,0],
                                [-1,0,0]], dtype=cp.float64).T
        R_z = cp.array([        [0,-1,0],
                                [1,0,0],
                                [0,0,0]], dtype=cp.float64).T
        R_x = cupysparse.csr_matrix(R_x, dtype= cp.float64)
        R_y = cupysparse.csr_matrix(R_y, dtype= cp.float64)
        R_z = cupysparse.csr_matrix(R_z, dtype= cp.float64)
        gx = (cp.array(X)@R_x).flatten()
        Q_HotellingDeflation[3,:] = gx/ cp.linalg.norm(gx,ord=2)
        gy = (cp.array(X)@R_y).flatten()
        Q_HotellingDeflation[4,:] = gy/ cp.linalg.norm(gy,ord=2)
        gz = (cp.array(X)@R_z).flatten()
        Q_HotellingDeflation[5,:] = gz/ cp.linalg.norm(gz,ord=2)
        

        
        for i_FRO in range(2):
            V = Q_HotellingDeflation.T
            
            for ix in range(6):
                if ix == 0:
                    continue
                V[:,ix] -= cp.matmul(V[:,:ix], cp.matmul( V[:, :ix].T,V[:,ix] ))
                V[:,ix] /= cp.sqrt(V[:, ix].T @ V[:, ix]) # TODO torch.matmul or mvs
                V[:,ix] -= cp.matmul(V[:,:ix], cp.matmul( V[:, :ix].T,V[:,ix] ))
                V[:,ix] /= cp.sqrt(V[:, ix].T @ V[:, ix])
            Q_HotellingDeflation = V.T
        
        gx = Q_HotellingDeflation[3]

        
        Q_HotellingDeflation = cupyx.scipy.sparse.csr_matrix(Q_HotellingDeflation, dtype = cp.float64)

        gx, gy, gz = None, None, None
        del gx, gy, gz
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()



    
    PART04_CalcualteEig = True
    if PART04_CalcualteEig:
        eigval, eigvec = S_HeigvalIRLMHD_HeigvecIRLMHD(A,
                    k = User_n_mode ,
                    User_HalfMemMode= True,
                    tol=User_EigTolerance,maxiter=User_MaxIter,    #set the tolerence and maximum iteration as a stop criteria.
                    User_Q_HotellingDeflation = None, #Q_HotellingDeflation,
                    User_HotellingShift = 10, # NOTE 10 is generally safe for first 64 modes, of course if you want to guarentee it you know a norm
                    )
        runtime = time.time() - st
        print("RUNNNTIME %s" %(runtime))
        peak_mem = cupy.get_default_memory_pool().used_bytes() / 1024 / 1024







        with open("%s/Eigval_InchingIRLMHD_%s_%s_%s.pkl" %(
                    Benchmarking_folder, pdbid, User_Platform, 
                    User_Device.replace(" ","")),"wb") as fn:
            pickle.dump(cupy.asnumpy(eigval) - User_PlusI ,fn, protocol=4)
        
        with open("%s/Eigvec_InchingIRLMHD_%s_%s_%s.pkl" %(
                    Benchmarking_folder, pdbid, User_Platform, 
                    User_Device.replace(" ","")),"wb") as fn:    
            tempeigvec = cupy.asnumpy(eigvec)
            tempeigvec = tempeigvec.T
            tempeigvec = tempeigvec.reshape((int(User_n_mode),int(n_atoms),int(3)))
            pickle.dump(tempeigvec[:,cuthill_undoorder,:] ,fn, protocol=4)

        
        # NOTE unfortunately We need to separate pymol from the main script and install a separate environment
        """
        for User_TheModeToShow in range(15):
            
            if User_TheModeToShow <=5:
                continue

            if pdbfn.split(".")[-1] == 'pdb':
                nmfactor = 0.1
            else:
                nmfactor = 1 

            if '3j3q' in pdbid:
               InchingLite.util.SaveOneModeLinearisedAnime(
                            torch.tensor(tempeigvec[User_TheModeToShow,cuthill_undoorder,:],
                                    dtype = torch.float64, device = torch.device('cpu')),
                            torch.tensor(X[cuthill_undoorder,:], device = torch.device('cpu'))*nmfactor,
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
                            torch.tensor(tempeigvec[User_TheModeToShow,cuthill_undoorder,:],
                                    dtype = torch.float64, device = torch.device('cpu')),
                            torch.tensor(X[cuthill_undoorder,:], device = torch.device('cpu'))*nmfactor,
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
        """         
        del tempeigvec
        gc.collect()
    
       




    PART05_Performance = True
    if PART05_Performance:
        #===================================
        # Check correct
        # =====================================
        #print(eigval)
        #print(eigvec.shape)
        delta_lambda_list = []

        for jj in range(User_n_mode):
            B = A@eigvec[:,jj].T + A.T@eigvec[:,jj].T - cupy.multiply(A.diagonal(k=0), eigvec[:,jj])  - eigval[jj]* eigvec[:,jj].T
            delta_lambda_list.append(cupy.asnumpy(cublas.nrm2(B)))
            if jj < 20:
                print(eigval[jj], cupy.asnumpy(cublas.nrm2(B)))
        
        eigval = cupy.asnumpy(eigval)

        n_atoms = protein_xyz.shape[0]

        GPU = "%s %s" %(User_Platform, User_Device.replace(" GPU", ""))

        performance = ["Inching (IRLM %s)" %(GPU), pdbfn, n_atoms, 
                        runtime, peak_mem, 
                        User_Platform, User_Device, 
                        User_maxleafsize]



        longperformance = []
        for i in range(len(delta_lambda_list)):
            longperformance.append(performance + [i ,delta_lambda_list[i], eigval[i] - User_PlusI])
        
        with open("%s/PerformanceList_InchingIRLMHD_%s_%s_%s.pkl" %(Benchmarking_folder, 
            pdbid, User_Platform, User_Device.replace(" ","")),"wb") as fn:   
            pickle.dump(longperformance,fn, protocol=4)


        del X_df, protein_xyz
        gc.collect()



        B = None
        A.data = None
        A.indices = None
        A.indptr = None
        Q_HotellingDeflation = None
        del Q_HotellingDeflation

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
        torch.cuda.empty_cache()    
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.memory_allocated(0)
        torch.cuda.max_memory_allocated(0)
