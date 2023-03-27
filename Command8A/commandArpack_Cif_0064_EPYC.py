# ===================== Basic Imports and Settings================================
import glob
import platform

# A list of pdb available at different sizes

Benchmarking_folder = "../BenchmarkLinuxArpack0064_EPYC/"
pdbavail = sorted(glob.glob('../DataRepo/CifByAtomCount/*.cif'))
User_Platform = platform.system() # Windows Darwin Linux
User_rc_Gamma = 8.0
User_maxleafsize = 100
User_n_mode = 64
User_tol = 1e-15

User_PlusI = 1 # NOTE we do not heal the condition number to mimick a less careful implementation
PDBCIF = "Cif"


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
    User_Device = "AMD-EPYC-7513-32Core" #str(platform.processor()) Obtained by cat /proc/cpuinfo
    if os.path.exists("%s/PerformanceList_ArpackOri_%s_%s_%s.pkl" %(Benchmarking_folder, pdbid, User_Platform, User_Device.replace(" ","") )):
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
    st = time.time()
    # NOTE Cuthill Order and Undo
    cuthill_order, cuthill_undoorder = InchingLite.Fuel.Coordinate.T1.X_KdCuthillMckeeOrder(protein_xyz,  
                                rc_Gamma = User_rc_Gamma, Reverse = True,
                                )
    #protein_xyz = protein_xyz[cuthill_order,:]
    protein_tree = cKDTree(protein_xyz, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)


    
    #print(A)
    print('start eigsh cupy')







    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()




    if os.path.exists("%s/Matrix_ArpackOri_%s.pkl" %(Benchmarking_folder, pdbid)):
        tracemalloc.start()
        import pickle
        with open("%s/Matrix_ArpackOri_%s.pkl" %(Benchmarking_folder, pdbid) , 'rb') as fn:
             A_scipy = pickle.load(fn)
    else:


        # ==================
        # Cupy hessian
        # =====================
        PART03_MakeCupyHessian = True
        if PART03_MakeCupyHessian:
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


            import scipy.sparse
            tracemalloc.start()
            A_scipy = scipy.sparse.csr_matrix((A.data.get(), A.indices.get(), A.indptr.get()), shape=(A.shape[0], A.shape[1]))
            A_scipy += scipy.sparse.tril(A_scipy, k=-1).T
            print("Maked e fedscipy")




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
            with open("%s/Matrix_ArpackOri_%s.pkl" %(Benchmarking_folder, pdbid) , 'wb') as fn:
                pickle.dump(A_scipy,fn, protocol = 4)



    # =========================
    # Arpack (3DRCM)
    # ========================+
    import scipy.sparse.linalg as scipy_sparse_la


    scipy_eigval, scipy_eigvec = scipy_sparse_la.eigsh(A_scipy, k=User_n_mode, which='SA')
    runtime = time.time() - st
    peak_mem = tracemalloc.get_traced_memory()
    peak_mem = peak_mem[1] / 1024 / 1024
    tracemalloc.stop()


    with open("%s/Eigval_ArpackOri_%s_%s_%s.pkl" %(
                    Benchmarking_folder, pdbid, User_Platform,
                    User_Device.replace(" ","")),"wb") as fn:
            pickle.dump(scipy_eigval -1 ,fn, protocol=4)

    with open("%s/Eigvec_ArpackOri_%s_%s_%s.pkl" %(
                    Benchmarking_folder, pdbid, User_Platform,
                    User_Device.replace(" ","")),"wb") as fn:
            tempeigvec = scipy_eigvec
            tempeigvec = tempeigvec.T
            tempeigvec = tempeigvec.reshape((int(User_n_mode),int(n_atoms),int(3)))
            pickle.dump(tempeigvec[:,:,:] ,fn, protocol=4)


    # ===============
    # Performance
    # ===============
    delta_lambda_list = []
    for jj in range(User_n_mode):
        B = A_scipy@scipy_eigvec[:,jj].T - scipy_eigval[jj] * scipy_eigvec[:,jj].T

        delta_lambda_list.append(np.linalg.norm(B, ord=2))
        print(scipy_eigval[jj], np.linalg.norm(B, ord=2))

    n_atoms = protein_xyz.shape[0]

    
    GPU = "%s %s" %(User_Platform, User_Device.replace(" GPU", ""))

    performance = ["Arpack %s" %(GPU), pdbfn, n_atoms,
                    runtime, peak_mem,
                    User_Platform, User_Device,
                    User_maxleafsize]



    longperformance = []
    for i in range(len(delta_lambda_list)):
        #benchmark_inching.append(performance + [i ,delta_lambda_list[i], eigval[i]])
        longperformance.append(performance + [i ,delta_lambda_list[i], scipy_eigval[i]])

    with open("%s/PerformanceList_ArpackOri_%s_%s_%s.pkl" %(Benchmarking_folder,
        pdbid, User_Platform, User_Device.replace(" ","")),"wb") as fn:
        pickle.dump(longperformance,fn, protocol=4)





    PART05_cleanup = True
    if PART05_cleanup:

        del A_scipy
        del X_df, protein_xyz
        gc.collect()



        #Xnumpy_SparseCupyMatrixUngapppedC.X, Xnumpy_SparseCupyMatrixUngapppedC.X_unsqueezed = None, None
        #del Xnumpy_SparseCupyMatrixUngapppedC.X, Xnumpy_SparseCupyMatrixUngapppedC.X_unsqueezed
        Xnumpy_SparseCupyMatrixUngapppedC = None
        del Xnumpy_SparseCupyMatrixUngapppedC
        eigvec, eigval = None, None
        del eigvec, eigval


