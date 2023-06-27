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
# NOTE This script only collect statistics without realizing the Hessian. 
#      For Hessian realization refer to commandTimeProcessing_*

import glob
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






import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt






import torch





sys.path.append('..')
sys.path.append('../../')
sys.path.append('../InchingLite/')
import InchingLite.util
import InchingLite.Fuel.Coordinate.T1
import InchingLite.Burn.Coordinate.T1
import InchingLite.Burn.Coordinate.T3


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




# A list of pdb available at different sizes
#set_matplotlib_formats('svg')
User_maxleafsize = 100 
User_GapSize = 100 # NOTE This gap refers to the gap inside the matrix not the eigenvalue gap esimate.
User_gamma = 8.0

User_Platform = platform.system() # Windows Darwin Linux
User_Device = "Linux" #"Ryzen5800H" # platform.processor()
Benchmarking_folder = "../MatrixStatistics/Leaf%sGap%s/" %(User_maxleafsize, User_GapSize)
pdbavail = sorted(glob.glob('../DataRepo/PdbByAtomCount/*.pdb')) + sorted(glob.glob('../DataRepo/CifByAtomCount/*.cif')) 
pdbavail = [InchingLite.util.WinFileDirLinux(i) for i in pdbavail]

InchingLite.util.MkdirList(["../MatrixStatistics/", Benchmarking_folder])


benchmark_inching = []
for pdbfn in pdbavail[1:]:

    devices_ = [d for d in range(torch.cuda.device_count())]
    device_names_  = [torch.cuda.get_device_name(d) for d in devices_]
    User_Device =  device_names_[0]


    pdbid = pdbfn.split("/")[-1].split(".")[0]
    #if pdbid == '1a1x':
    #    continue
    if os.path.exists("%s/Statistics_Inching_%s_%s_%s.pkl" %(Benchmarking_folder, pdbid, User_Platform, User_Device.replace(" ","") )):
        continue

    st = time.time()

    X_df, X_top = InchingLite.util.BasicPdbCifLoading(pdbfn)
    protein_xyz = X_df[['x','y','z']].to_numpy().astype(np.float64)
    #print(protein_xyz[:,:])
    protein_xyz -= np.around(protein_xyz.mean(axis= 0), decimals=4)
    #print(protein_xyz[:,:])
    orig_ = protein_xyz
    n_atoms = protein_xyz.shape[0]
    print(pdbid, n_atoms, protein_xyz.shape)

    # ===============================================
    # K-d Cuthill (NOTE CPU np array)
    # ===================================
   

    st = time.time()
    tracemalloc.start()
    # NOTE Cuthill Order and Undo
    cuthill_order, cuthill_undoorder, mean_nn, std_nn = InchingLite.Fuel.Coordinate.T1.X_KdCuthillMckeeOrder(protein_xyz,  
                                rc_Gamma = User_gamma, Reverse = False, ReturnStat= True,
                                )
    protein_xyz_origorder = protein_xyz
    protein_xyz = protein_xyz[cuthill_order,:]

    peak_mem = tracemalloc.get_traced_memory() 
    peak_mem = peak_mem[1] / 1024 / 1024

    tracemalloc.stop()
    runtime = time.time() - st


    # ===================================
    # NOTE Collect bandwidth
    # ===================================

    # NOTE w/ cuthill
    NnzMinMaxDict = InchingLite.Fuel.Coordinate.T1.X_KdUngappedMinMaxNeighbor(protein_xyz, 
                                User_GapSize = User_GapSize,
                                rc_Gamma = User_gamma, 
                                maxleafsize = User_maxleafsize,
                                CollectStat = False,
                                SliceForm= True)    

    # NOTE w/o cuthill
    NnzMinMaxDict_origorder = InchingLite.Fuel.Coordinate.T1.X_KdUngappedMinMaxNeighbor(protein_xyz_origorder, 
                                User_GapSize = User_GapSize,
                                rc_Gamma = User_gamma, 
                                maxleafsize = User_maxleafsize,
                                CollectStat = False,
                                SliceForm= True)    


    batch_head = []
    PartitionTree = InchingLite.util.GetPartitionTree(range(n_atoms), maxleafsize = User_maxleafsize)
    FlattenPartitionTree_generator = InchingLite.util.FlattenPartitionTree(PartitionTree)
    batch_head = [0]
    # NOTE THe sorted here is necessary as it promote preallocation fo memory
    for i in sorted(FlattenPartitionTree_generator)[::-1]:
        batch_head.append(batch_head[-1] + i)


    print('cuthill finished. Now collect nnz')
    matrixstat_df = []
    matrixstat_df_3DRCM_Gapped = []
    matrixstat_df_3DRCM_Ungapped = []
    matrixstat_df_PDB_Gapped = []
    matrixstat_df_PDB_Ungapped = []
    for i in tqdm.tqdm(range(len(batch_head) - 1)):

        # =========================
        # Batch Height
        # ========================
        start = batch_head[i]
        end   = batch_head[i+1]+1
        batchheight = end-start


        # =============================
        # w/o Cuthill
        # =============================
        v_origorder = (NnzMinMaxDict_origorder[i][0][0], NnzMinMaxDict_origorder[i][-1][1])
        bandwidth_origorder = v_origorder[1] - v_origorder[0]
        n_islands_origorder = len(NnzMinMaxDict_origorder[i])
        distribution_islandlength_origorder = [(iii[1] - iii[0]) for iii in NnzMinMaxDict_origorder[i]]
        v_ungapped_origorder = sum(distribution_islandlength_origorder)
        # NOTE Fraction denominator is fixed for both w/ and w/o cuthill this allow comparsion
        area_fullrectagular_origorder = batchheight*bandwidth_origorder
        area_ungapped_origorder = batchheight*v_ungapped_origorder
        fraction_fullrectagular_origorder = area_fullrectagular_origorder / (batchheight* protein_xyz.shape[0])        
        fraction_ungapped_origorder = area_ungapped_origorder / (batchheight* protein_xyz.shape[0])   
        # =============================
        # w/ Cuthill
        # =============================
        v = (NnzMinMaxDict[i][0][0], NnzMinMaxDict[i][-1][1])
        bandwidth_cuthill = v[1] - v[0]
        n_islands_cuthill = len(NnzMinMaxDict[i]) # NOTE we want n_island as small as possible to reduce number of explicit indexing
        distribution_islandlength_cuthill = [(iii[1] - iii[0]) for iii in NnzMinMaxDict[i]]
        v_ungapped_cuthill = sum(distribution_islandlength_cuthill)
        # NOTE Fraction denominator is fixed for both w/ and w/o cuthill this allow comparsion
        area_fullrectagular_cuthill = batchheight*bandwidth_cuthill
        area_ungapped_cuthill = batchheight*v_ungapped_cuthill
        fraction_fullrectagular_cuthill = area_fullrectagular_cuthill / (batchheight* protein_xyz.shape[0])        
        fraction_ungapped_cuthill = area_ungapped_cuthill / (batchheight* protein_xyz.shape[0])   
        

        # =======================================
        # NNZ Common to w/ and w/o cuthill
        # ====================================
        dist = scipy.spatial.distance_matrix(
                protein_xyz[start:end,:], 
                protein_xyz[v[0]:v[1],:], # NOTE we will use the cuthill here but the nnz is the same because we are just doing permutation
                p=2)

        nnz = np.int64(np.sum(dist <= User_gamma/10))

        


        # =================================
        # w/o cuthill
        # =================================
        zerosencountered_fullrectagular_origorder = area_fullrectagular_origorder - nnz
        nnzfraction_fullrectagular_origorder = nnz/area_fullrectagular_origorder       # NOTE Larger the better
        zerosencountered_ungapped_origorder = area_ungapped_origorder - nnz
        nnzfraction_ungapped_origorder = nnz/area_ungapped_origorder

        # =================================
        # w/ cuthill
        # =================================
        zerosencountered_fullrectagular_cuthill = area_fullrectagular_cuthill - nnz
        nnzfraction_fullrectagular_cuthill = nnz/area_fullrectagular_cuthill
        zerosencountered_ungapped_cuthill = area_ungapped_cuthill - nnz
        nnzfraction_ungapped_cuthill = nnz/area_ungapped_cuthill


        # =============================
        # Data lines
        # =============================
        matrixstat_df_3DRCM_Gapped.append(
            [pdbid, protein_xyz.shape[0], 
            User_gamma,
            User_maxleafsize,User_maxleafsize*3,
            batchheight, batchheight *3,
            nnz, nnz*9,

            bandwidth_cuthill, bandwidth_cuthill*3, 
            #bandwidth_origorder, bandwidth_origorder*3,

            zerosencountered_fullrectagular_cuthill, zerosencountered_fullrectagular_cuthill *9,
            #zerosencountered_ungapped_cuthill, zerosencountered_ungapped_cuthill*9,
            nnzfraction_fullrectagular_cuthill, nnzfraction_fullrectagular_cuthill *100,
            #nnzfraction_ungapped_cuthill, nnzfraction_ungapped_cuthill*100,

            #zerosencountered_fullrectagular_origorder, zerosencountered_fullrectagular_origorder *9,
            #zerosencountered_ungapped_origorder, zerosencountered_ungapped_origorder*9,
            #nnzfraction_fullrectagular_origorder, nnzfraction_fullrectagular_origorder *100,
            #nnzfraction_ungapped_origorder, nnzfraction_ungapped_origorder*100,

            area_fullrectagular_cuthill, area_fullrectagular_cuthill *9, 
            #area_ungapped_cuthill, area_ungapped_cuthill*9,
            #area_fullrectagular_origorder, area_fullrectagular_origorder *9, 
            #area_ungapped_origorder, area_ungapped_origorder*9,

            n_islands_cuthill,
            distribution_islandlength_cuthill,
            #n_islands_origorder,
            #distribution_islandlength_origorder,

            mean_nn, std_nn,
            peak_mem, runtime,
            '3DRCM Gapped',
            User_Platform, User_Device.replace(" ",""),
            '3DRCM',
            ]
            ) 

        matrixstat_df_3DRCM_Ungapped.append(
            [pdbid, protein_xyz.shape[0], 
            User_gamma,
            User_maxleafsize,User_maxleafsize*3,
            batchheight, batchheight *3,
            nnz, nnz*9,

            bandwidth_cuthill, bandwidth_cuthill*3, 
            #bandwidth_origorder, bandwidth_origorder*3,

            #zerosencountered_fullrectagular_cuthill, zerosencountered_fullrectagular_cuthill *9,
            zerosencountered_ungapped_cuthill, zerosencountered_ungapped_cuthill*9,
            #nnzfraction_fullrectagular_cuthill, nnzfraction_fullrectagular_cuthill *100,
            nnzfraction_ungapped_cuthill, nnzfraction_ungapped_cuthill*100,

            #zerosencountered_fullrectagular_origorder, zerosencountered_fullrectagular_origorder *9,
            #zerosencountered_ungapped_origorder, zerosencountered_ungapped_origorder*9,
            #nnzfraction_fullrectagular_origorder, nnzfraction_fullrectagular_origorder *100,
            #nnzfraction_ungapped_origorder, nnzfraction_ungapped_origorder*100,

            #area_fullrectagular_cuthill, area_fullrectagular_cuthill *9, 
            area_ungapped_cuthill, area_ungapped_cuthill*9,
            #area_fullrectagular_origorder, area_fullrectagular_origorder *9, 
            #area_ungapped_origorder, area_ungapped_origorder*9,

            n_islands_cuthill,
            distribution_islandlength_cuthill,
            #n_islands_origorder,
            #distribution_islandlength_origorder,

            mean_nn, std_nn,
            peak_mem, runtime,
            '3DRCM Ungapped',
            User_Platform, User_Device.replace(" ",""),
            '3DRCM',
            ]
            ) 

        matrixstat_df_PDB_Gapped.append(
            [pdbid, protein_xyz.shape[0], 
            User_gamma,
            User_maxleafsize,User_maxleafsize*3,
            batchheight, batchheight *3,
            nnz, nnz*9,

            #bandwidth_cuthill, bandwidth_cuthill*3, 
            bandwidth_origorder, bandwidth_origorder*3,

            #zerosencountered_fullrectagular_cuthill, zerosencountered_fullrectagular_cuthill *9,
            #zerosencountered_ungapped_cuthill, zerosencountered_ungapped_cuthill*9,
            #nnzfraction_fullrectagular_cuthill, nnzfraction_fullrectagular_cuthill *100,
            #nnzfraction_ungapped_cuthill, nnzfraction_ungapped_cuthill*100,

            zerosencountered_fullrectagular_origorder, zerosencountered_fullrectagular_origorder *9,
            #zerosencountered_ungapped_origorder, zerosencountered_ungapped_origorder*9,
            nnzfraction_fullrectagular_origorder, nnzfraction_fullrectagular_origorder *100,
            #nnzfraction_ungapped_origorder, nnzfraction_ungapped_origorder*100,

            #area_fullrectagular_cuthill, area_fullrectagular_cuthill *9, 
            #area_ungapped_cuthill, area_ungapped_cuthill*9,
            area_fullrectagular_origorder, area_fullrectagular_origorder *9, 
            #area_ungapped_origorder, area_ungapped_origorder*9,

            #n_islands_cuthill,
            #distribution_islandlength_cuthill,
            n_islands_origorder,
            distribution_islandlength_origorder,

            mean_nn, std_nn,
            peak_mem, runtime,
            'PDB Gapped',
            User_Platform, User_Device.replace(" ",""),
            'PDB',
            ]
            ) 

        matrixstat_df_PDB_Ungapped.append(
            [pdbid, protein_xyz.shape[0], 
            User_gamma,
            User_maxleafsize,User_maxleafsize*3,
            batchheight, batchheight *3,
            nnz, nnz*9,

            #bandwidth_cuthill, bandwidth_cuthill*3, 
            bandwidth_origorder, bandwidth_origorder*3,

            #zerosencountered_fullrectagular_cuthill, zerosencountered_fullrectagular_cuthill *9,
            #zerosencountered_ungapped_cuthill, zerosencountered_ungapped_cuthill*9,
            #nnzfraction_fullrectagular_cuthill, nnzfraction_fullrectagular_cuthill *100,
            #nnzfraction_ungapped_cuthill, nnzfraction_ungapped_cuthill*100,

            #zerosencountered_fullrectagular_origorder, zerosencountered_fullrectagular_origorder *9,
            zerosencountered_ungapped_origorder, zerosencountered_ungapped_origorder*9,
            #nnzfraction_fullrectagular_origorder, nnzfraction_fullrectagular_origorder *100,
            nnzfraction_ungapped_origorder, nnzfraction_ungapped_origorder*100,

            #area_fullrectagular_cuthill, area_fullrectagular_cuthill *9, 
            #area_ungapped_cuthill, area_ungapped_cuthill*9,
            #area_fullrectagular_origorder, area_fullrectagular_origorder *9, 
            area_ungapped_origorder, area_ungapped_origorder*9,

            #n_islands_cuthill,
            #distribution_islandlength_cuthill,
            n_islands_origorder,
            distribution_islandlength_origorder,

            mean_nn, std_nn,
            peak_mem, runtime,
            'PDB Ungapped',
            User_Platform, User_Device.replace(" ",""),
            'PDB',
            ]
            ) 



        
    


    matrixstat_df_3DRCM_Gapped = pd.DataFrame(matrixstat_df_3DRCM_Gapped, columns = [ 'Pdbid', 'Number of atoms', 
                                                            r'Distance ($\AA$)',
                                                            'Max Leaf Size (n)', 'Max Leaf Size (3n)',
                                                            'Batchheight (n)','Batchheight (3n)',
                                                            r'NNZ ($n \times n$)', r'NNZ ($3n \times 3n$)',

                                                            'Bandwidth (n)', 'Bandwidth (3n)',


                                                            r'NZE ($n \times n$)', r'NZE ($3n \times 3n$)',
                                                           
                                                            'NNZ Fraction', 'NNZ Percent',


                                                            r'Batch Area ($n \times n$)', r'Batch Area ($3n \times 3n$)',
                                                            
                                                            'Number of Islands (n)',
                                                            'Island Lengths (n)',
                                                      

                                                            r'Mean Number of Atoms in 8 $\AA$', r'Std Number of Atoms in 8 $\AA$',
                                                            'Peak Memory (MB)', 'Run time (second)',
                                                            'Algorithm',
                                                            'Platform', 'Device',
                                                            'Atom Ordering',
                                                            ])
    matrixstat_df_3DRCM_Ungapped = pd.DataFrame(matrixstat_df_3DRCM_Ungapped, columns = [ 'Pdbid', 'Number of atoms', 
                                                            r'Distance ($\AA$)',
                                                            'Max Leaf Size (n)', 'Max Leaf Size (3n)',
                                                            'Batchheight (n)','Batchheight (3n)',
                                                            r'NNZ ($n \times n$)', r'NNZ ($3n \times 3n$)',

                                                            'Bandwidth (n)', 'Bandwidth (3n)',


                                                            r'NZE ($n \times n$)', r'NZE ($3n \times 3n$)',
                                                           
                                                            'NNZ Fraction', 'NNZ Percent',


                                                            r'Batch Area ($n \times n$)', r'Batch Area ($3n \times 3n$)',
                                                            
                                                            'Number of Islands (n)',
                                                            'Island Lengths (n)',
                                                      

                                                            r'Mean Number of Atoms in 8 $\AA$', r'Std Number of Atoms in 8 $\AA$',
                                                            'Peak Memory (MB)', 'Run time (second)',
                                                            'Algorithm',
                                                            'Platform', 'Device',
                                                            'Atom Ordering',
                                                            ])
    matrixstat_df_PDB_Gapped = pd.DataFrame(matrixstat_df_PDB_Gapped, columns = [ 'Pdbid', 'Number of atoms', 
                                                            r'Distance ($\AA$)',
                                                            'Max Leaf Size (n)', 'Max Leaf Size (3n)',
                                                            'Batchheight (n)','Batchheight (3n)',
                                                            r'NNZ ($n \times n$)', r'NNZ ($3n \times 3n$)',

                                                            'Bandwidth (n)', 'Bandwidth (3n)',


                                                            r'NZE ($n \times n$)', r'NZE ($3n \times 3n$)',
                                                           
                                                            'NNZ Fraction', 'NNZ Percent',


                                                            r'Batch Area ($n \times n$)', r'Batch Area ($3n \times 3n$)',
                                                            
                                                            'Number of Islands (n)',
                                                            'Island Lengths (n)',
                                                      

                                                            r'Mean Number of Atoms in 8 $\AA$', r'Std Number of Atoms in 8 $\AA$',
                                                            'Peak Memory (MB)', 'Run time (second)',
                                                            'Algorithm',
                                                            'Platform', 'Device',
                                                            'Atom Ordering',
                                                            ])
    matrixstat_df_PDB_Ungapped = pd.DataFrame(matrixstat_df_PDB_Ungapped, columns = [ 'Pdbid', 'Number of atoms', 
                                                            r'Distance ($\AA$)',
                                                            'Max Leaf Size (n)', 'Max Leaf Size (3n)',
                                                            'Batchheight (n)','Batchheight (3n)',
                                                            r'NNZ ($n \times n$)', r'NNZ ($3n \times 3n$)',

                                                            'Bandwidth (n)', 'Bandwidth (3n)',


                                                            r'NZE ($n \times n$)', r'NZE ($3n \times 3n$)',
                                                           
                                                            'NNZ Fraction', 'NNZ Percent',


                                                            r'Batch Area ($n \times n$)', r'Batch Area ($3n \times 3n$)',
                                                            
                                                            'Number of Islands (n)',
                                                            'Island Lengths (n)',
                                                      

                                                            r'Mean Number of Atoms in 8 $\AA$', r'Std Number of Atoms in 8 $\AA$',
                                                            'Peak Memory (MB)', 'Run time (second)',
                                                            'Algorithm',
                                                            'Platform', 'Device',
                                                            'Atom Ordering',
                                                            ])

    #TODO Separate as two dfs and concat we will use hue 3DRCM vs PDB / Gapped vs Ungapped
    matrixstat_df = pd.concat([ matrixstat_df_3DRCM_Gapped, matrixstat_df_3DRCM_Ungapped, 
                                matrixstat_df_PDB_Gapped,   matrixstat_df_PDB_Ungapped])


    matrixstat_df.to_pickle("%s/Statistics_Inching_%s_%s_%s.pkl" %(Benchmarking_folder, pdbid, User_Platform, User_Device.replace(" ","")), )
    

    del X_df, protein_xyz#, H_eigval, H_eigvec, eigvec, eigval
    gc.collect()

    InchingLite.util.TorchEmptyCache()

    # Complete Cool Down 
    #if n_atoms > 20000:
    #    time.sleep(200)
