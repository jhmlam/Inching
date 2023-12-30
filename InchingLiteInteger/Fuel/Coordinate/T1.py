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


from collections import defaultdict
import tqdm
import sys
import itertools
import time


import sys
import tqdm
import gc



import numpy as np
#import numba as nb
import scipy
from scipy.spatial import cKDTree
import scipy.stats
import torch
from torch import jit


sys.path.append('..')
sys.path.append('../Script/Burn/')

#import InchingLiteInteger.Fuel.Coordinate.T2



import InchingLiteInteger.util




# =============================
# PBC aware kdtree
# ==============================
class X_cKDTreePbcXy():
    
    def __init__(self, X, User_DictCharmmGuiPbc = {}):
        # NOTE X and User_DictCharmmGuiPbc has to have the same unit which is NM!
        
        # NOTE Dict_Pbc = {} see util.py for structure of the dictionary
        # NOTE The key idea is to search ball point using the periodic image as a query 
        #      The tree is at the unitcell.
        # NOTE As of 2023 Jan the periodic ckdtree on scipy official 
        #      only handles toroidal and ignores the corners. We need to do it ourselves here.
        #      Potentially you can also made this handle the Z direction pbc 
        #      but for now we only focus on membrane systems. You can supply a X_cKDTreePbcXyz easily
        self.atomtree = cKDTree(X, compact_nodes=True, copy_data=False, 
                                    balanced_tree=True, boxsize=None)
        self.User_DictCharmmGuiPbc = User_DictCharmmGuiPbc
        self.BoxsizeVector = np.array([ self.User_DictCharmmGuiPbc['RectBox_Xsize'],
                                        self.User_DictCharmmGuiPbc['RectBox_Ysize'],
                                        self.User_DictCharmmGuiPbc['RectBox_Zsize']])


    def query_ball_point(self, xx, rc_Gamma, p=2., eps=0, 
                        workers=-1, return_sorted=None, return_length=False # NOTE THese are not used but we followed the same flags for coding compatibility only 
                        ):

        # NOTE It is correct iff the PBC is larger than the rc gamma.
        assert (self.User_DictCharmmGuiPbc['RectBox_Xsize'] > rc_Gamma), "ABORTED. The PBC box size X is smaller than rc gamma."
        assert (self.User_DictCharmmGuiPbc['RectBox_Ysize'] > rc_Gamma), "ABORTED. The PBC box size Y is smaller than rc gamma."

        # NOTE Instruction to translate
        instruction = [ np.array([0,0,0]), #central unit
                        np.array([1,0,0]), #xp
                        np.array([-1,0,0]),#xm
                        np.array([0,1,0]), #yp
                        np.array([0,-1,0]), #ym
                        np.array([1,1,0]), #xpyp
                        np.array([1,-1,0]),#xpym
                        np.array([-1,1,0]),#xmyp
                        np.array([-1,-1,0]), #xmym
                        ]
        # Check if any point is at boundary
        if len(xx.shape) == 2:
            xx_is_2d = True
        else:
            xx = xx[np.newaxis,...]
            xx_is_2d = False
            #print(xx, 'newaxis?')
        check_xp = np.sum(xx[:,0] > (self.User_DictCharmmGuiPbc["X"][1] - rc_Gamma))
        check_xm = np.sum(xx[:,0] < (self.User_DictCharmmGuiPbc["X"][0] + rc_Gamma))
        check_yp = np.sum(xx[:,1] > (self.User_DictCharmmGuiPbc["Y"][1] - rc_Gamma))
        check_ym = np.sum(xx[:,1] < (self.User_DictCharmmGuiPbc["Y"][0] + rc_Gamma))



        # NOTE we made the following hardcoded. return_sorted=None, return_length=False
        # NOTE While a < 8/3 times speed up will be achieved with splitting the system into octrant
        #      we abandon the idea for its verbosity. Besides it is only necessary for boundary points 
        #      which are few for a largeg membrane system.
        nnlolol = []
        for i_instruction in range(len(instruction)):
            if i_instruction == 0:
                # NOTE The central cell is always done
                nnlolol.append(
                        self.atomtree.query_ball_point(
                                xx , 
                                rc_Gamma, p=p, eps=eps, workers=-1, 
                                return_sorted=None, return_length=False).tolist()
                                )
            else:
                
                if np.sum(check_xp + check_xm + check_yp + check_ym) > 0:
                    # NOTE if any point is at boundary
                    nnlolol.append(
                        self.atomtree.query_ball_point(
                                xx + (self.BoxsizeVector * instruction[i_instruction]) , 
                                rc_Gamma, p=p, eps=eps, workers=-1, 
                                return_sorted=None, return_length=False).tolist()
                                )
                else:
                    # NOTE It is not at boundary at all! We need not check the pbc!
                    nnlolol.append([[]]*int(xx.shape[0]))

        nnlol_recombined = [a0+a1+a2+a3+a4+a5+a6+a7+a8 for (a0,a1,a2,a3,a4,a5,a6,a7,a8) in zip(*nnlolol)]

        # NOTE scipy cKDtree has this behavior
        if xx_is_2d:
            pass
        else:
            return nnlol_recombined[0] # which is a list instead of lol
            #print(xx, 'newaxis?')

        return nnlol_recombined




# ===========================
# Cuthill related
# =============================
# NOTE A flag is added to handle the pbc

def X_KdCuthillMckeeOrder(  X, 
                            rc_Gamma = 15.0, Reverse = True,
                            ReturnStat = False,
                            User_maxleafsize = 1000,
                            User_DictCharmmGuiPbc = None,
                            ):

    # NOTE Cuthill Mckee on a large coordinate
    #      This function will be done on CPU for simplicity. 
    #      Rather than working on a realised CSR matrix, a k-d tree is used to surrogate memory demand.
    #      The retrieval of neighborhood in k-d tree is O(b log N)

    #      The input is a numpy array (n_atom, 3) interestingly torch also support numpy array as index

    # NOTE Reference
    #      * https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html#scipy.spatial.cKDTree.query_ball_point
    #      * https://github.com/scipy/scipy/blob/main/scipy/sparse/csgraph/_reordering.pyx

    # NOTE Remarks
    #      * It is assumed that the X comes from a bonafide PDB format s.t. it is written in nanometer
    #        nm, otherwise the order will not be correct



    from scipy.spatial import cKDTree

    import numpy as np



    # ============================
    # Preprocessing
    # ============================

    n_atoms = X.shape[0]
    degree = np.zeros(n_atoms, dtype=np.int32)
    order = np.zeros(n_atoms, dtype=np.int32)

    rc_Gamma /= 10.0      # nm



    if User_DictCharmmGuiPbc is None:
        atomtree = cKDTree(X)
    else:

        # NOTE It is correct iff the PBC is larger than the rc gamma.
        assert (User_DictCharmmGuiPbc['RectBox_Xsize'] > rc_Gamma), "ABORTED. The PBC box size X is smaller than rc gamma."
        assert (User_DictCharmmGuiPbc['RectBox_Ysize'] > rc_Gamma), "ABORTED. The PBC box size Y is smaller than rc gamma."

        atomtree = X_cKDTreePbcXy(X, User_DictCharmmGuiPbc = User_DictCharmmGuiPbc)




    # NOTE While storage of neighbor is pseudo linear O(E[N atoms in radius] N_atoms ).
    #      This is still huge memory demand. we will trade off with calcualtino speed. 
    # TODO MinMax Neighbor here
    sttime = time.time()
    jj = 0
    for i in tqdm.tqdm(range(int(n_atoms/User_maxleafsize)+1)):
        start = i*User_maxleafsize
        end   = (i+1)*User_maxleafsize
        nnlol = atomtree.query_ball_point(X[start:end,:], rc_Gamma, p=2., eps=0, workers=-1, return_sorted=None, return_length=False)

        # NOTE Collect some stat
        tempdeg = list(map(lambda n: len(n), nnlol))
        tempdeg = np.array(tempdeg)
        degree[start:end] = tempdeg
        jj += len(nnlol)
    
    print("N_neighbor within %s angstrom Mean %s, Std %s" %(rc_Gamma * 10, np.mean(degree), np.std(degree)))
    print("NN search in %s s" %(time.time() - sttime))

    
    # ============================
    # Cuthill Mckee
    # ============================
    inds = np.argsort(degree)
    rev_inds = np.argsort(inds)
    temp_degrees = np.zeros(np.max(degree), dtype=np.int32)

    N = 0

    # loop over zz takes into account possible disconnected graph.
    for zz in tqdm.tqdm(range(n_atoms)):
        if inds[zz] != -1:   # Do BFS with seed=inds[zz]
            seed = inds[zz]
            order[N] = seed
            N += 1
            inds[rev_inds[seed]] = -1
            level_start = N - 1
            level_end = N

            while level_start < level_end:
                for ii in range(level_start, level_end):
                    i = order[ii]
                    N_old = N

                    # Unvisited neighbors
                    ind = atomtree.query_ball_point(
                        X[i,:], rc_Gamma, p=2., eps=0, workers=-1, return_sorted=True, return_length=False)[::-1]
                    #print(type(ind))

                    for jj in range(len(ind)):
                        j = ind[jj]
                        #print(inds[rev_inds[j]])
                        if inds[rev_inds[j]] != -1: # Unvisited neighbors
                            inds[rev_inds[j]] = -1
                            order[N] = j
                            N += 1

                    # Add values to temp_degrees array for insertion sort
                    level_len = 0
                    for kk in range(N_old, N):
                        temp_degrees[level_len] = degree[order[kk]]
                        level_len += 1
                
                    # Do insertion sort for nodes from lowest to highest degree
                    for kk in range(1,level_len):
                        temp = temp_degrees[kk]
                        temp2 = order[N_old+kk]
                        ll = kk
                        while (ll > 0) and (temp < temp_degrees[ll-1]):
                            temp_degrees[ll] = temp_degrees[ll-1]
                            order[N_old+ll] = order[N_old+ll-1]
                            ll -= 1
                        temp_degrees[ll] = temp
                        order[N_old+ll] = temp2
                
                # set next level start and end ranges
                level_start = level_end
                level_end = N

        if N == n_atoms:
            break

    # return reversed order for RCM ordering and undoordering
    if ReturnStat:
        if Reverse:
            return order[::-1] , np.argsort(order[::-1]), np.mean(degree), np.std(degree)
        else:
            return order, np.argsort(order), np.mean(degree), np.std(degree)
    else:
        if Reverse:
            return order[::-1] , np.argsort(order[::-1])
        else:
            return order, np.argsort(order)







def X_KdUngappedMinMaxNeighbor(  X, 
                            rc_Gamma = 15.0,
                            maxleafsize = 1000,
                            CollectStat = False,
                            User_ReturnHalfNnz = False,
                            User_GapSize = 100,
                            User_DictCharmmGuiPbc = None,
                            SliceForm = True):
    # NOTE Returns a list of tuple rather than just tuple
    from scipy.spatial import cKDTree
    from scipy.sparse import dok_matrix
    import numpy as np


    # NOTE While it will work with any X it is intended that X has been reorderd by cuthill 
    # NOTE Input is a numpy array

    # ============================
    # Preprocessing
    # ============================

    n_atoms = X.shape[0]
    degree = np.zeros(n_atoms, dtype=np.int32)
    order = np.zeros(n_atoms, dtype=np.int32)
    
    rc_Gamma /= 10.0      # nm


    if User_DictCharmmGuiPbc is None:
        atomtree = cKDTree(X)
    else:

        # NOTE It is correct iff the PBC is larger than the rc gamma.
        assert (User_DictCharmmGuiPbc['RectBox_Xsize'] > rc_Gamma), "ABORTED. The PBC box size X is smaller than rc gamma."
        assert (User_DictCharmmGuiPbc['RectBox_Ysize'] > rc_Gamma), "ABORTED. The PBC box size Y is smaller than rc gamma."

        atomtree = X_cKDTreePbcXy(X, User_DictCharmmGuiPbc = User_DictCharmmGuiPbc)
    
    batch_head = []
    PartitionTree = InchingLiteInteger.util.GetPartitionTree(range(n_atoms), maxleafsize = maxleafsize)
    FlattenPartitionTree_generator = InchingLiteInteger.util.FlattenPartitionTree(PartitionTree)
    batch_head = [0]
    # NOTE THe sorted here is necessary as it promote preallocation fo memory
    for i in sorted(FlattenPartitionTree_generator)[::-1]:
        batch_head.append(batch_head[-1] + i)
    


    NnzMinMaxDict = {}
    NnzMinMaxDict_ = {}
    Stat_Number_Batch_gap = defaultdict(int)
    for i in tqdm.tqdm(range(len(batch_head) - 1)):
        Stat_Number_Batch_gap[i] = 0
    Stat_Gap_length = []
    Total_Savings = 0
    Total_RectangleEntries = 0
    Total_NijExpected = 0
    
    for i in tqdm.tqdm(range(len(batch_head) - 1)):
        start = batch_head[i]           
        end   = batch_head[i+1]
        nnlol = atomtree.query_ball_point(X[start:end,:], rc_Gamma, p=2., eps=0, workers=-1, return_sorted=None, return_length=False)


        #if CollectStat:
        for i_nnlol in range(len(nnlol)):
            Total_NijExpected += len(nnlol[i_nnlol])


        batch_height = len(nnlol)
        
        nnlolflat = list(itertools.chain(*nnlol)) # NOTE These are all the columns
        nnlolflat_unique = sorted(set(nnlolflat))

        Total_RectangleEntries += ((max(nnlolflat) - min(nnlolflat) ) * batch_height)
        adjacent_differences = [(yyy - xxx) for (xxx, yyy) in zip(nnlolflat_unique[:-1], nnlolflat_unique[1:])]
        gap_start_end = [min(nnlolflat)]

        for (iii, xxx) in enumerate(adjacent_differences):

            # NOTE Bleeding edges
            if iii == len(nnlolflat_unique)-5:
                continue
            if iii < 5:
                continue
            # NOTE Report index starting gap and the next nnz after gap
            if xxx > User_GapSize: 

                # NOTE Avoid gapping the diagonal. 
                #      This should not happen as we are connected by covavlent bonds.
                #      But for safety we will do it.
                if (gap_start_end[-1] +5 >= start) and (nnlolflat_unique[iii]+1 <= (start + batch_height+5)):
                    #print(i, 'WARNING. An atom is more than rc_Gamma away from all other atoms. You sure your structure is good?')
                    continue

                Stat_Number_Batch_gap[i] += 1
                Stat_Gap_length.append(xxx)
                Total_Savings += xxx * batch_height
                gap_start_end.extend([nnlolflat_unique[iii]+1,nnlolflat_unique[iii+1]]) # NOTE slice form true
                #print('batch i', i)
                #print(nnlolflat_unique[iii],nnlolflat_unique[iii+1] )
        # NOTE if no gap then it still works?
        gap_start_end.append(max(nnlolflat)+1) # NOTE slice form true


        tuple_start_end = []
        for (iii, xxx) in enumerate(gap_start_end):
            if iii%2 == 0:
                tuple_start_end.append((xxx, gap_start_end[iii+1]))
        #print(tuple_start_end)
        NnzMinMaxDict_[i] = tuple_start_end
        #print([iii for (iii, xxx) in enumerate(adjacent_differences) if xxx > User_GapSize])
        #"""
        if SliceForm:
            NnzMinMaxDict[i] = (min(nnlolflat),max(nnlolflat)+1)
        else:
            NnzMinMaxDict[i] = (min(nnlolflat),max(nnlolflat))
        #"""
    # NOTE Assume Poisson process and similar bandwidth per row (i.e. small cornering quadrature), P(island length | batchwidth) ~ Exponential
    try:
        print('Mean number of Gaps > %s is %s. Mean Gap Length Given Gap is %s' 
                    %(User_GapSize, 
                    np.mean(list(Stat_Number_Batch_gap.values())), 
                    np.mean(Stat_Gap_length)))
        print('Max number of Gaps > %s is %s. Max Gap Length Given Gap is %s' 
                    %(User_GapSize, 
                    np.max(list(Stat_Number_Batch_gap.values())), 
                    np.max(Stat_Gap_length)))
        print('Median number of Gaps > %s is %s. Median Gap Length Given Gap is %s' 
                    %(User_GapSize, 
                    np.median(list(Stat_Number_Batch_gap.values())), 
                    np.median(Stat_Gap_length)))
        print('Total Entry Savings %s which is %s percent of a Rectangular Batch' %(Total_Savings, Total_Savings/Total_RectangleEntries*100))
    except:
        print('Ungapping yield no improvement in this case.')


    print("Nnz in Hessian (L+D) is %s. This will occupy %s GB for (L+D) data and at max %s GB for all indexings. Acceptable?" %(
            (((Total_NijExpected - n_atoms)/2) + n_atoms)*9, 
                Total_NijExpected*9/2*8/1024/1024/1024, 
                Total_NijExpected*9/2*8/1024/1024/1024
            ) )
    if User_ReturnHalfNnz:
        return NnzMinMaxDict_, (((Total_NijExpected - n_atoms)/2) + n_atoms)*9
    else:
        return NnzMinMaxDict_






def X_KdMinMaxNeighbor(  X, 
                            rc_Gamma = 15.0,
                            maxleafsize = 1000,
                            CollectStat = False,
                            SliceForm = True):

    from scipy.spatial import cKDTree
    from scipy.sparse import dok_matrix
    import numpy as np


    # NOTE While it will work with any X it is intended that X has been reorderd by cuthill 
    # NOTE Input is a numpy array

    # ============================
    # Preprocessing
    # ============================

    n_atoms = X.shape[0]
    degree = np.zeros(n_atoms, dtype=np.int32)
    order = np.zeros(n_atoms, dtype=np.int32)
    atomtree = cKDTree(X)
    rc_Gamma /= 10.0      # nm


    
    batch_head = []
    PartitionTree = InchingLiteInteger.util.GetPartitionTree(range(n_atoms), maxleafsize = maxleafsize)
    FlattenPartitionTree_generator = InchingLiteInteger.util.FlattenPartitionTree(PartitionTree)
    batch_head = [0]
    # NOTE THe sorted here is necessary as it promote preallocation fo memory
    for i in sorted(FlattenPartitionTree_generator)[::-1]:
        batch_head.append(batch_head[-1] + i)
    


    NnzMinMaxDict = {}
    for i in range(len(batch_head) - 1):
        start = batch_head[i]
        end   = batch_head[i+1]
        nnlol = atomtree.query_ball_point(X[start:end,:], rc_Gamma, p=2., eps=0, workers=-1, return_sorted=None, return_length=False)

        nnlolflat = list(itertools.chain(*nnlol))
        if SliceForm:
            NnzMinMaxDict[i] = (min(nnlolflat),max(nnlolflat)+1)
        else:
            NnzMinMaxDict[i] = (min(nnlolflat),max(nnlolflat))


    if CollectStat:
        print("E[Kissing number in 15 angstrom], Std, Bin Count. Matrix Bandwidth.")


    return NnzMinMaxDict










# ============================
# Dynamics related
# ============================

# NOTE Accept Heigvec[i,:natoms,:3] return with unit magnitude flattened [:n_atoms]
def HeigvecOne_BoxCoxMagnitude( deltaX,
                        User_WinsorizingWindow = (0.025, 0.975),
                        User_LogisticParam = (0.05, 1.0),

                        ):
    # NOTE The distribution of magnitude is often skewed to the small magnitude side i.e. right skewed
    #      But at the same time large magnitude pops up We will use box-cox transform to reduce skewness
    #      The Box cox lambda is a free parameter; note that when lambda --> 0 the transform is log
    #      lambda can be estimated with MLE or a designated 'well-behaved' value 
    #      It maps to -inf, +inf s.t. we can apply e.g. logistic to make it [0,1]
    #      However, lambda from MLE can be harsh. I would still recommend clipping by quantile.

    if torch.is_tensor(deltaX):
        deltaX = deltaX.detach().cpu().numpy()
    else:
        pass
    deltaX_magnitude =  np.sqrt(
                    np.sum( deltaX*  deltaX, axis =1)
                    ).flatten()


    lower_quan = np.quantile(deltaX_magnitude, User_WinsorizingWindow[0])
    upper_quan = np.quantile(deltaX_magnitude, User_WinsorizingWindow[1])

    deltaX_magnitude = np.clip(deltaX_magnitude, lower_quan, upper_quan)
    deltaX_magnitude_, lmax_mle = scipy.stats.boxcox(deltaX_magnitude, lmbda=None, alpha=None, optimizer=None)
    #deltaX_magnitude = (deltaX_magnitude_ ) / (np.std(deltaX_magnitude_)) # NOTE If std is too small overflow

    param_Q = User_LogisticParam[0]
    param_nu = User_LogisticParam[1]
    deltaX_magnitude = 1.0 / np.power((1 + param_Q * np.exp( -1.0 * param_nu * (deltaX_magnitude ) )) , 1.0 / param_nu)
    #deltaX_magnitude = 1.0/np.exp(-1.0 * deltaX_magnitude) # NOTE If deltaX_magnitude is ln(orig) i.e. lambda == 0, then this returns the linear scale
    deltaX_magnitude = (deltaX_magnitude - np.min(deltaX_magnitude) )/ (np.max(deltaX_magnitude) - np.min(deltaX_magnitude))
    #deltaX_magnitude = np.clip(deltaX_magnitude, 0.01, 0.99)

    return deltaX_magnitude



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
