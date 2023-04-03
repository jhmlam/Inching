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
import torch
from torch import jit


sys.path.append('..')
sys.path.append('../Script/Burn/')

#import InchingLite.Fuel.Coordinate.T2



import InchingLite.util



# ======================================
# Ansatz related
# =====================================
# NOTE This provides k means centers to "cluster" the coordinates for local ansatz proposal
def X_KmeansCenters(X, k = 100, 
    MiniBatch = False, # NOTE Advisable not to use minibatch as it may sacrifice accuracy
    InBfsOrder = True
    ):

    from sklearn.cluster import MiniBatchKMeans, KMeans
    #from scipy.spatial import cKDTree
    from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
    from scipy.spatial import distance_matrix

    # NOTE Arranged in BFS order
    if MiniBatch:
        kmeans = MiniBatchKMeans(n_clusters=k, 
                init='k-means++', max_iter=1000, 
                batch_size=1024, verbose=0, compute_labels=True, random_state=None, 
                tol=0.0, max_no_improvement=10, init_size=None, n_init=3, 
                reassignment_ratio=0.01).fit(X)
    else:
        kmeans = KMeans(n_clusters=k,
                init='k-means++', n_init=10, max_iter=1000, 
                tol=0.0001, verbose=0, random_state=None, 
                copy_x=True).fit(X)

    if InBfsOrder:
        candidate_centers = kmeans.cluster_centers_
        D = distance_matrix(
            candidate_centers, candidate_centers, 
            p=2, threshold=1000000)

        Tcsr = minimum_spanning_tree(D).toarray().astype(int)
        # NOTE Choose the starting node with lowest row sum
        startingnodelowest = np.argmin(np.sum(D, axis=1))
        bfs_order = breadth_first_order(
            Tcsr, startingnodelowest, 
            directed=False, return_predecessors=False)

        return candidate_centers[bfs_order]
    else:
        return candidate_centers




# NOTE This provide the local ansatz as a pytorch tensor
def X_AnsatzFromLocalCoord(X, 
                    User_n_overlap = 500, 
                    User_n_mode = 100,
                    User_n_NearestAtoms = 2000,
                    rc_Gamma = 8.0,
                    ReshapeAsRitz = False,
                    extremecase_batchsize = 100000, # NOTE This patches for > 350k atoms. This IS NOT the batch height for sparse matrix vector mult. This is for Ritz vector orthogonalisation 
                    extremecasedefinition = 350000, # NOTE This patches for > 350k atoms
                    device=torch.device(0), 
                    dtype_temp = torch.float64,

                    ):
    
    print("Generating Ansatz from local coordinates.")
    import InchingLite.Burn.LanczosIrlmAnsatz.T1
    import InchingLite.Burn.Coordinate.T1
    from scipy.spatial import cKDTree
    #from scipy.spatial.transform import Rotation as ScipyRotation


    assert (User_n_overlap <= User_n_NearestAtoms), "ABORTED. X_AnsatzFromLocalCoord assume User_n_overlap <= User_n_NearestAtoms"


    protein_xyz = X.detach().cpu().numpy()
    protein_tree = cKDTree(protein_xyz, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)
    n_atoms = protein_xyz.shape[0]
    RecombinedEigvec = np.zeros((User_n_mode, n_atoms, 3))

    if n_atoms < extremecasedefinition:
        R_FullReorthogonalizeAllBeforeClassC = InchingLite.Burn.LanczosIrlmAnsatz.T1.R_FullReorthogonalizeAllBeforeClass()
    else:
        R_FullReorthogonalizeAllBeforeClassC = InchingLite.Burn.LanczosIrlmAnsatz.T1.R_FullReorthogonalizeAllBeforeClassLargeStructure(
            n_atoms *3,
            extremecase_batchsize = extremecase_batchsize,
            device = device, 
            dtype_temp = dtype_temp)





    
    # ==============================
    # Local with overlaps
    # ==============================
    PART00_RoughKMeansLocality = True
    if PART00_RoughKMeansLocality:

        # NOTE Cluster into overlapping centers
        #      Assume that the density of heavy atoms are pretty uniform, 
        #      we will cluster into more-than-sufficient centers

        KmeansCenters = InchingLite.Fuel.Coordinate.T1.X_KmeansCenters(protein_xyz, k = int(X.shape[0]/User_n_overlap)+20, MiniBatch = False ,InBfsOrder = True)
 
 



        
        NearestNeighborEncountered = []
        for center_i in tqdm.tqdm(range(KmeansCenters.shape[0])):

            # NOTE Retrieve nearest neighbor from each center 
            #      Note that         within 8 angstrom there are at max ~100 heavy atoms 
            #                        within 20 angstrom ~2500 atoms

            NearestNeighborDistance_2000 , NearestNeighborIndex_2000 = protein_tree.query(
                                    KmeansCenters[center_i], 
                                    k= User_n_NearestAtoms, 
                                    eps=0, p=2, distance_upper_bound=np.inf, workers=-1)

            # NOTE Calcualtino of eigenproblem
            #      This typically takes 25 seconds for 3000 atoms 10 secons for 2000 atoms
            #      for each cluster obtain 100 eigenvectors.
            # NOTE THe turbo mode returns also the zero-modes
            TurboHeigval, TurboHeigvec, _ = InchingLite.Burn.LanczosIrlmAnsatz.T1.X_TurboHeigval_TurboHeigvec_K(
                            X[NearestNeighborIndex_2000], 
                            n_modes = User_n_mode + 6, # TODO This should be the full Q size as later program
                            dtype_temp = torch.float64, device = torch.device(0),
                            rc_Gamma = rc_Gamma)

            TurboHeigvec = TurboHeigvec.detach().cpu().numpy()
            TurboHeigvec = TurboHeigvec[6:,:,:]
            #print(TurboHeigval[:7], TurboHeigvec[:3,:3,:])
            #sys.exit()

            # NOTE Get the closest e.g. 500 atoms to the center. 
            #      Note the neaestet neigbpr index is consisitent with that of protein tree i.e. proteic xyz
            NearestNeighborIndex_Core = NearestNeighborIndex_2000[:User_n_overlap]
            # NOTE This indexing correspond to turbo ie.. just local 2000 atoms
            NearestNeighborIndexLocal_Core = [ii for ii in range(User_n_NearestAtoms) if NearestNeighborIndex_2000[ii] in NearestNeighborIndex_Core.tolist()]

            # =======================================================
            # Find out sign correlation with previously encountered
            # =====================================================
            # NOTE Calculate the myopic +- direction with the overlapping indices
            if len(NearestNeighborEncountered) > 0:
                OverlapNeighborIndex = sorted(set(NearestNeighborEncountered) & set(NearestNeighborIndex_Core.tolist()))
                if len(OverlapNeighborIndex) == 0:
                    print("Zero Overlap at ", KmeansCenters[center_i])
                    pass
                else:
                    
                    # NOTE Retrieve the overlap in TurboHeigvec
                    OverlapLocal = [] # NOTE consistent with protein_xyz
                    for i_overlap_local in range(len(NearestNeighborIndex_Core)):
                        if NearestNeighborIndex_Core[i_overlap_local] in OverlapNeighborIndex:
                            OverlapLocal.append(i_overlap_local)

                    #print("Number overlap atoms %s" %(len(OverlapNeighborIndex)))

                    # NOTE Implement best greedy sign
                    # NOTE Getting a negative sign means it can be slightly better if we change the sign
                    #      The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.

                    # 3. Unit Direction only
                    normmm = np.sqrt(
                            np.sum(TurboHeigvec[:,OverlapLocal,:]  * TurboHeigvec[:,OverlapLocal,:] , axis = 2)
                            )
                    normmm[normmm == 0.0] = 1.0
                    TurboHeigvecOverlap_unit = TurboHeigvec[:,OverlapLocal,:] / normmm[:,:,np.newaxis]


                    normmm = np.sqrt(
                            np.sum(RecombinedEigvec[:,OverlapNeighborIndex,:]  * RecombinedEigvec[:,OverlapNeighborIndex,:] , axis = 2)
                            )
                    normmm[normmm == 0.0] = 1.0
                    RecombinedEigvecOverlap_unit = RecombinedEigvec[:,OverlapNeighborIndex,:] / normmm[:,:,np.newaxis]

                    cosinecorrelation = InchingLite.Fuel.Coordinate.T2.X_Y_CosineSimilarity(
                        TurboHeigvecOverlap_unit.reshape((User_n_mode,3*len(OverlapLocal))), 
                        RecombinedEigvecOverlap_unit.reshape((User_n_mode,3*len(OverlapLocal)))
                                        )


                    OBSOLETE_Raw_Sign = False
                    if OBSOLETE_Raw_Sign:
                        # 1. Raw 
                        cosinecorrelation = InchingLite.Fuel.Coordinate.T2.X_Y_CosineSimilarity(
                            TurboHeigvec[:,OverlapLocal,:].reshape((User_n_mode,3*len(OverlapLocal))), 
                            RecombinedEigvec[:,OverlapNeighborIndex,:].reshape((User_n_mode,3*len(OverlapLocal)))
                                            )
                        # 2. Sign only                    
                        cosinecorrelation = InchingLite.Fuel.Coordinate.T2.X_Y_CosineSimilarity(
                            np.sign(TurboHeigvec[:,OverlapLocal,:] ).reshape((User_n_mode,3*len(OverlapLocal))), 
                            np.sign(RecombinedEigvec[:,OverlapNeighborIndex,:]).reshape((User_n_mode,3*len(OverlapLocal)))
                                            )
                    
                    cosinecorrelation_abs = np.abs(cosinecorrelation)
                    cosinecorrelation_abs[:6,:] = 0                                        # NOTE Forbid taking the the first six
                    cosinecorrelation_abs[:,:6] = 0     
                    print(np.max(cosinecorrelation_abs, axis=0), 'cosien 3a')
                    print(np.max(cosinecorrelation_abs, axis=1), 'cosine 3b')
                    cosinecorrelation_abs[list(range(6)), list(range(6))] = 1.0
                    #print(cosinecorrelation_abs[list(range(User_n_mode)), list(range(User_n_mode))])
                    # NOTE To enforce one-one lower mode and ignore the correlation
                    #      This is so because the similarity is very low e.g. arccos(0.1) 84 deg vs arccors(0.04) 87 deg. 
                    #      Note that the cosine here is equivalent to average of cosine among each atom 
                    #cosinecorrelation_abs[list(range(User_n_mode)), list(range(User_n_mode))] = 1.0
                    cosinecorrelation_abs = cosinecorrelation_abs.T
                    cosinecorrelation_abs_argmax = np.argmax(cosinecorrelation_abs, axis=1) # NOTE argmax on the new turbo eigvecs
                    cosinecorrelation = cosinecorrelation.T
                    cosinecorrelation_sign = np.sign(cosinecorrelation)
                    
                    sign_instruction = cosinecorrelation_sign[list(range(User_n_mode)), cosinecorrelation_abs_argmax]


                    #print(np.max(cosinecorrelation_abs, axis=0), 'cosien 3a')
                    #print(np.max(cosinecorrelation_abs, axis=1), 'cosine 3b')
                    #print(cosinecorrelation_abs)
                    #print(cosinecorrelation_abs_argmax)

                    sign_instruction[sign_instruction == 0 ] = 1 # NOTE guard the zero just in case


                    TurboHeigvec = sign_instruction[:,np.newaxis,np.newaxis] * TurboHeigvec
                    # NOTE Stitch
                    TurboHeigvec = TurboHeigvec[cosinecorrelation_abs_argmax,:,:]

                    normmm = np.sqrt(
                                            np.sum(TurboHeigvec[:,NearestNeighborIndexLocal_Core,:]  * TurboHeigvec[:,NearestNeighborIndexLocal_Core,:] , axis = 2)
                                            )
                    normmm[normmm == 0.0] = 1.0
                        
                    # NOTE Unit direction only
                    TurboHeigvec_core_unit = TurboHeigvec[:,NearestNeighborIndexLocal_Core,:]  / normmm[:,:,np.newaxis]
                    TurboHeigvec[:,NearestNeighborIndexLocal_Core,:] = TurboHeigvec_core_unit


                    # NOTE Average with the overlap
                    TurboHeigvec[:,OverlapLocal,:] += RecombinedEigvec[:,OverlapNeighborIndex,:]
                    TurboHeigvec[:,OverlapLocal,:] /= 2
            else:
                TurboHeigvec_core_unit = TurboHeigvec[:,NearestNeighborIndexLocal_Core,:]  / np.sqrt(
                                        np.sum(TurboHeigvec[:,NearestNeighborIndexLocal_Core,:]  * TurboHeigvec[:,NearestNeighborIndexLocal_Core,:] , axis = 2)
                                        )[:,:,np.newaxis]
                TurboHeigvec[:,NearestNeighborIndexLocal_Core,:] = TurboHeigvec_core_unit


            # NOTE Update recombined eigvec
            #      :1500 as X is ordered accordingly.
            RecombinedEigvec[:,NearestNeighborIndex_Core,:] = TurboHeigvec[:,NearestNeighborIndexLocal_Core,:] 

            # NOTE Update the Encoutnered
            NearestNeighborEncountered.extend(NearestNeighborIndex_Core.tolist())
            NearestNeighborEncountered = sorted(set(NearestNeighborEncountered))
            del NearestNeighborDistance_2000 , NearestNeighborIndex_2000
            gc.collect()


    # ==============================
    # Patching up the gaps
    # ==============================
    PART01_PatchingUpAnyGaps = True
    if PART01_PatchingUpAnyGaps:
        # NOTE Check if all NN were collected.
        NearestNeighborYetEncountered = sorted(set(list(range(X.shape[0]))) - set(NearestNeighborEncountered))
        print("Remaining Nearest Neighbor Yet Encountered : %s" %(len(NearestNeighborYetEncountered)))

        # TODO Cluster on the nearest neighbor to find an efficient patch
        KmeansCenters = InchingLite.Fuel.Coordinate.T1.X_KmeansCenters(
                                        protein_xyz[NearestNeighborYetEncountered], 
                                        k = int(len(NearestNeighborYetEncountered)/10)+1, 
                                        MiniBatch = False ,InBfsOrder = True)



        # NOTE For the auxiliary just take average of the nearby encountered?

        Number_AuxUpdate = 0
        while len(NearestNeighborYetEncountered) > 0:

            print("Remaining Nearest Neighbor Yet Encountered : %s" %(len(NearestNeighborYetEncountered)))

            protein_remain_tree = cKDTree(protein_xyz[NearestNeighborYetEncountered], 
                                    leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)


            if KmeansCenters.shape[0] > 0:
                _, NearestRemainIndex = protein_remain_tree.query(
                                    KmeansCenters[0], 
                                    k=1, eps=0, p=2, 
                                    distance_upper_bound=np.inf, workers=-1)
                KmeansCenters = np.delete(KmeansCenters, obj= 0, axis = 0) # NOTE pop the first
                NearestRemain_Coord = protein_xyz[NearestNeighborYetEncountered][NearestRemainIndex]
            else:
                NearestRemain_Coord = protein_xyz[NearestNeighborYetEncountered[0]]

                
            NearestNeighborDistance_2000 , NearestNeighborIndex_2000 = protein_tree.query(
                                    NearestRemain_Coord,
                                    k=User_n_NearestAtoms, eps=0, p=2, 
                                    distance_upper_bound=np.inf, workers=-1)

            # NOTE Calcualtino of eigenproblem
            #      This typically takes 25 seconds for 3000 atoms 10 secons for 2000 atoms
            #      for each cluster obtain 100 eigenvectors.
            TurboHeigval, TurboHeigvec, _ = InchingLite.Burn.LanczosIrlmAnsatz.T1.X_TurboHeigval_TurboHeigvec_K(
                            X[NearestNeighborIndex_2000], 
                            n_modes = User_n_mode + 6, # TODO This should be the full Q size as later program
                            dtype_temp = torch.float64, device = torch.device(0),
                            rc_Gamma = rc_Gamma)
            TurboHeigvec = TurboHeigvec[6:,:,:]
            TurboHeigvec = TurboHeigvec.detach().cpu().numpy()




            # NOTE Get the closest e.g. 500 atoms to the center. 
            #      Note the neaestet neigbpr index is consisitent with that of protein tree i.e. proteic xyz




            NearestNeighborIndex_Core = NearestNeighborIndex_2000[:User_n_overlap*2]
            # NOTE This indexing correspond to turbo ie.. just local 2000 atoms
            NearestNeighborIndexLocal_Core = [ii for ii in range(User_n_NearestAtoms) if NearestNeighborIndex_2000[ii] in NearestNeighborIndex_Core.tolist()]

            # =======================================================
            # Find out sign correlation with previously encountered
            # =====================================================
            # NOTE Calculate the myopic +- direction with the overlapping indices
            if len(NearestNeighborEncountered) > 0:
                OverlapNeighborIndex = sorted(set(NearestNeighborEncountered) & set(NearestNeighborIndex_Core.tolist()))
                if len(OverlapNeighborIndex) == 0:
                    #print("Zero Overlap at ", KmeansCenters[center_i])
                    pass
                else:
                    
                    # NOTE Retrieve the overlap in TurboHeigvec
                    OverlapLocal = [] # NOTE consistent with protein_xyz
                    for i_overlap_local in range(len(NearestNeighborIndex_Core)):
                        if NearestNeighborIndex_Core[i_overlap_local] in OverlapNeighborIndex:
                            OverlapLocal.append(i_overlap_local)

                    #print("Number overlap atoms %s" %(len(OverlapNeighborIndex)))

                    # NOTE Implement best greedy sign
                    # NOTE Getting a negative sign means it can be slightly better if we change the sign
                    #      The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.

                    # 3. Unit Direction only
                    TurboHeigvecOverlap_unit = TurboHeigvec[:,OverlapLocal,:] / np.sqrt(
                                        np.sum(TurboHeigvec[:,OverlapLocal,:] * TurboHeigvec[:,OverlapLocal,:], axis = 2)
                                        )[:,:,np.newaxis]
                    RecombinedEigvecOverlap_unit = RecombinedEigvec[:,OverlapNeighborIndex,:] / np.sqrt(
                                        np.sum(RecombinedEigvec[:,OverlapNeighborIndex,:] * RecombinedEigvec[:,OverlapNeighborIndex,:], axis = 2)
                                        )[:,:,np.newaxis]

                    cosinecorrelation = InchingLite.Fuel.Coordinate.T2.X_Y_CosineSimilarity(
                        TurboHeigvecOverlap_unit.reshape((User_n_mode,3*len(OverlapLocal))), 
                        RecombinedEigvecOverlap_unit.reshape((User_n_mode,3*len(OverlapLocal)))
                                        )


                    OBSOLETE_Raw_Sign = False
                    if OBSOLETE_Raw_Sign:
                        # 1. Raw 
                        cosinecorrelation = InchingLite.Fuel.Coordinate.T2.X_Y_CosineSimilarity(
                            TurboHeigvec[:,OverlapLocal,:].reshape((User_n_mode,3*len(OverlapLocal))), 
                            RecombinedEigvec[:,OverlapNeighborIndex,:].reshape((User_n_mode,3*len(OverlapLocal)))
                                            )
                        # 2. Sign only                    
                        cosinecorrelation = InchingLite.Fuel.Coordinate.T2.X_Y_CosineSimilarity(
                            np.sign(TurboHeigvec[:,OverlapLocal,:] ).reshape((User_n_mode,3*len(OverlapLocal))), 
                            np.sign(RecombinedEigvec[:,OverlapNeighborIndex,:]).reshape((User_n_mode,3*len(OverlapLocal)))
                                            )
                    
                    cosinecorrelation_abs = np.abs(cosinecorrelation)
                    cosinecorrelation_abs[:6,:] = 0                                        # NOTE Forbid taking the the first six
                    cosinecorrelation_abs[:,:6] = 0     
                    cosinecorrelation_abs[list(range(6)), list(range(6))] = 1.0
                    #print(cosinecorrelation_abs[list(range(User_n_mode)), list(range(User_n_mode))])
                    # NOTE To enforce one-one lower mode and ignore the correlation
                    #      This is so because the similarity is very low e.g. arccos(0.1) 84 deg vs arccors(0.04) 87 deg. 
                    #      Note that the cosine here is equivalent to average of cosine among each atom 
                    cosinecorrelation_abs[list(range(User_n_mode)), list(range(User_n_mode))] = 1.0
                    cosinecorrelation_abs = cosinecorrelation_abs.T
                    cosinecorrelation_abs_argmax = np.argmax(cosinecorrelation_abs, axis=1) # NOTE argmax on the new turbo eigvecs
                    cosinecorrelation = cosinecorrelation.T
                    cosinecorrelation_sign = np.sign(cosinecorrelation)
                    
                    sign_instruction = cosinecorrelation_sign[list(range(User_n_mode)), cosinecorrelation_abs_argmax]


                    #print(np.max(cosinecorrelation_abs, axis=0), 'cosien 3a')
                    #print(np.max(cosinecorrelation_abs, axis=1), 'cosine 3b')
                    #print(cosinecorrelation_abs)
                    #print(cosinecorrelation_abs_argmax)

                    sign_instruction[sign_instruction == 0 ] = 1 # NOTE guard the zero just in case


                    TurboHeigvec = sign_instruction[:,np.newaxis,np.newaxis] * TurboHeigvec
                    # NOTE Stitch
                    TurboHeigvec = TurboHeigvec[cosinecorrelation_abs_argmax,:,:]


                    normmm = np.sqrt(
                                            np.sum(TurboHeigvec[:,NearestNeighborIndexLocal_Core,:]  * TurboHeigvec[:,NearestNeighborIndexLocal_Core,:] , axis = 2)
                                            )
                    normmm[normmm == 0.0] = 1.0

                    # NOTE Unit direction only
                    TurboHeigvec_core_unit = TurboHeigvec[:,NearestNeighborIndexLocal_Core,:]  / normmm[:,:,np.newaxis]
                    TurboHeigvec[:,NearestNeighborIndexLocal_Core,:] = TurboHeigvec_core_unit


                    # NOTE Average with the overlap
                    TurboHeigvec[:,OverlapLocal,:] += RecombinedEigvec[:,OverlapNeighborIndex,:]
                    TurboHeigvec[:,OverlapLocal,:] /= 2
            else:
                # NOTE Unit direction only
                TurboHeigvec_core_unit = TurboHeigvec[:,NearestNeighborIndexLocal_Core,:]  / np.sqrt(
                                        np.sum(TurboHeigvec[:,NearestNeighborIndexLocal_Core,:]  * TurboHeigvec[:,NearestNeighborIndexLocal_Core,:] , axis = 2)
                                        )[:,:,np.newaxis]
                TurboHeigvec[:,NearestNeighborIndexLocal_Core,:] = TurboHeigvec_core_unit

            # NOTE Update recombined eigvec
            #      :1500 as X is ordered accordingly.
            RecombinedEigvec[:,NearestNeighborIndex_Core,:] = TurboHeigvec[:,NearestNeighborIndexLocal_Core,:] 

            # NOTE Update the Encoutnered
            NearestNeighborEncountered.extend(NearestNeighborIndex_Core.tolist())
            NearestNeighborEncountered = sorted(set(NearestNeighborEncountered))

            NearestNeighborYetEncountered = sorted(set(list(range(X.shape[0]))) - set(NearestNeighborEncountered))

            del NearestNeighborDistance_2000 , NearestNeighborIndex_2000
            gc.collect()
            Number_AuxUpdate +=1
        print(Number_AuxUpdate, 'total aux update')






    # ====================
    # Tidy Up the ansatz
    # =====================
    PART02_TidyUpAnsatz = True
    if PART02_TidyUpAnsatz:


        # NOTE Unit vector
        normmm = np.sqrt(
                np.sum(RecombinedEigvec  * RecombinedEigvec, axis = 2)
                )
        RecombinedEigvec = RecombinedEigvec  / normmm[:,:,np.newaxis]

        # NOTE Get the rot trans here
        RotTransAnsatz = InchingLite.Burn.Coordinate.T1.X_TransRotRitz(X,
                    device=device, 
                    dtype_temp = dtype_temp,
                    )
        # NOTE Converted to long form ritz before reorthog
        RecombinedEigvec = np.concatenate(
            (RotTransAnsatz.detach().cpu().numpy(), 
            RecombinedEigvec.reshape((User_n_mode, n_atoms*3)).T), 
            axis=1)

        RecombinedEigvec = torch.tensor(RecombinedEigvec, dtype=dtype_temp, device=device)

        # NOTE Full Reorthogonalise
        for jj in range(User_n_mode+6):
            if jj <= 5:
                continue
            RecombinedEigvec = R_FullReorthogonalizeAllBeforeClassC.forward(RecombinedEigvec, 0+jj, 0+jj-1)

        RecombinedEigvec = RecombinedEigvec.T.reshape((User_n_mode+6, n_atoms,3))

    # ======================
    # Save and Reshape 
    # ======================





    if ReshapeAsRitz:
        return RecombinedEigvec.reshape((User_n_mode+6, n_atoms*3)).T # NOTE Our ritz vector have a shape.
    else:
        return RecombinedEigvec


# NOTE Patchy Surface Normals
def X_PatchySurfaceNormals(protein_xyz, 
                    User_InnerOnion_n_NearestAtoms = 256,
                    ):
    

    # NOTE This provides patchy surface normals to be rotated by the network.
    #      https://pcl.readthedocs.io/projects/tutorials/en/latest/don_segmentation.html#don-segmentation
    #      http://graphics.stanford.edu/courses/cs164-10-spring/Handouts/papers_gumhold.pdf

    #      A small inner onion neighborhood will make the surface normals fluctuate quite a lot.
    #      I would recommend to work on 64-128 neighbors with consideration on tradeoff in memory. 

    """
    # How to use it
    PatchySurfaceNormal = X_PatchySurfaceNormals(X, 
                    User_InnerOnion_n_NearestAtoms = 256,
                    )

    InchingLite.util.ShowOneMode(PatchySurfaceNormal[0,:,:], X)
    """


    from scipy.spatial import cKDTree

    PART00_OnionDefinitions = True
    if PART00_OnionDefinitions:
        # NOTE We will define the sign of the normals by taking an vector poiniting from center
        #      of a outer neighborhood. This produce a deterministic stable signed normals
        #      regardless of rotations.
        User_OnionCenter = True
        User_InnerOnion_n_NearestAtoms += 1 # NOTE because of self
        if User_OnionCenter:
            k_NearestAtomsRun = User_InnerOnion_n_NearestAtoms * 2
        else:
            k_NearestAtomsRun = User_InnerOnion_n_NearestAtoms

    
    print("Generating Patchy Surface Normals from coordinates")

    PART01_GetNeighborhood = True
    if PART01_GetNeighborhood:
        if torch.is_tensor(protein_xyz):
            protein_xyz = protein_xyz.detach().cpu().numpy()

        protein_xyz -= np.mean(protein_xyz,axis=0)
        protein_tree = cKDTree(protein_xyz, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)
        nearestneighbor_distance, nearestneighbor_index = protein_tree.query(protein_xyz, k = k_NearestAtomsRun, workers=-1) 
        del nearestneighbor_distance 
        gc.collect()

    PART02_GetPsnPerAtom = True
    if PART02_GetPsnPerAtom:
        n_atoms = protein_xyz.shape[0]
        PatchyNormal = np.zeros((n_atoms, 3))
        for i in tqdm.tqdm(range(n_atoms)):
            local_centered_xyz = protein_xyz[
                nearestneighbor_index[i, :User_InnerOnion_n_NearestAtoms],:] - np.mean(
                    protein_xyz[nearestneighbor_index[i, :User_InnerOnion_n_NearestAtoms],:], axis=0
                    ) # NOTE Inner Onion Center

            U, Sigma, Vt = np.linalg.svd(local_centered_xyz, full_matrices=False, compute_uv=True, hermitian=False)

            # NOTE Vt[2,:] refers to the third column of V[] due to its already transpose. 
            surfacenormal = Vt[2,:] / np.linalg.norm(Vt[2,:])

            # NOTE We want the third component to point away from protein_xyz[i,:]
            #      Note that the view center is taken from the Outer Onion
            if User_OnionCenter:
                center_to_i_vec = protein_xyz[i,:] - np.mean(
                    protein_xyz[nearestneighbor_index[i],:], 
                    axis=0) # NOTE Outer onion centers which is all the rest i nthe k nn searcg
            else:
                center_to_i_vec = protein_xyz[i,:]
                
            center_to_i_vec = center_to_i_vec /np.linalg.norm(center_to_i_vec, ord=2)
            
            sign_instruction = np.sign(
                np.dot(
                    center_to_i_vec, 
                    surfacenormal
                )
                )
            PatchyNormal[i,:] = sign_instruction * surfacenormal

    PatchyNormal = PatchyNormal.reshape((1,n_atoms, 3))
    return PatchyNormal #torch.tensor(PatchyNormal, dtype = dtype_temp, device = device)



# ============================
# GNM
# ============================
def Xnumpy_ScipyTurboGnmEigval_ScipyTurboGnmEigvec(protein_xyz, 
                            User_rc_Gamma = 5.5, # NOTE in angstrom
                            User_n_mode = 10, 
                            User_ScipyArpackTol= 1e-12,
                            device = torch.device(0),
                            dtype_temp = torch.double):

    # NOTE This can be used as a subroutine to obtain GNM eigenvector and 
    #      subsequently the 3D heat kernel signature, which is a useful 
    #      estimate of NMA/ANM magnitude.
    # TODO Find a rc_Gamma that best fit ANM magnitude 


    """
    # How to use it?
    # NOTE With a tolerance at 1e-16
    #      1000k atoms takes 3000 seconds (50 min) and 5 GB RAM on laptop.
    #      2000k atoms takes 9000 seconds (150 min) and 10GB RAM on laptop.
    #      Highly doable, seemingly we can use this strategy to train things 

    # NOTE Lowering tolerance may result in wrong arpack eigenvalue e.g. having strong negatives for a laplacian...
    #      Also note that the lowest eigenvalue of a laplacian must be 0 because of its structure.
    Gnmval, Gnmvec = Xnumpy_ScipyTurboGnmEigval_ScipyTurboGnmEigvec(protein_xyz,
                            User_rc_Gamma = 8,
                            User_n_mode = 100, 
                            User_ScipyArpackTol= 0)
    """

    from scipy.spatial import cKDTree
    from scipy import sparse

    User_rc_Gamma /= 10
    if torch.is_tensor(protein_xyz):
        protein_xyz = protein_xyz.detach().cpu().numpy()

    
    protein_xyz -= np.mean(protein_xyz, axis = 0)
    n_atoms = protein_xyz.shape[0]
    protein_tree = cKDTree(protein_xyz, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)
    result_indexes = protein_tree.query_ball_tree(protein_tree, User_rc_Gamma, p=2., eps=0)

    _Edges = []
    for i in range(len(result_indexes)):
        for j in result_indexes[i]:
            if i >= j:
                continue
            _Edges.append([i,j])

    _Edges = np.array( _Edges, dtype = np.int32 )
    print("Average edge number", _Edges.shape[0]*2 / protein_xyz.shape[0])
    
    st = time.time()
    # NOTE Make laplacian sparse
    L = sparse.coo_matrix((
                np.ones(_Edges.shape[0]),           # NOTE values == 1
                (_Edges.T[0], _Edges.T[1])), 
        shape=(protein_xyz.shape[0], protein_xyz.shape[0])).tocsr()
    L += L.T

    # NOTE Diagonal
    L = sparse.dia_matrix((L.sum(1).flatten() +1,  # NOTE A + I trick
                            0), L.shape) - L
    #print(time.time() - st)
    #print(L[:10,:10])
    st = time.time()
    # NOTE Find n_modes
    laplacian_eigval, laplacian_eigvec = sparse.linalg.eigsh(L,
                                                k=User_n_mode, 
                                                M=None, 
                                                sigma=None,             # NOTE search around Close to 1
                                                which='SM',             # NOTE smallest only
                                                v0=None, ncv=None, maxiter=None, tol=User_ScipyArpackTol, 
                                                return_eigenvectors=True, 
                                                Minv=None, OPinv=None, mode='normal')

    laplacian_eigval -= 1 # NOTE A + I trick
    print("Time to find Gnm amounts to %s seconds"%(time.time() - st))

    return laplacian_eigval, laplacian_eigvec #torch.tensor(laplacian_eigval, device = device, dtype= dtype_temp) , torch.tensor(laplacian_eigvec, device = device, dtype= dtype_temp)



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
                        workers=1, return_sorted=None, return_length=False # NOTE THese are not used but we followed the same flags for coding compatibility only 
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
                                rc_Gamma, p=p, eps=eps, workers=1, 
                                return_sorted=None, return_length=False).tolist()
                                )
            else:
                
                if np.sum(check_xp + check_xm + check_yp + check_ym) > 0:
                    # NOTE if any point is at boundary
                    nnlolol.append(
                        self.atomtree.query_ball_point(
                                xx + (self.BoxsizeVector * instruction[i_instruction]) , 
                                rc_Gamma, p=p, eps=eps, workers=1, 
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
    from scipy.sparse import dok_matrix
    import numpy as np

    #import multiprocessing 

    # ============================
    # Preprocessing
    # ============================

    n_atoms = X.shape[0]
    degree = np.zeros(n_atoms, dtype=np.int64)
    order = np.zeros(n_atoms, dtype=np.int64)

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
    jj = 0
    for i in tqdm.tqdm(range(int(n_atoms/1000)+1)):
        start = i*1000
        end   = (i+1)*1000 
        nnlol = atomtree.query_ball_point(X[start:end,:], rc_Gamma, p=2., eps=0, workers=1, return_sorted=None, return_length=False)

        # NOTE Collect some stat
        tempdeg = list(map(lambda n: len(n), nnlol))
        tempdeg = np.array(tempdeg)
        degree[start:end] = tempdeg
        jj += len(nnlol)
    
    print("N_neighbor within %s angstrom Mean %s, Std %s" %(rc_Gamma * 10, np.mean(degree), np.std(degree)))


    
    # ============================
    # Cuthill Mckee
    # ============================
    inds = np.argsort(degree)
    rev_inds = np.argsort(inds)
    temp_degrees = np.zeros(np.max(degree), dtype=np.int64)

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
                    ind = atomtree.query_ball_point(X[i,:], rc_Gamma, p=2., eps=0, workers=1, return_sorted=True, return_length=False)[::-1]
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
    degree = np.zeros(n_atoms, dtype=np.int64)
    order = np.zeros(n_atoms, dtype=np.int64)
    
    rc_Gamma /= 10.0      # nm


    if User_DictCharmmGuiPbc is None:
        atomtree = cKDTree(X)
    else:

        # NOTE It is correct iff the PBC is larger than the rc gamma.
        assert (User_DictCharmmGuiPbc['RectBox_Xsize'] > rc_Gamma), "ABORTED. The PBC box size X is smaller than rc gamma."
        assert (User_DictCharmmGuiPbc['RectBox_Ysize'] > rc_Gamma), "ABORTED. The PBC box size Y is smaller than rc gamma."

        atomtree = X_cKDTreePbcXy(X, User_DictCharmmGuiPbc = User_DictCharmmGuiPbc)
    
    batch_head = []
    PartitionTree = InchingLite.util.GetPartitionTree(range(n_atoms), maxleafsize = maxleafsize)
    FlattenPartitionTree_generator = InchingLite.util.FlattenPartitionTree(PartitionTree)
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
        nnlol = atomtree.query_ball_point(X[start:end,:], rc_Gamma, p=2., eps=0, workers=1, return_sorted=None, return_length=False)


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
    degree = np.zeros(n_atoms, dtype=np.int64)
    order = np.zeros(n_atoms, dtype=np.int64)
    atomtree = cKDTree(X)
    rc_Gamma /= 10.0      # nm


    
    batch_head = []
    PartitionTree = InchingLite.util.GetPartitionTree(range(n_atoms), maxleafsize = maxleafsize)
    FlattenPartitionTree_generator = InchingLite.util.FlattenPartitionTree(PartitionTree)
    batch_head = [0]
    # NOTE THe sorted here is necessary as it promote preallocation fo memory
    for i in sorted(FlattenPartitionTree_generator)[::-1]:
        batch_head.append(batch_head[-1] + i)
    


    NnzMinMaxDict = {}
    for i in range(len(batch_head) - 1):
        start = batch_head[i]
        end   = batch_head[i+1]
        nnlol = atomtree.query_ball_point(X[start:end,:], rc_Gamma, p=2., eps=0, workers=1, return_sorted=None, return_length=False)

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


def HeigvecOne_RecipLogHeigvec( deltaX,
                        User_WinsorizingWindow = (0.025, 0.975),
                        
    ):


    if torch.is_tensor(deltaX):
        deltaX = deltaX.detach().cpu().numpy()
    else:
        pass
    deltaX_magnitude =  np.sqrt(
                    np.sum( deltaX*  deltaX, axis =1)
                    ).flatten()

    deltaX_magnitude = -1.0/( np.log10(deltaX_magnitude))
    lower_quan = np.quantile(deltaX_magnitude, User_WinsorizingWindow[0])
    upper_quan = np.quantile(deltaX_magnitude, User_WinsorizingWindow[1])
    deltaX_magnitude = np.clip(deltaX_magnitude, lower_quan, upper_quan)
    deltaX_magnitude = (deltaX_magnitude - (np.min(deltaX_magnitude)) )/ (np.max(deltaX_magnitude) - np.min(deltaX_magnitude))


    return  deltaX_magnitude




# =================================
# OBSOLETE
# =================================

# NOTE Obsolete GNM. The two routines below are underperformant. It can be more tightly integrated with the use of kdtree above
def Xnumpy_Dnumpy(X):
    n_atoms = X.shape[0]
    
    # Gram
    G = np.matmul(X, X.T)

    # Distance
    g_1 = np.matmul(np.diag(G, diagonal=0).unsqueeze(0).T, np.ones(1, n_atoms))
    R = g_1 + g_1.T - 2*G

    # NOTE This is nm squared. Below I convert it to the euclidean form in nm
    R = np.sqrt(R)#*10

    return R

# NOTE This is the BIG case Gamma in 2007 Bahar i.e. Laplacian a.k.a. Kirchoff in GNM
def Dnumpy_K(R, rc_Gamma = 1.0, User_sparse_form = True):
    """kirchoff matrix is the connectivity matrix
       diagonal gives 
       offdiag gives adjacency matrix  
       R is the EDM m*m matrix
    """
    # The given matrix should be a EDM
    K = np.zeros((R.shape[0],R.shape[1]))#
    #K[R > rc_Gamma] = 0.0
    K[R <= rc_Gamma] = -1.0
    #K = K.fill_diagonal_(0.0)
    #K_offdiagsum = torch.sum(K,1) # NOTE the diagonal is positive
    K -= np.diag(np.sum(K,axis=1), diagonal=0)
    K = K.astype(np.int64)
    if User_sparse_form:
        K = scipy.sparse.csr_matrix(K)

    return K

