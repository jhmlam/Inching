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


import torch
import sys
import tqdm
sys.path.append('..')





import InchingLiteInt64.util


# ====================================================
# Misc 
# =================================================
# NOTE Effcient build of whole hessian matrix
@torch.no_grad()
def X_D_K_Hessian(X,  D, Gamma, 
            maxleafsize = 100, PlusI = 0.0, dtype_temp = torch.float32):

    torch.no_grad()
    InchingLiteInt64.util.TorchMakePrecision(Precision = str(X.dtype))
    InchingLiteInt64.util.TorchEmptyCache()

    dtype_orig = X.dtype
    
    Gamma = Gamma.type(dtype_orig)
    D = D.type(dtype_orig)

    n_atoms = X.shape[0]    

    dof = 3* n_atoms
    n_nonzero_modes = dof -6

    if n_atoms > 5000:
       print("Warning. GPU at risk of memory overflow X_D_K_Hessian not recommended for system > 5000 atoms")

    # ========================================
    # Precalculate constant
    # ========================================
    # NOTE The D = 0.0 is self atom. In the Hessian the diagonal will be a sum from atom row so set 0.0
    constant = -1. * Gamma/(D**2)
    constant = torch.nan_to_num(constant, nan = 0.0, posinf=0.0, neginf= 0.0)

    # ==================================
    # Batch size calculation
    # ==================================
    PartitionTree = InchingLiteInt64.util.GetPartitionTree(range(n_atoms), maxleafsize = maxleafsize)
    FlattenPartitionTree_generator = InchingLiteInt64.util.FlattenPartitionTree(PartitionTree)
    batch_head = [0]
    for i in FlattenPartitionTree_generator:
        batch_head.append(batch_head[-1] + i)

    # ==================================
    # Version 3 of Hessian Making
    # ===================================
    # NOTE Highly memory demanding. 
    hessian_torch_put = torch.zeros((dof,dof), device = torch.device(0)).type(dtype_temp)
    for i in range(len(batch_head)-1):

        # ==============================================
        # On-demand synthesis of Hessian
        # ==============================================

        # Batching
        Xij_batch = X[:, :] - X[batch_head[i]:batch_head[i+1],:].unsqueeze(1)
        einsum = torch.einsum('bij,bik->bijk', (Xij_batch,Xij_batch))
        einsum = einsum * constant[batch_head[i]:batch_head[i+1],:].unsqueeze(2).unsqueeze(2)

        # Putting the diagonal as in Lezon p.136 eq 7.17
        #"""
        jj = 0
        for j in range(batch_head[i],batch_head[i+1]):
            einsum[jj,j,:,:] -= torch.sum(einsum[jj,:,:,:], axis=0)
            jj +=1
        #"""

        n_einsum_rows = einsum.shape[0]
        temp_index_ii = torch.arange(n_einsum_rows, dtype= torch.long)
        temp_index_jj = torch.arange(batch_head[i], batch_head[i+1], dtype= torch.long)    


        # TODO PlusI
        einsum[temp_index_ii, temp_index_jj,0,0] += PlusI* 1.0
        einsum[temp_index_ii, temp_index_jj,1,1] += PlusI* 1.0
        einsum[temp_index_ii, temp_index_jj,2,2] += PlusI* 1.0

        einsum_nrow, einsum_ncol = einsum.shape[0], einsum.shape[1]
        einsum = einsum.permute(1,3,0,2).contiguous().view(einsum_ncol, 3, einsum_nrow*3)
        einsum = einsum.permute(2,0,1).contiguous().view( einsum_nrow*3 , einsum_ncol*3)
        hessian_torch_put[batch_head[i]*3:batch_head[i+1]*3,:] += einsum
        
    #print(hessian_torch_put.max(), hessian_torch_put.min())
    #InchingLiteInt64.util.ShowImageGrid(hessian_torch_put.unsqueeze(0).unsqueeze(0), num_images=1, precision = 0.1, nrow = 1)
    del einsum
    InchingLiteInt64.util.TorchEmptyCache()

    return hessian_torch_put
    old_versions = False
    if old_versions:
        # ==========================
        # Version 2
        # ==========================
        # NOTE The co-existance of triu and hessian make memory demand severe...

        # 1. Precalculate the difference in atoms coord
        Xij = torch.zeros((n_atoms, n_atoms, 3), device = device)
        for i in range(n_atoms):
            Xij[i,:] = X[:,:] - X[i,:]

        InchingLiteInt64.util.TorchEmptyCache()


        # 2. To create upper triangle index
        # TODO We should do this in batch to reduce stress in memory
        InchingLiteInt64.util.TorchEmptyCache()

        triu_indices = torch.triu_indices(n_atoms, n_atoms)   
        triu_indices_element = triu_indices.T.unsqueeze(1).unsqueeze(1).repeat(1,3,3,1)*3    
        InchingLiteInt64.util.TorchEmptyCache()

        # Columns
        triu_indices_element[:,:,1,1] +=1
        triu_indices_element[:,:,2,1] +=2
        # Rows
        triu_indices_element[:,1,:,0] +=1
        triu_indices_element[:,2,:,0] +=2
        # This is the element in the upper triangular of hessian.
        triu_indices_element = torch.flatten(triu_indices_element, start_dim=0, end_dim=-2).t() 

        # 3. Extract the upper triangle of difference tensor
        
        InchingLiteInt64.util.TorchEmptyCache()

        Xij = Xij[triu_indices[0], triu_indices[1], :]
        # Another way to do it
        #st = time.time()
        #res = torch.bmm(Xij.unsqueeze(2), Xij.unsqueeze(1))
        #print(st - time.time())

        #st = time.time()
        einsum = torch.einsum('bi,bj->bij', (Xij,Xij))
        del Xij
        InchingLiteInt64.util.TorchEmptyCache()

        constant = -1. * Gamma/(D**2)
        constant[constant == float("-inf")] = 0.
        constant = constant[triu_indices[0], triu_indices[1]]
        constant = constant.unsqueeze(1).unsqueeze(1).repeat(1,3,3)
        einsum = constant*einsum
        print("Time consumed in Calculating Xij", time.time()- st)
        del constant, triu_indices


        # Getting the hessian finally
        InchingLiteInt64.util.TorchEmptyCache()

        # NOTE This version do it by put. 0.8 second for a (2500*3)^2 matrix, but consider you have 100 proteins in one epoch...
        #      but still greater than all versions below (and also numpy in cpu of course!) 
        st = time.time()
        hessian_torch_put = torch.zeros((dof,dof), device = torch.device(0))
        hessian_torch_put.index_put_( tuple(triu_indices_element),  
                            torch.flatten(einsum, start_dim=0, end_dim=-1)) 
        # Symmetrize
        hessian_torch_put = torch.triu(hessian_torch_put).T + hessian_torch_put
        # diagonal
        for j in range(n_atoms):
            ii = j*3
            ii3 = j*3 + 3
            jj = j*3
            jj3 = j*3+3

            hessian_torch_put[ii:ii3, ii:ii3 ] -=  torch.sum(hessian_torch_put[:, ii:ii3 ].reshape(n_atoms,3,3),0)

        del triu_indices_element,  einsum, Gamma, D
        InchingLiteInt64.util.TorchEmptyCache()

        # ==========================
        # Version 1
        # ==========================
        # This version do it by sum. 70 seconds for a (2500*3)^2 matrix, but consider you have 100 proteins in one epoch...
        st = time.time()
        hessian_torch = torch.zeros((dof,dof), device = torch.device(0))
        ein_k = 0
        for i in triu_indices_:
            ii = i[0]*3
            ii3 = i[0]*3 + 3
            jj = i[1]*3
            jj3 = i[1]*3+3

            hessian_torch[ii:ii3, jj:jj3 ] += einsum[ein_k,:,:] 
            #hessian_torch[jj:jj3, ii:ii3 ] += einsum[ein_k,:,:] 

            ein_k+=1

        # Symmetrize
        hessian_torch = torch.triu(hessian_torch).T + hessian_torch

        # Get the diagonal row summed
        for j in range(n_atoms):
            ii = j*3
            ii3 = j*3 + 3
            jj = j*3
            jj3 = j*3+3

            hessian_torch[ii:ii3, ii:ii3 ] -=  torch.sum(hessian_torch[:, ii:ii3 ].reshape(n_atoms,3,3),0)
        #print(torch.sort(torch.real(torch.linalg.eigvals(hessian_torch)))[0])
        print("v1", time.time()-st)

        #print(torch.cuda.memory_summary())
        # ===========================
        # Version 0
        # ===========================
        # I will do the slow version here for correctness benchmark 360 seconds for (2500*3)^2 matrix...
        st = time.time()
        hessian = torch.zeros((dof, dof), device=device)
        for i in tqdm.tqdm(range(n_atoms)):
            for j in range(n_atoms):
                if i >= j:
                       continue
                res_i3 = i*3
                res_i33 = res_i3+3
                res_j3 = j*3
                res_j33 = res_j3+3

                i2j = X[j] - X[i]

                if i == j:
                     constant = 0.
                else:
                    constant = -1. * Gamma[i,j]/D[i,j]

                super_element = torch.outer(i2j, i2j) * constant# * (- g / dist2)

                # The ij and ji of hessian are the same,
                hessian[res_i3:res_i33, res_j3:res_j33] = super_element
                hessian[res_j3:res_j33, res_i3:res_i33] = super_element

                # The diagonal is similar to gamma where the off diagonals are subtracted from it
                hessian[res_i3:res_i33, res_i3:res_i33] = \
                    hessian[res_i3:res_i33, res_i3:res_i33] - super_element
                hessian[res_j3:res_j33, res_j3:res_j33] = \
                    hessian[res_j3:res_j33, res_j3:res_j33] - super_element

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