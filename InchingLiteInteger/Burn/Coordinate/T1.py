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
import pickle
sys.path.append('..')





import InchingLiteInteger.util



# ===================
# Coordinate Related
# ===================


@torch.no_grad()
def X_Xcentered(X, device = torch.device(0)):
    return X-torch.mean(X, axis=0)



# NOTE This is just a template 
@torch.no_grad()
def X_Dbatched(X, maxleafsize = 100, rc_Gamma = 15.0):
    # ==================================
    # Batch size calculation
    # ==================================
    n_atoms = X.shape[0]    
    PartitionTree = InchingLiteInteger.util.GetPartitionTree(range(n_atoms), maxleafsize = maxleafsize)
    FlattenPartitionTree_generator = InchingLiteInteger.util.FlattenPartitionTree(PartitionTree)
    batch_head = [0]
    for i in sorted(FlattenPartitionTree_generator)[::-1]:
        batch_head.append(batch_head[-1] + i)


    X = X.type(torch.float32)
    InchingLiteInteger.util.TorchMakePrecision(Precision = str(X.dtype))
    # n-th row X * n-th column X.T is simply the magnitude 
    g_1 = torch.sum(X * X, axis =1)
    for i in range(len(batch_head)-1):

        # ==========================================
        # On-demand realization of Constant Gamma/D
        # ==========================================
        # NOTE in a cycle of 1000 call of this function it adds 5.5 seconds...
        # NOTE Batching of making distance from gram matrix
        R = g_1.repeat(batch_head[i+1]-batch_head[i], 1).T + \
            g_1[batch_head[i]:batch_head[i+1]].repeat(n_atoms,1) - \
            2* torch.einsum('bi,ai->ba', (X,X[batch_head[i]:batch_head[i+1],:]))

        # NOTE This is nm squared. YOu should not convert it to angstrom as pdb are written in nm
        #      sometimes -0.0000000XXX appear and sqrt turn nan
        #R = torch.nan_to_num(torch.sqrt(R), nan = 0.0)
        Gamma = (R <= rc_Gamma**2)
        constant = -1. * Gamma/R
        constant = torch.nan_to_num(constant, nan = 0.0, posinf=0.0, neginf= 0.0).unsqueeze(2).unsqueeze(2)





@torch.no_grad()
def X_D(X, device = torch.device(0)):

    n_atoms = X.shape[0]
    
    # Gram
    G = torch.matmul(X, X.T)

    # Distance
    g_1 = torch.matmul(torch.diag(G, diagonal=0).unsqueeze(0).T, torch.ones(1, n_atoms, device=device))
    R = g_1 + g_1.T - 2*G

    # NOTE This is nm squared. Below I convert it to the euclidean form in nm
    R = torch.sqrt(R)#*10
    del G, g_1
    torch.cuda.empty_cache()
    return R




# =====================
# Distance Related
# =====================

# NOTE This is the BIG case Gamma in 2007 Bahar i.e. Laplacian a.k.a. Kirchoff in GNM
@torch.no_grad()
def D_K(R, rc_Gamma = 1.0, device = torch.device(0), M_GammaMask = None):
    """kirchoff matrix is the connectivity matrix
       diagonal gives 
       offdiag gives adjacency matrix  
       R is the EDM m*m matrix
    """
    # The given matrix should be a EDM
    K = torch.zeros((R.size()[0],R.size()[1]), device = device) + R
    K[R > rc_Gamma] = 0.0
    K[R <= rc_Gamma] = -1.0
    K = K.fill_diagonal_(0.0)
    #K_offdiagsum = torch.sum(K,1) # NOTE the diagonal is positive
    K -= torch.diag(torch.sum(K,1), diagonal=0)
    if M_GammaMask is not None:
        K = K * M_GammaMask


    return K





# NOTE This is the SMALL case gamma in 2007 Bahar i.e. ANM gamma spring constant taken to be 1 when within rc NOT Kirchoff!
@torch.no_grad()
def D_MaskRc(D, rc_Gamma = 1.0,M_GammaMask = None , device = torch.device(0)):

    Gamma = (D <= rc_Gamma).to(device)
    if M_GammaMask is not None:
        Gamma = Gamma * M_GammaMask

    return Gamma




# NOTE OBSOLETE Any square symetric matrix to normaalised eig 
@torch.no_grad()
def S_Neigval_Neigvec(K, device = torch.device(0)):
    """
    This function does a few things. 
    1. rearrange the eig vec in descending order of eigval
    2. normalise the eigvec making the eigvec matrix orthoNormal.
    # NOTE I find out that it is actually already done 
    #      eigvec_size = torch.sum(torch.square(eigvec), dim = 0)
    """
    # NOTE I prefer to return the eigvec in descending order! The default is ascending order
    eigval, eigvec = torch.linalg.eigh(K, UPLO='L',out=None)

    # NOTE Anything wrong? shouldn't the first dimension be the index of eigevec?
    #      This correct. The Second dimension is the index of eigvec. Check with. Note the tolerance has to be raised  as below for float 16 or float 32
    #      v = eigvec
    #      w = eigval
    #      a = Local_Laplacian
    #      print(torch.allclose(torch.matmul(v, torch.matmul(w.diag_embed(), v.transpose(-2, -1))), a,  rtol=1e-03, atol=1e-02))
    idx   = torch.flip(torch.argsort(eigval), [0])
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    return eigval, eigvec




# ==================================
# Unorder the cuthill order
# ==================================
def Heigvec_HeigvecUnordered(Heigvec, cuthill_undoorder = [], device = torch.device(0)):
    # This assumes taking in a  (n_eigpair, n_atoms, 3) tensor and a cuthill_unorder np array
    
    return Heigvec[:,cuthill_undoorder,:]



def X_XUnordered(X, cuthill_undoorder = [], device = torch.device(0)):
    # This assumes taking in a  (n_atoms, 3) tensor and a cuthill_unorder np array

    return X[cuthill_undoorder,:]


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