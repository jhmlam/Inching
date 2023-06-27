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
sys.path.append('../Script/Burn/')

import torch.nn as nn
import torch.nn.functional as F
import time

import InchingLite.Burn.Visualisation.T1

@torch.no_grad()
def Heigval_Heigvec_SqFluctPerMode_SqFluct_RatioSqFPM_RatioVar(H_eigval, H_eigvec):

    RatioVariance = 1 / torch.sqrt(torch.abs(H_eigval))
    RatioVariance = RatioVariance / RatioVariance.max()

    # This is the magnitude  of each eigenvector on each atom
    SqFluctPerMode = torch.sqrt(torch.sum(H_eigvec * H_eigvec, axis = 2))
    RatioSqFPM = RatioVariance.unsqueeze(1) *SqFluctPerMode
    # This gives the square fluctuation per atom averaged over eigenvectors calculated.
    SqFluct = torch.sum(RatioSqFPM, axis = 0)
    return SqFluctPerMode.unsqueeze(2), SqFluct, RatioSqFPM.unsqueeze(2), RatioVariance


# NOTE For fun only! don't do this with >100 k atoms to fry your computer 
@torch.no_grad()
def Heigval_Heigvec_HccPairlist_HccBatch(H_eigval, H_eigvec, 
                                         plot_CC = True, device = torch.device(0), 
                                         c_mode = 5, SelfOnly = True):
    # NOTE The Hcc Batch is sorted by eigval i * eigval j, which is by default as pair indices are sorted! 
    # (0,0) (0,1) (0,2) .... (k-1,k)
    # NOTE The eigenvec is stabilised by sign of the first element. TODO We should have two CC i.e. CC+ and CC-?
    #      Basically the resultant CC will have a switch of sign behaving like a cosine deterministically

    n_atoms = H_eigvec.shape[1]

    # ========================================
    # Attribute of hessian eigenpair
    # ========================================
    # 2. Explained variane
    # NOTE Because we did not calculate all the eigenvalues
    # Calcualte fractional explained variance
    H_FractionalExplainedVariance = H_eigval**2 /  torch.sum(H_eigval**2)
    #print(H_FractionalExplainedVariance)



    # 2. Calculate square fluctuation per atom 
    # NOTE However it is shown on papers that GNM i.e. using magnitude of the laplacian's eigen vec is more accurate in sqfluct
    # calcSqFlucts. summing up the top k eigenvec
    H_sqfluct = torch.sum(H_eigvec**2, dim=2)
    #print(H_sqfluct.shape)



    
    # 3. Calculate combinatorial CC
    #https://github.com/prody/ProDy/blob/9e0e07ffb1c6a060cf2abce855f18d2b41b7d693/prody/dynamics/analysis.py


    # NOTE That hessian is not efficient for storage (3N^2)=9 (N^2) ... and it can always be calculated from X again, 
    #      so the return of this function should be log_10 (covariance/cosine distance) which each pair takes only N^2
    #      A table of c_mode choose 2 for reference
    #      c_mode = 5    10 combination
    #      c_mode = 10   45 combination

    # NOTE This is the CC for overall c_mode taken mean. Things got averaged. not a good featurization option.
    #CC = torch.tensordot(H_eigvec.permute(2, 0, 1), H_eigvec.permute(0, 2, 1) , dims=([0, 1], [1, 0]))

    # NOTE Below is CC b/w mode 1 and mode 0
    st = time.time()


    if SelfOnly:
        pair_CC = sorted([(i, i) for i in range(c_mode)])
    else:
        pair_CC = sorted(torch.triu_indices(c_mode,c_mode).T.tolist())

    hessian_CC = torch.zeros((len(pair_CC),n_atoms, n_atoms), device = device)
    j = 0
    for i in pair_CC:
        hessian_CC[j,:,:] += torch.tensordot(H_eigvec[i[0],:,:].permute(0, 1), H_eigvec[i[1],:,:].permute(1, 0) , dims=1) * H_eigval[i[0]] * H_eigval[i[1]]
        j+=1


    # NOTE In general the eigenval varies in scale 10^1 to 10^-3 so the CC has to be rescaled.
    #      Also note that we are de facto doing cosine distance among eigenvectors. Neat and clean
    #      The cross-correlation saying is valid as (X-X_mean)(Y - Y_mean) / denominator takes *_mean to be centered i.e. 0,0,0
    #      A log modulus transform https://blogs.sas.com/content/iml/2014/07/14/log-transformation-of-pos-neg.html
    #      https://www.jstor.org/stable/2986305

    hessian_CC = InchingLite.Burn.Visualisation.T1.S_LogModulusS(hessian_CC, precision = 2)
    #torch.sign(hessian_CC) * torch.log10(torch.abs(hessian_CC)+ 0.0000001)
    print("Time consumed in CC(Hessian)", time.time()- st)



    # TODO Extract this as a T1
    if plot_CC:
        #import itertools
        import warnings
        #import torchvision
        #import sys
        #sys.path.append('..')
        import InchingLite.util as util
        warnings.filterwarnings('ignore')

        # TODO Use imageGrid in util

        # ===============================
        # CC considering unit size
        # ===============================
        H_eigvec_unit = H_eigvec / torch.sqrt(torch.sum(H_eigvec * H_eigvec, axis = 2)).unsqueeze(2)
        H_eigvec_num = H_eigvec.shape[0]
        CC_batch = []
        for pairs in pair_CC:
            pairs = sorted(pairs)
            CC_batch.append(torch.tensordot(H_eigvec_unit[pairs[0],:,:].permute(0, 1), H_eigvec_unit[pairs[1],:,:].permute(1, 0) , dims=1).unsqueeze(0).unsqueeze(0))
        CC_batch = torch.cat(CC_batch, dim=0)

        # If unit sized then no need for log mod transform.
        #CC_batch = InchingLite.Burn.T1.S_LogModulusS(CC_batch, precision = 2)
        if SelfOnly:
            if CC_batch.shape[0]%2 == 0:
                image_per_row = int(CC_batch.shape[0]/2)
            else:
                image_per_row = int(CC_batch.shape[0]/2)+1
        else:
            image_per_row = H_eigvec_num
        util.ShowImageGrid(CC_batch, num_images = CC_batch.shape[0], SymLogNorm_precision = 0.00, nrow =image_per_row )

        torch.cuda.empty_cache()

    #del H_eigval, H_eigvec
    torch.cuda.empty_cache()
    # Unfortunately torch nn takes [n_sample, n_channel, H,W]
    return torch.triu_indices(c_mode,c_mode).T, hessian_CC #.permute(1,2,0)



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
