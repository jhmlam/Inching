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






# ==========================
# Arithmatics  Operators
# ==========================
@torch.no_grad()
def S_LogModulusS(S, precision = 1):
    """The first step of this function is to define the precision interested.
       For numbers in S > 1.0 it does not matter. 
       Starting from 0.1, log(0.1) will be a negative number, we do not want this sign interfere w/ 
       the log so we take log modulus i.e. torch.abs(S)+ 1.0. 
       But in case 0.1 is important we may want to raise it before taking the log modulus."""
    S = S*(100.0**precision)
    S = torch.sign(S) * torch.log2(torch.abs(S)+ 1.0)
    return S





@torch.no_grad()
def M_BatchUnitVector(M, axis = 2):
    """M is (b,N, axis)"""
    M_unit = M / torch.sqrt(torch.sum(M * M, axis = axis)).unsqueeze(axis)
    return M_unit

def M_BatchMagnitude(M, axis = 2):
    """M is (b,N, axis)"""
    M_mag = torch.sqrt(torch.sum(M * M, axis = axis)).unsqueeze(axis)
    return M_mag



@torch.no_grad()
def M_BatchMinMaxVector(M, axis = 2):
    """M is (b,N, axis)"""
    M_unit = M / torch.sqrt(torch.sum(M * M, axis = axis)).unsqueeze(axis)
    return M_unit

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