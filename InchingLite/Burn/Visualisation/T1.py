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

