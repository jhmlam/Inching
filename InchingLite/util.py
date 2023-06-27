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



import subprocess
import os
import sys
import gc
import io
import tqdm
import datetime
import time
import zipfile


import collections
from collections.abc import Iterable





import pandas as pd
import numpy as np
import scipy
import openmm.app as mmapp
import mdtraj

#import seaborn as sns
import matplotlib.pyplot as plt


from torch.nn import functional as F
import torch

# ==========================
# Basic IO
# ==========================
# Mkdir if not exist # NOTE now os.makedirs do the same effect??
def MkdirList(folderlist):
    for i in folderlist:
        if not os.path.exists('%s' %(i)):
            os.mkdir('%s' %(i))


def BasicPdbLoading(structure_dir):

    # NOTE Since pdb is written with NM by default we will disallow angstrom!
    OutputInAngstrom = False


    import mdtraj
    traj = mdtraj.load(structure_dir)
    top = traj.topology
    topology_df = top.to_dataframe() [0]


    if OutputInAngstrom:
        topology_df.loc[:, ['x', 'y', 'z']] = traj.xyz[0] * 10.0
    else:
        topology_df.loc[:, ['x', 'y', 'z']] = traj.xyz[0]
    
    return topology_df

def BasicPdbCifLoading(structure_dir):
    
    file_format = structure_dir.split(".")[-1]
    if file_format =='pdb':
        with open(structure_dir, 'r') as tempfile:
            pdb = mmapp.pdbfile.PDBFile(tempfile)
            fileunit = 1                            # NOTE nm is the native uit in pdb
    if file_format =='cif':
        with open(structure_dir, 'r') as tempfile:
            pdb = mmapp.pdbxfile.PDBxFile(tempfile)
            fileunit = 1.0                           # NOTE angstrom is the native uit in pdbx

    pdb_position = pdb.getPositions(asNumpy=True, frame=0) * fileunit

    # NOTE This is a redundant step we need cleaning but not at openmm
    top = mdtraj.Topology.from_openmm(pdb.topology) # TODO Copy the whole class from mdtraj to a util to reduce the load and reliance
    topology_df = top.to_dataframe() [0]
    topology_df.loc[:, ['x', 'y', 'z']] = pdb_position
    return topology_df, top


# ======================
# Helpers from openmm
# =========================
def _formatIndex(index, places):
    """Create a string representation of an atom or residue index.  If the value is larger than can fit
    in the available space, switch to hex.
    """
    if index < 10**places:
        format = f'%{places}d'
        return format % index
    format = f'%{places}X'
    shiftedIndex = (index - 10**places + 10*16**(places-1)) % (16**places)
    return format % shiftedIndex
def _format_83(f):
    """Format a single float into a string of width 8, with ideally 3 decimal
    places of precision. If the number is a little too large, we can
    gracefully degrade the precision by lopping off some of the decimal
    places. If it's much too large, we throw a ValueError"""
    if -999.999 < f < 9999.999:
        return '%8.3f' % f
    if -9999999 < f < 99999999:
        return ('%8.3f' % f)[:8]
    raise ValueError('coordinate "%s" could not be represented '
                     'in a width-8 field' % f)

def OOC_Openmmapp_Pdbfile_writeModel_Bfactor(
    topology, positions, 
    file=sys.stdout, modelIndex=None, keepIds=False, extraParticleIdentifier='EP', 
    User_Bfactor = None):


    import math
    from openmm.unit import nanometers, angstroms, is_quantity, norm, Quantity


    _standardResidues = ['ALA', 'ASN', 'CYS', 'GLU', 'HIS', 'LEU', 'MET', 'PRO', 'THR', 'TYR',
                         'ARG', 'ASP', 'GLN', 'GLY', 'ILE', 'LYS', 'PHE', 'SER', 'TRP', 'VAL',
                         'A', 'G', 'C', 'U', 'I', 'DA', 'DG', 'DC', 'DT', 'DI', 'HOH']

    if len(list(topology.atoms())) != len(positions):
        raise ValueError('The number of positions must match the number of atoms')
    if is_quantity(positions):
        positions = positions.value_in_unit(angstroms)
    if any(math.isnan(norm(pos)) for pos in positions):
        raise ValueError('Particle position is NaN.  For more information, see https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions#nan')
    if any(math.isinf(norm(pos)) for pos in positions):
        raise ValueError('Particle position is infinite.  For more information, see https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions#nan')
    nonHeterogens = _standardResidues[:]
    nonHeterogens.remove('HOH')


    atomIndex = 1
    posIndex = 0
    if modelIndex is not None:
        print("MODEL     %4d" % modelIndex, file=file)

    
    for (chainIndex, chain) in enumerate(topology.chains()):
        if keepIds and len(chain.id) == 1:
            chainName = chain.id
        else:
            chainName = chr(ord('A')+chainIndex%26)
        residues = list(chain.residues())
        for (resIndex, res) in enumerate(residues):
            if len(res.name) > 3:
                resName = res.name[:3]
            else:
                resName = res.name
            if keepIds and len(res.id) < 5:
                resId = res.id
            else:
                resId = _formatIndex(resIndex+1, 4)
            if len(res.insertionCode) == 1:
                resIC = res.insertionCode
            else:
                resIC = " "
            if res.name in nonHeterogens:
                recordName = "ATOM  "
            else:
                recordName = "HETATM"
            for atom in res.atoms():
                if atom.element is not None:
                    symbol = atom.element.symbol
                else:
                    symbol = extraParticleIdentifier
                if len(atom.name) < 4 and atom.name[:1].isalpha() and len(symbol) < 2:
                    atomName = ' '+atom.name
                elif len(atom.name) > 4:
                    atomName = atom.name[:4]
                else:
                    atomName = atom.name
                coords = positions[posIndex]

                if User_Bfactor is None:
                    line = "%s%5s %-4s %3s %s%4s%1s   %s%s%s  1.00  0.00          %2s  " % (
                    recordName, _formatIndex(atomIndex, 5), atomName, resName, chainName, resId, resIC, _format_83(coords[0]),
                    _format_83(coords[1]), _format_83(coords[2]), symbol)
                else:
                    line = "%s%5s %-4s %3s %s%4s%1s   %s%s%s  1.00  %.2f          %2s  " % (
                    recordName, _formatIndex(atomIndex, 5), atomName, resName, chainName, resId, resIC, _format_83(coords[0]),
                    _format_83(coords[1]), _format_83(coords[2]), User_Bfactor[posIndex], symbol)
                if len(line) != 80:
                    raise ValueError('Fixed width overflow detected')
                print(line, file=file)
                posIndex += 1
                atomIndex += 1


            if resIndex == len(residues)-1:
                print("TER   %5s      %3s %s%4s" % (_formatIndex(atomIndex, 5), resName, chainName, resId), file=file)
                atomIndex += 1
    if modelIndex is not None:
        print("ENDMDL", file=file)

def OOC_Openmmapp_Pdbxfile_writeModel_Bfactor(
    topology, positions, 
    file=sys.stdout, modelIndex=1, keepIds=False,
    User_Bfactor = None):
    

        import math
        from openmm.unit import nanometers, angstroms, is_quantity, norm, Quantity


        if len(list(topology.atoms())) != len(positions):
            raise ValueError('The number of positions must match the number of atoms')
        if is_quantity(positions):
            positions = positions.value_in_unit(angstroms)
        if any(math.isnan(norm(pos)) for pos in positions):
            raise ValueError('Particle position is NaN.  For more information, see https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions#nan')
        if any(math.isinf(norm(pos)) for pos in positions):
            raise ValueError('Particle position is infinite.  For more information, see https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions#nan')


        _standardResidues = ['ALA', 'ASN', 'CYS', 'GLU', 'HIS', 'LEU', 'MET', 'PRO', 'THR', 'TYR',
                         'ARG', 'ASP', 'GLN', 'GLY', 'ILE', 'LYS', 'PHE', 'SER', 'TRP', 'VAL',
                         'A', 'G', 'C', 'U', 'I', 'DA', 'DG', 'DC', 'DT', 'DI', 'HOH']

        nonHeterogens = _standardResidues[:]
        nonHeterogens.remove('HOH')

        atomIndex = 1
        posIndex = 0
        for (chainIndex, chain) in enumerate(topology.chains()):
            if keepIds:
                chainName = chain.id
            else:
                chainName = chr(ord('A')+chainIndex%26)
            residues = list(chain.residues())
            for (resIndex, res) in enumerate(residues):
                if keepIds:
                    resId = res.id
                    resIC = (res.insertionCode if res.insertionCode.strip() else '.')
                else:
                    resId = resIndex + 1
                    resIC = '.'
                if res.name in nonHeterogens:
                    recordName = "ATOM"
                else:
                    recordName = "HETATM"
                for atom in res.atoms():
                    coords = positions[posIndex]
                    if atom.element is not None:
                        symbol = atom.element.symbol
                    else:
                        symbol = '?'
                    if User_Bfactor is None:
                        line = "%s  %5d %-3s %-4s . %-4s %s ? %5s %s %10.4f %10.4f %10.4f  0.0  0.0  ?  ?  ?  ?  ?  .  %5s %4s %s %4s %5d"
                        print(line % (recordName, atomIndex, symbol, atom.name, res.name, chainName, resId, resIC, coords[0], coords[1], coords[2],
                                    resId, res.name, chainName, atom.name, modelIndex), file=file)
                    else:
                        line = "%s  %5d %-3s %-4s . %-4s %s ? %5s %s %10.4f %10.4f %10.4f  0.0  %.2f  ?  ?  ?  ?  ?  .  %5s %4s %s %4s %5d"
                        print(line % (recordName, atomIndex, symbol, atom.name, res.name, chainName, resId, resIC, coords[0], coords[1], coords[2],
                                    User_Bfactor[posIndex],
                                    resId, res.name, chainName, atom.name, modelIndex), file=file)
                    posIndex += 1
                    atomIndex += 1


                    

def BasicPdbCifWriting( ref_structure_dir = '',         # Expect a pdb file directory
                        save_structure_dir = "",        # Expect a pdb file directory
                        position = np.array([[],[]]),   # Accepting a 3D tensor (t,n,3)
                        keepIds=True,
                        SaveFormat = 'cif', SaveSeparate = False, User_Bfactor = None):

    assert len(position.shape) == 3, "Accepting a 3D tensor as position (t,n,3)"


    file_format = ref_structure_dir.split(".")[-1]
    if file_format =='pdb':
        with open(ref_structure_dir, 'r') as tempfile:
            pdb = mmapp.pdbfile.PDBFile(tempfile)
            fileunit = 1     
    if file_format =='cif':
        with open(ref_structure_dir, 'r') as tempfile:
            pdb = mmapp.pdbxfile.PDBxFile(tempfile)
            fileunit = 0.1   

    from openmm import Vec3
    from openmm.unit import nanometers

    # Overwrite exisiting
    pdb._positions = []
    for t in range(position.shape[0]):
        temppositions = []
        #pdb._positions.append([])
        for i in range(position.shape[1]):
            temppositions.append(Vec3(float(position[t,i,0]), float(position[t,i,1]), float(position[t,i,2]))*fileunit)
        pdb._positions.append(temppositions)




    for i in range(len(pdb._positions)):
            pdb._positions[i] = pdb._positions[i]*nanometers

    # ===================
    # Save
    # ===================
    if User_Bfactor is None:
        if SaveFormat == 'cif':
            with open(save_structure_dir, 'w') as tempfile:
                mmapp.pdbxfile.PDBxFile.writeHeader(pdb.topology, file=tempfile,)
                for i in  range(len(pdb._positions)):
                    mmapp.pdbxfile.PDBxFile.writeModel(
                                    pdb.topology, pdb._positions[i]*10.0, 
                                    file=tempfile, keepIds=keepIds, modelIndex = i)
        else:
            with open(save_structure_dir, 'w') as tempfile:
                mmapp.pdbfile.PDBFile.writeHeader(pdb.topology, file=tempfile,)
                for i in  range(len(pdb._positions)):
                    mmapp.pdbfile.PDBFile.writeModel(
                        pdb.topology, pdb._positions[i], 
                        file=tempfile, keepIds=keepIds, modelIndex = i)
                mmapp.pdbfile.PDBFile.writeFooter(pdb.topology, file=tempfile)

    else:
        if SaveFormat == 'cif':
            with open(save_structure_dir, 'w') as tempfile:
                mmapp.pdbxfile.PDBxFile.writeHeader(pdb.topology, file=tempfile)
                for i in  range(len(pdb._positions)):
                    OOC_Openmmapp_Pdbxfile_writeModel_Bfactor(
                                    pdb.topology, pdb._positions[i]*10.0, 
                                    file=tempfile, keepIds=keepIds, modelIndex = i, User_Bfactor= User_Bfactor)
        else:
            with open(save_structure_dir, 'w') as tempfile:
                mmapp.pdbfile.PDBFile.writeHeader(pdb.topology, file=tempfile,)
                for i in  range(len(pdb._positions)):
                    OOC_Openmmapp_Pdbfile_writeModel_Bfactor(
                        pdb.topology, pdb._positions[i], 
                        file=tempfile, keepIds=keepIds, modelIndex = i, 
                        User_Bfactor= User_Bfactor)
                mmapp.pdbfile.PDBFile.writeFooter(pdb.topology, file=tempfile)



def SaveOneModeLinearisedAnime(deltaX, X, 
                    n_timestep = 10, 
                    DIR_ReferenceStructure = "",
                    DIR_SaveFolder = "", 
                    SaveFormat = 'cif',
                    outputlabel = '',
                    max_abs_deviation = 1.0,
                    stepsize = 0.5,
                    max_n_output = 10,
                    SaveSeparate = False,
                    UnitMovement = False,
                    RemoveOrig = False, # NOTE This flag remove the unmoved structure from the trajectory produce
                    User_Bfactor = None,
                    ):
    import numpy as np
    import pandas as pd


    MkdirList([DIR_SaveFolder])



    if torch.is_tensor(deltaX):
        deltaX = deltaX.detach().cpu().numpy()
    else:
        pass

    if torch.is_tensor(X):
        xyz = X.detach().cpu().numpy()
    else:
        xyz = X
        pass



    # TODO Save +- eigevector v/||v|| * 0.5 angstrom
    if UnitMovement:
        deltaX = deltaX / np.sqrt(
                    np.sum( deltaX*  deltaX, axis =1)
                    )[:,None]
        deltaX *= stepsize 
    else:
        deltaX = deltaX / np.max(np.sqrt(
                    np.sum( deltaX*  deltaX, axis =1)
                    )[:,None]) # NOTE This make sure everyone is bounded by 1
        deltaX *= stepsize 

    


    df = []
    # Positive Direction
    for t in range(n_timestep):
        uvw =  xyz+(deltaX)/n_timestep*t
        uvw = uvw.tolist()

        if RemoveOrig:
            if t == 0:
                continue


        if np.abs(t)*stepsize >  max_abs_deviation:
            continue

        atom_index = 0
        for j in range(len(uvw)):
            df.append([t, atom_index]+uvw[j]  )
            atom_index +=1        

    # Negative direction
    for t in range(n_timestep):

        if t == 0:
            continue

        if np.abs(t)*stepsize >  max_abs_deviation:
            continue

        uvw =  xyz-(deltaX)/n_timestep*t
        uvw = uvw.tolist()

        atom_index = 0
        for j in range(len(uvw)):
            df.append([-1*t, atom_index]+uvw[j]  )
            atom_index +=1 


    df = pd.DataFrame(df, columns = ['Time', 'atom', 'x','y','z'])
    df = df.sort_values('Time', axis=0, ascending=True, inplace=False, 
                        kind='quicksort', na_position='last', ignore_index=False, key=None)

    pdbid = DIR_ReferenceStructure.replace("\\", "/").split("/")[-1].split(".")[0]
    structure_count = 0
    traj = []
    for x, y in tqdm.tqdm(df.groupby('Time', as_index=False)):

        if structure_count > max_n_output:
            continue

        pos = y.sort_values('atom', axis=0, ascending=True, inplace=False, 
                        kind='quicksort', na_position='last', ignore_index=False, key=None)
        pos = pos[['x','y','z']].to_numpy().astype(np.float64)
        traj.append(pos)
        structure_count += 1


    # =============
    # save
    # ===============
    traj = np.array(traj, dtype=np.float64) # NOTE float64 is must

    if SaveSeparate:
        for t in range(traj.shape[0]):
            try:
                BasicPdbCifWriting(ref_structure_dir = DIR_ReferenceStructure, 
                        save_structure_dir = DIR_SaveFolder + '/%s_%s%s.%s' %(pdbid, outputlabel, str(t).zfill(len(str(n_timestep))), SaveFormat), 
                        position =traj[t:t+1,:,:], keepIds=True,
                        SaveFormat = SaveFormat, User_Bfactor = User_Bfactor)      
            except:
                print('/%s_%s%s.%s produce a Nan rejected' %(pdbid, outputlabel, str(t).zfill(len(str(n_timestep))), SaveFormat))
    else:
        BasicPdbCifWriting(ref_structure_dir = DIR_ReferenceStructure, 
                    save_structure_dir = DIR_SaveFolder + '/%s_%s.%s' %(pdbid, outputlabel, SaveFormat), 
                    position =traj, keepIds=True,
                    SaveFormat = SaveFormat, User_Bfactor = User_Bfactor)          
    

    return




# ================
# Platform tricks
# ==================


def WinFileDirLinux(s):
    return s.replace("\\", "/")





def GetDateTimeNowString(indexformat = False):
    now = datetime.datetime.now()
    if indexformat:
        d = now.strftime("%Y-%m-%d %H:%M:%S")
    else:
        d = now.strftime("%Y%m%d%H%M%S")
    return d






# =========================
# Torch setting util
# =========================

def TorchMakePrecision(Precision = "torch.float16"):
    PrecisionDict = {
        "torch.bfloat16": (torch.bfloat16, torch.cuda.BFloat16Tensor),
        "torch.float16":(torch.float16, torch.cuda.HalfTensor),
        "torch.float32":(torch.float32, torch.cuda.FloatTensor),
        "torch.float64":(torch.float64, torch.cuda.DoubleTensor),
    }

    torch.set_default_dtype(PrecisionDict[str(Precision)][0])
    torch.set_default_tensor_type(PrecisionDict[str(Precision)][1])





def TorchEmptyCache():
    torch.cuda.empty_cache()    
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.memory_allocated(0)
    torch.cuda.max_memory_allocated(0)








# ===========================
# Recursions
# ===========================

def GetPartitionTree(iteratorA, maxleafsize = 108):
    n = len(iteratorA)
    if n <= maxleafsize:
        return n
    k = np.floor(n/2).astype(int)
    return GetPartitionTree(range(0,k+1), maxleafsize = maxleafsize),  GetPartitionTree(range(k+1,n), maxleafsize = maxleafsize)





def FlattenPartitionTree(nested):
    from collections.abc import Iterable
    def flatten(collection):
        for x in collection:
            if isinstance(x, Iterable):
                yield from flatten(x)
            else:
                yield x

    def extract(nested):
        yield from (x for x in flatten(nested))

    generator = extract(nested)
    return generator




def PrimeFactorList(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d) 
            n //= d
        d += 1
    if n > 1:
       primfac.append(int(n))
    return primfac

# ==================================
# Visual tools
# ===================================
def AnimateOneMode(deltaX, X, 
            n_timestep = 10, 
            StripDirection = 2, # NOTE :2 means using x,y but not z
            ):
    import numpy as np

    # NOTE https://plotly.com/python/visualizing-mri-volume-slices/
    import plotly.io as pio
    pio.renderers.default = "notebook_connected"
    import plotly.express as px
    import pandas as pd

    deltaX = deltaX / torch.sqrt(torch.sum( deltaX*  deltaX, axis =1)).unsqueeze(1) *0.5
    xyz = (X-torch.mean(X, axis=0)).cpu().numpy()
    colorstrip = (xyz[:, :StripDirection].mean(axis=1) % 1.0).tolist()    

    UsePcaColorstrip = True
    if UsePcaColorstrip:
        
        X_ = X.cpu().numpy()
        X_ = (X_-np.mean(X_,axis=0) )/np.std(X_,axis = 0) 
        X_cov = np.cov(X_.T)
        _, pcs = np.linalg.eig(X_cov)
        projection_matrix = (pcs.T[:,1]) # Using the second mode
        X_pca = X_.dot(projection_matrix)#[:,0]
        colorstrip = (X_pca.flatten() % 1.0).tolist()
        
    # TODO View at the pc2

    df = []
    for t in range(n_timestep):
        uvw =  xyz+(deltaX).cpu().numpy()/n_timestep*t
        uvw = uvw.tolist()

        atom_index = 0
        for j in range(len(uvw)):
            df.append([t, atom_index]+uvw[j]+[colorstrip[j]]  )
            atom_index +=1        
    
    df = pd.DataFrame(df, columns = ['Time', 'atom', 'x','y','z', 'Colorstrip'])
    fig = px.scatter_3d(df, x='x', y='y', z='z',
                color = 'Colorstrip', size_max = 0.2,
                opacity=0.08, animation_frame = 'Time', template='plotly',
                color_continuous_scale=px.colors.sequential.Viridis,
                    range_color=[min(colorstrip),max(colorstrip)])


    fig.update_layout(
                    title='Vibrational Mode',
                    width=600,
                    height=600,
                    scene=dict(
                                xaxis=dict(range=[df.x.min()-0.5, df.x.max()+0.5], autorange=False),
                                yaxis=dict(range=[df.y.min()-0.5, df.y.max()+0.5], autorange=False),
                                zaxis=dict(range=[df.z.min()-0.5, df.z.max()+0.5], autorange=False),
                                aspectratio=dict(x=1, y=1, z=1),
                                camera = dict(
                                    projection = dict(type = "orthographic")
                                    )
                                )
            )
    fig.show()

def ShowOneMode(deltaX, X, 
                User_Stride = 1,
                User_Size = 0.25):

    """Accepting two torch (n,3) deltaX refers to H_eigvec[0] for example"""


    if torch.is_tensor(deltaX):
        deltaX = deltaX.detach().cpu().numpy()
    else:
        pass

    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    else:
        pass
    
    deltaX = deltaX / np.sqrt(
                    np.sum( deltaX*  deltaX, axis =1)
                    )[:,None]
    deltaX *= User_Size # NOTE Make it less busy visually

    X = X[::User_Stride,:]
    deltaX = deltaX[::User_Stride,:]

    xyz = (X-np.mean(X, axis=0))
    uvw = (X-np.mean(X, axis=0)+deltaX)
    
    #deltaX = deltaX / torch.sqrt(torch.sum( deltaX*  deltaX, axis =1)).unsqueeze(1) *0.5
    #xyz = (X-torch.mean(X, axis=0)).cpu().numpy()
    #uvw = (X-torch.mean(X, axis=0)+deltaX).cpu().numpy()


    import plotly.express as px
    import pandas as pd
    import plotly as py
    import plotly.graph_objs as go

    df1 = pd.DataFrame(xyz, columns=['x', 'y', 'z'])
    df1.loc[:,"Label"] =1.0
    df2 = pd.DataFrame(uvw, columns=['x', 'y', 'z'])
    df2.loc[:,"Label"] =0.0
    df = pd.concat([df1,df2], ignore_index=True)

    x_lines = []
    y_lines = []
    z_lines = []
    for i in range(df1.shape[0]):
        x_lines.extend([xyz[i,0 ], uvw[i,0 ]])
        y_lines.extend([xyz[i,1 ], uvw[i,1 ]])
        z_lines.extend([xyz[i,2 ], uvw[i,2 ]])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)    

    # Thin Red line
    trace2 = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        name='Movement'
    )
    # Initial
    trace1 =go.Scatter3d( x=xyz[:,0].flatten(), 
                        y=xyz[:,1].flatten(), 
                        z=xyz[:,2].flatten(), 
                        mode = 'markers', opacity=0.5, marker=dict(size=4),
                        name = 'Initial')
    # Final
    trace3 =go.Scatter3d( x=uvw[:,0].flatten(), 
                        y=uvw[:,1].flatten(), 
                        z=uvw[:,2].flatten(), 
                        mode = 'markers', opacity=0.5, marker=dict(size=4),
                        name = 'Final')

    fig = go.Figure(data=[trace1,trace2,trace3])
    fig.update_layout(
        margin=dict(l=1, r=1, b=1, t=1),  
        dragmode= 'zoom',
        autosize=True, scene=dict(
                        camera=dict(
                            #up=dict(x=0, y=0, z=1), 
                        center=dict(x=0, y=0, z=0),
                        eye=dict({'x': 0, 'y': 1, 'z': 0}),
                        projection=dict(type='perspective'))))
    fig.show(config = dict({'scrollZoom': True, 'responsive': False, 'displayModeBar': True}))


def ShowOneModeMagnitude(deltaX, X, 
                        BoxCox = False,
                        User_WinsorizingWindow = (0.01, 0.99),
                        User_LogisticParam = (1.0, 1.0)
                        ):

    """Accepting two torch (n,3) deltaX refers to H_eigvec[0] for example"""





    if torch.is_tensor(deltaX):
        deltaX = deltaX.detach().cpu().numpy()
    else:
        pass

    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    else:
        pass
    
    deltaX_magnitude =  np.sqrt(
                    np.sum( deltaX*  deltaX, axis =1)
                    )[:,None]
    deltaX_magnitude = deltaX_magnitude.flatten()
    #deltaX_magnitude = torch.sqrt(torch.sum( deltaX*  deltaX, axis =1)).unsqueeze(1) *0.5
    #deltaX_magnitude = deltaX_magnitude.detach().cpu().numpy().flatten()



    xyz = (X-np.mean(X, axis=0))
    uvw = (X-np.mean(X, axis=0)+deltaX*0.2)
    deltaX_magnitude = deltaX_magnitude.tolist()

    import plotly.express as px
    import pandas as pd
    import plotly as py
    import plotly.graph_objs as go

    df1 = pd.DataFrame(xyz, columns=['x', 'y', 'z'])
    df1.loc[:,"Label"] =1.0
    df1.loc[:,"Magnitude"] = deltaX_magnitude
    #df2 = pd.DataFrame(uvw, columns=['x', 'y', 'z'])
    #df2.loc[:,"Label"] =0.0
    #df = pd.concat([df1,df2], ignore_index=True)
    df = df1
    
    x_lines = []
    y_lines = []
    z_lines = []
    for i in range(df1.shape[0]):
        x_lines.extend([xyz[i,0 ], uvw[i,0 ]])
        y_lines.extend([xyz[i,1 ], uvw[i,1 ]])
        z_lines.extend([xyz[i,2 ], uvw[i,2 ]])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)    

    # Thin Red line
    trace2 = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        name='Movement'
    )

    print(df)
    import plotly.express as px
    #df = px.data.iris()
    fig = px.scatter_3d(df, x = 'x', y = 'y', z = 'z',
              color='Magnitude', size='Magnitude', 
              size_max=18, #size_min=1,

            opacity=0.8)

    # tight layout
    fig.add_traces([trace2])
    fig.update_traces(marker=dict(
                              line=dict(width=0,
                                        )))

    camera = dict(
        eye=dict(x=0.1, y=0.1, z=0.1)
        )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene_camera=camera, 
                        
    )
    fig.show()



def ShowValuePerNode(deltaX_magnitude, X, User_MagnitudeOnly = False, User_Stride = 1):

    """Accepting two torch (n,3) deltaX refers to H_eigvec[0] for example"""

    if torch.is_tensor(deltaX_magnitude):
        deltaX_magnitude = deltaX_magnitude.detach().cpu().numpy().flatten()
    else:
        pass

    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    else:
        pass

    # NOTE Apply stride
    deltaX_magnitude = deltaX_magnitude[::User_Stride]
    X = X[::User_Stride,:]

    
    xyz = (X-np.mean(X, axis=0))



    import plotly.express as px
    import pandas as pd
    import plotly as py
    import plotly.graph_objs as go

    df1 = pd.DataFrame(xyz, columns=['x', 'y', 'z'])
    df1.loc[:,"Label"] =1.0
    if User_MagnitudeOnly:

        df1.loc[:,"Magnitude"] = (1.0/(-1.0 * np.log10(deltaX_magnitude))).tolist()
    else:
        df1.loc[:,"Magnitude"] = deltaX_magnitude.tolist()
    #df2 = pd.DataFrame(uvw, columns=['x', 'y', 'z'])
    #df2.loc[:,"Label"] =0.0
    #df = pd.concat([df1,df2], ignore_index=True)
    df = df1
    
    print(df)
    import plotly.express as px
    #df = px.data.iris()

    df.loc[:,'AbsMagnitude'] = df['Magnitude'].abs()
    fig = px.scatter_3d(df, x = 'x', y = 'y', z = 'z',
            color='Magnitude', size='AbsMagnitude', 
            size_max=18, #size_min=4,

            opacity=0.8)

    # tight layout
    #fig.add_traces([trace2])
    fig.update_traces(marker=dict(
                              line=dict(width=0,
                                        )))

    camera = dict(
        eye=dict(x=0.1, y=0.1, z=0.1)
        )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene_camera=camera, 
                        
    )
    fig.show()






def ShowImageGrid(images, num_images = 8*8, 
                    SymLogNorm_precision = 0.1, 
                    nrow = 8, channels = 3):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import SymLogNorm

    npimg = images.cpu()
    npimg = npimg[:num_images,:channels,:,:].float() # NOTE While we may have a thousand channel we can only visualise 3 channels in RGB
    npimg = npimg.numpy()  



    num_gridrow = int(num_images/nrow)
    img_concat = []
    for i in range(num_gridrow):
        img_concat.append(np.concatenate(npimg[i*nrow:(i+1)*nrow, :,:,:],axis = 2))
    npimg = np.concatenate(img_concat,axis = 1)
    



    npimg = np.transpose(npimg, (1, 2, 0))

    if images.shape[1]  >= 3:
       pass
    else:
       npimg = npimg[:,:,0]

    if npimg.shape[0] > 100: 
       plt.figure(figsize = (15, 15 ))
    
    if SymLogNorm_precision > 0.0:
        im = plt.imshow(npimg,  cmap='jet', aspect = 'equal', norm=SymLogNorm(SymLogNorm_precision))
    else:
        im = plt.imshow(npimg,  cmap='jet', aspect = 'equal')

    ax = plt.gca()
    for i in range(int(num_images/nrow) + 1):
        ax.axhline(i*images.shape[3], linewidth=2, c = 'k')
    for i in range(nrow):
        ax.axvline(i*images.shape[2], linewidth=2, c = 'k')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.show()

    #del img
    del npimg
    TorchEmptyCache()
    gc.collect()





def ShowActiveTensorboard():
    from tensorboard import notebook
    import tempfile
    import os
    print("=============================================")
    path = os.path.join(tempfile.gettempdir(), ".tensorboard-info") 
    print(path)
    notebook.list()
    print("=============================================")





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

