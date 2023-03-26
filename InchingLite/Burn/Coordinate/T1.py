import torch
import sys
import tqdm
import pickle
sys.path.append('..')
sys.path.append('../Script/Burn/')





import InchingLite.util



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
    PartitionTree = InchingLite.util.GetPartitionTree(range(n_atoms), maxleafsize = maxleafsize)
    FlattenPartitionTree_generator = InchingLite.util.FlattenPartitionTree(PartitionTree)
    batch_head = [0]
    for i in sorted(FlattenPartitionTree_generator)[::-1]:
        batch_head.append(batch_head[-1] + i)


    X = X.type(torch.float32)
    InchingLite.util.TorchMakePrecision(Precision = str(X.dtype))
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


@torch.no_grad()
def BatchRotVecRodriguezEuler_BatchRotMat(RotVec, 
                    device=torch.device(0), 
                    dtype_temp = torch.float64,
                    User_NoGrad = True):

    """
    # How to use it?

    RotVec = torch.randn((77777,4), device = device, dtype=torch.double,requires_grad=True)
    RotMat = BatchRotVecRodriguezEuler_BatchRotMat(RotVec, 
                        device=torch.device(0), 
                        dtype_temp = torch.float64,)

    X = torch.randn((77777,3), device = device, dtype=torch.double)
    X = X.unsqueeze(2)

    # NOTE There are below machine episoln e-18 discrepancy. Fine.
    Y = torch.bmm(torch.bmm(RotMat.permute(0,2,1),RotMat),X)

    # NOTE Do we require grad? seems no as it can be part of loss function
    """

    #if User_NoGrad:
    #    torch.no_grad()

    # NOTE Batched Euler-Rodriguez with correct propagation.
    #      https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    #      Assume a random vector of n*4 in range (-1,1), we will first normalise the first 3 element
    #      to get the axis and the last element will be used as the counterclockwise angle rotation after times pi
    #      In particular, I like Euler-Rodriguez because we only need to call sine cosine once.

    axiss = torch.nn.functional.normalize(RotVec[:,:3], p=2, dim=1, eps=1e-16)  # (N,3)
    theta = RotVec[:,3] * torch.acos(torch.zeros(1, requires_grad=False, device=device,dtype=dtype_temp)).item() #(N,)


    a = torch.cos(theta / 2.0)
    bcd = torch.sin(theta / 2.0).unsqueeze(1) * axiss


    aa = a * a
    bb = bcd[:,0] * bcd[:,0]
    cc = bcd[:,1] * bcd[:,1]
    dd = bcd[:,2] * bcd[:,2]

    ab = a * bcd[:,0]
    ac = a * bcd[:,1]
    ad = a * bcd[:,2]

    bc = bcd[:,0] * bcd[:,1]
    bd = bcd[:,0] * bcd[:,2]
    cd = bcd[:,1] * bcd[:,2]


    return torch.stack([    torch.stack([aa + bb - cc - dd  , 2 * (bc + ad)     , 2 * (bd - ac)]),
                            torch.stack([2 * (bc - ad)      , aa + cc - bb - dd , 2 * (cd + ab)]),
                            torch.stack([2 * (bd + ac)      , 2 * (cd - ab)     , aa + dd - bb - cc])
                            ]).permute((2,1,0)) # NOTE permute (2,1,0) as we need row column correct 


# ========================
# Ansatz Related
# =========================
@torch.no_grad()
def X_TransRotRitz(X,
                    device=torch.device(0), 
                    dtype_temp = torch.float64,
                    User_Noise = True,
                    #extremecase_batchsize = 100,
                    #extremecasedefinition = 350,
                    ):

    n_atoms = int(X.shape[0])
    dof = int(n_atoms*3)
    n_modes = 6
    #print(n_atoms,n_modes, dof )
    V = torch.zeros((n_modes,dof), device=device, dtype=dtype_temp)
       #torch.zeros((n,m+1), device=device, dtype=dtype_temp) 
    # NOTE Translationals
    tx_matrix = torch.zeros((n_atoms, 3),device=device, dtype=dtype_temp)
    tx_matrix[:,0] += 1
    #tx_matrix[0,0] += 1e-7
    V[0,:] = tx_matrix.reshape(dof)
    ty_matrix = torch.zeros((n_atoms, 3),device=device, dtype=dtype_temp)
    ty_matrix[:,1] += 1
    #ty_matrix[0,1] += 1e-7
    V[1,:] = ty_matrix.reshape(dof)
    tz_matrix = torch.zeros((n_atoms, 3),device=device, dtype=dtype_temp)
    tz_matrix[:,2] += 1
    #tz_matrix[0,2] += 1e-7
    V[2,:] = tz_matrix.reshape(dof)

    # NOTE Rotationals
    #      Indeed this is the same as X@R.T - X but we don't want fill in due to subtraction
    #      s.t. the Rotation matrix R has one 1 zeroed
    rx_matrix = torch.tensor([  [0,0,0],
                                [0,0,-1],
                                [0,1,0]], device=device, dtype=dtype_temp)
    ry_matrix = torch.tensor([  [0,0,1],
                                [0,0,0],
                                [-1,0,0]], device=device, dtype=dtype_temp)
    rz_matrix = torch.tensor([  [0,-1,0],
                                [1,0,0],
                                [0,0,0]], device=device, dtype=dtype_temp)
    V[3,:] = (X@rx_matrix.T ).reshape(dof)
    V[4,:] = (X@ry_matrix.T ).reshape(dof)
    V[5,:] = (X@rz_matrix.T ).reshape(dof)


    # NOTE Normalise. This is necessary before our reorth!
    for iii in range(6):
        V[iii,:] /= torch.linalg.vector_norm(V[iii,:], ord = 2)


    #"""
    # NOTE If the ritz is exactly the same as the eigenvector presented i.e. of the symmetric hessian
    #      then it will never converge as we expect the eigvec are orthogonal! see problem 7.3 of ANL
    #      The reason is that the first eigvec0 will produce a zero ritz1 and this will be reorthogonalise
    #      against the eigvec0. nonsense 
    if User_Noise:
        V += torch.nan_to_num(
            torch.randn((n_modes,dof), device=device, dtype=dtype_temp)  * max((1/(9*n_atoms*n_atoms)), 5e-11), 
            nan=0.0)

        #print("With noise %s" %(max((1/(9*n_atoms*n_atoms)), 1e-11)))
    #"""

    V = V.T
    for ix in range(6):
        if ix == 0:
            continue
        V[:,ix] -= torch.mv(V[:,:ix], torch.mv( V[:, :ix].T,V[:,ix] ))
        V[:,ix] /= torch.sqrt(V[:, ix].T @ V[:, ix]) # TODO torch.matmul or mvs
        V[:,ix] -= torch.mv(V[:,:ix], torch.mv( V[:, :ix].T,V[:,ix] ))
        V[:,ix] /= torch.sqrt(V[:, ix].T @ V[:, ix])
    V = V.T
    

    # NOTE Normalise
    for iii in range(6):
        V[iii,:] /= torch.linalg.vector_norm(V[iii,:], ord = 2)



    del tx_matrix ,ty_matrix ,tz_matrix, rx_matrix, ry_matrix, rz_matrix
    try:
        InchingLite.util.TorchEmptyCache()
    except RuntimeError:
        print("The GPU is free to use. THere is no existing occupant")

    return V.T  # NOTE Our Ritz vector is (3n_atoms, n_modes) hence Transpose





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




# ================
# OBSOLETE
# ==================

def X_EmpiricalTransRotRitz(X,
                    device=torch.device(0), 
                    dtype_temp = torch.float64,
                    extremecase_batchsize = 100000,
                    extremecasedefinition = 350000,
                    ):
    with open("../Script/Burn/Coordinate/OBSOLETE_Ansatz_EmpiricalTranslationVec.pkl",'rb') as fn :
        proposed_translation = pickle.load(fn)
    with open("../Script/Burn/Coordinate/OBSOLETE_Ansatz_EmpiricalRotationMat.pkl",'rb') as fn :
        proposed_rotation = pickle.load(fn)


    n_atoms = int(X.shape[0])
    dof = int(n_atoms*3)
    n_modes = 6
    #print(n_atoms,n_modes, dof )
    V = torch.zeros((n_modes,dof), device=device, dtype=dtype_temp)
    
    proposed_translation_vecs = torch.tensor(proposed_translation, dtype= dtype_temp, device=device)
    proposed_rotation_mats = torch.tensor(proposed_rotation, dtype= dtype_temp, device=device)


    for i_mode in range(6):
        V[i_mode,:] = (X@proposed_rotation_mats[i_mode].T + proposed_translation_vecs[i_mode] - X).reshape(1,dof)
    

    # NOTE Normalise
    for iii in range(6):
        V[iii,:] /= torch.linalg.vector_norm(V[iii,:], ord = 2)

    # ====================
    # Reorthog
    # ====================
    V = V.T
    for ix in range(6):
        if ix == 0:
            continue
        V[:,ix] -= torch.mv(V[:,:ix], torch.mv( V[:, :ix].T,V[:,ix] ))
        V[:,ix] /= torch.sqrt(V[:, ix].T @ V[:, ix]) # TODO torch.matmul or mvs
        V[:,ix] -= torch.mv(V[:,:ix], torch.mv( V[:, :ix].T,V[:,ix] ))
        V[:,ix] /= torch.sqrt(V[:, ix].T @ V[:, ix])
    V = V.T
    """
    # NOTE If the ritz is exactly the same as the eigenvector presented i.e. of the symmetric hessian
    #      then it will never converge as we expect the eigvec are orthogonal! see problem 7.3 of ANL
    V += torch.nan_to_num(torch.randn((n_modes,dof), device=device, dtype=dtype_temp)  * (1/(9*n_atoms*n_atoms)), nan=0.0)
    """

    # NOTE Normalise
    for iii in range(6):
        V[iii,:] /= torch.linalg.vector_norm(V[iii,:], ord = 2)

    try:
        InchingLite.util.TorchEmptyCache()
    except RuntimeError:
        print("The GPU is free to use. THere is no existing occupant")
    print(V.T[:10,:])
    return V.T  # NOTE Our Ritz vector is (3n_atoms, n_modes) hence Transpose
