import torch
from torch import jit
import cupy
from cupyx.scipy import sparse as cupysparse
import cupy.cusparse
import numpy as np
import tqdm
import sys
sys.path.append('..')
sys.path.append('../Script/Burn/')
#import time
import InchingLite.util
#import InchingLite.Burn.LanczosIrlmAnsatz.T1
import InchingLite.Fuel.Coordinate.T1

import gc

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack



# ========================
# Correct
# ==========================




# NOTE This is the ungapped version. 
@torch.no_grad()
class Xnumpy_SparseCupyMatrixUngappped():
    def __init__(self, X, 
        batch_head = None, 
        maxleafsize = 100, rc_Gamma = 8.0,
        device  = torch.device(0), 
        User_PlusI = 1.0,
        dtype_temp = torch.float64, 
        X_precision = torch.cuda.DoubleTensor,
        NnzMinMaxDict = None,
        User_DictCharmmGuiPbc = None,
        ):
        super().__init__()
        




        self.device = device
        self.dtype_temp = dtype_temp
        self.nan = torch.finfo(dtype_temp).eps
        self.dtype_orig = X.dtype    
        self.n_atoms = X.shape[0]
        rc_Gamma /= 10.0
        self.rc_Gamma = rc_Gamma
        self.dof = int(3* self.n_atoms)
        self.User_PlusI = User_PlusI



        # =======================
        # Handling PBC
        # ==========================
        self.User_DictCharmmGuiPbc = User_DictCharmmGuiPbc
        if self.User_DictCharmmGuiPbc is not None:
            self.BoxsizeVector = cupy.array(
                                        np.array([ self.User_DictCharmmGuiPbc['RectBox_Xsize'],
                                            self.User_DictCharmmGuiPbc['RectBox_Ysize'],
                                            self.User_DictCharmmGuiPbc['RectBox_Zsize']])
                                )

            # NOTE It is correct iff the PBC is larger than the rc gamma.
            assert (User_DictCharmmGuiPbc['RectBox_Xsize'] > rc_Gamma), "ABORTED. The PBC box size X is smaller than rc gamma."
            assert (User_DictCharmmGuiPbc['RectBox_Ysize'] > rc_Gamma), "ABORTED. The PBC box size Y is smaller than rc gamma."

        else:
            self.BoxsizeVector = cupy.array(
                                        np.array([  0.0,
                                                    0.0,
                                                    0.0])
                                            )


        # NOTE Instruction to translate
        self.PbcXyInstruction = [   cupy.array([0,0,0]), #central unit
                                    cupy.array([1,0,0]), #xp
                                    cupy.array([-1,0,0]),#xm
                                    cupy.array([0,1,0]), #yp
                                    cupy.array([0,-1,0]), #ym
                                    cupy.array([1,1,0]), #xpyp
                                    cupy.array([1,-1,0]),#xpym
                                    cupy.array([-1,1,0]),#xmyp
                                    cupy.array([-1,-1,0]), #xmym
                            ]



        # =================================
        # Coordinates
        # ================================
        # NOTE Now rc_gamma is supposed nm
        self.X = cupy.array(X, dtype = self.dtype_orig )
        self.X_unsqueezed = cupy.expand_dims(self.X, 1)

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()

        # =======================
        # Size of batch
        # =======================
        if batch_head is None:
            batch_head = []
            PartitionTree = InchingLite.util.GetPartitionTree(range(self.n_atoms), maxleafsize = maxleafsize)
            FlattenPartitionTree_generator = InchingLite.util.FlattenPartitionTree(PartitionTree)
            batch_head = [0]
            # NOTE THe sorted here is necessary as it promote preallocation fo memory
            for i in sorted(FlattenPartitionTree_generator)[::-1]:
                batch_head.append(batch_head[-1] + i)
            batch_head = torch.LongTensor(batch_head)

            del PartitionTree, FlattenPartitionTree_generator
            gc.collect()
        self.batch_head = batch_head
        self.n_batch_min1 = self.batch_head.shape[0] -1



        if NnzMinMaxDict is None:
            self.LeftRightNnzBound = InchingLite.Fuel.Coordinate.T1.X_KdUngappedMinMaxNeighbor(X.detach().cpu().numpy(), 
                                                        rc_Gamma=rc_Gamma, maxleafsize = maxleafsize,
                                                        CollectStat = False, SliceForm = True )
        else:
            self.LeftRightNnzBound = NnzMinMaxDict




        # =======================================
        # Make some range vectors before hand
        # =========================================
        self.temp_index_ii = {} # called by size of einsum_rows
        self.temp_index_jj = {} # Called by batch index
        for i in range(self.n_batch_min1):
            # NOTE This will need to be left right bounded
            self.temp_index_jj[i] = np.arange(self.batch_head[i], self.batch_head[i+1], dtype= np.int64)  - self.LeftRightNnzBound[i][0][0]

            # NOTE Unchanged
            n_einsum_rows = self.temp_index_jj[i].shape[0]
            if n_einsum_rows not in self.temp_index_ii.keys():
                self.temp_index_ii[n_einsum_rows] = np.arange(n_einsum_rows, dtype= np.int64) 

        # =========================
        # Make Ungapped on CPU
        # =========================
        self.frontal_gap_offset = {} 
        self.ungapped_column_indices = {} 
        for i in range(self.n_batch_min1):
            # TODO Move to init and save it as a dictionary
            total_column_indices = torch.arange(self.LeftRightNnzBound[i][0][0],self.LeftRightNnzBound[i][-1][1], device='cpu')
            n_bounds = len(self.LeftRightNnzBound[i])
            if n_bounds == 1:
                temp_mask =  torch.ones_like(total_column_indices, dtype=torch.bool, device='cpu')
                self.frontal_gap_offset[i] = torch.tensor(0,dtype=torch.int32, device='cpu')

            else:

                temp_mask = torch.zeros_like(total_column_indices, dtype=torch.bool, device='cpu')
                
                first_frontal_record = torch.ones(1, dtype=torch.bool, device='cpu')#, device='cpu')
                last_band = 0
                for i_boundrange in range(len(self.LeftRightNnzBound[i])):
                    temp_mask[torch.arange( self.LeftRightNnzBound[i][i_boundrange][0]- self.LeftRightNnzBound[i][0][0],
                                            self.LeftRightNnzBound[i][i_boundrange][1]- self.LeftRightNnzBound[i][0][0], device='cpu')] = True
                    #print(first_frontal_record.device,  self.batch_head[i].device, )
                    if (self.LeftRightNnzBound[i][i_boundrange][1] >= self.batch_head[i]) & first_frontal_record:
                        first_frontal_record = torch.zeros(1, dtype=torch.bool, device='cpu')
                        last_band = self.LeftRightNnzBound[i][i_boundrange][1]- self.LeftRightNnzBound[i][0][0]

                frontal_gap_offset =  torch.sum(~temp_mask[:last_band])
                self.frontal_gap_offset[i] = torch.tensor(frontal_gap_offset,dtype=torch.int32, device='cpu')#.clone().detach().cpu().requires_grad_(False) #hare_memory_()
            self.ungapped_column_indices[i] = torch.masked_select(total_column_indices, temp_mask).numpy()#.clone().detach().cpu().requires_grad_(False).numpy() #.share_memory_()




    def ReturnNumberTotalBatch(self):
        return self.n_batch_min1 + 1

    def ReturnCupyH(self): # NOTE This is ARCHIVED
        """
        if help:
            This is a on-demand memory Hessian Matrix-vector product.
            The coeff gamma/distance is also synthesised on demand. 
            ultimately reducing the product memery footprint from O(n_atom ^2 ) to O(n_atom , leaf size)
            Hq = b
            q is a flat vector of size (3 n_atoms)
            b w/ the same shape is the product
        """
        
        #return
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        A = cupysparse.csr_matrix((self.n_atoms * 3, self.n_atoms * 3), dtype=cupy.float64) #cupysparse.eye(self.n_atoms * 3, dtype=np.float64, format='csc') # NOTE 32 easily produce nan!
        CumulativeStat = []
        compute_stream = cupy.cuda.stream.Stream(non_blocking=False)
        with compute_stream:
            for i in tqdm.tqdm(range(self.n_batch_min1)):

                    # ==============================================
                    # Differences 
                    # ==============================================
                    # Batching 
                    # NOTE While this is also pseudo linear bound considering the zeroing by coeff, 
                    #      it has a O(9bE[N]) with notorious coeff 9! unrealistic to store a (9*1000)*N_atom* 4 byte matrix...
                    # NOTE This is a broadcasted tensor
                    #      (m,n,3)    =    (n,3) - (m,1,3) 
                    #      I will denote the inter-point index as i and j 
                    #                    the inter-point generalised coordinate as pq
                    # NOTE Clearly the trace of each (i,j) block i.e. p==q gives the distance!
                    #      torch.diagonal(x, offset=0, dim1=0, dim2=1)
                    #print(self.rc_Gamma**2)


                    # TODO PDB format
                    Xij_batch = self.X[self.LeftRightNnzBound[i][0]:self.LeftRightNnzBound[i][1], :] - self.X_unsqueezed[self.batch_head[i]:self.batch_head[i+1], :,:]

                    # NOTE PDB format has 3 decimal digits
                    #      These are the fill-ins that will persist! 
                    #      (i,j,p) ~ 0.2 percent at this stage, but will propagate into the einsum!
                    fillin_index = cupy.where(cupy.abs(Xij_batch) < 1e-4)
                    einsum = cupy.einsum('ijp,ijq->ijpq', Xij_batch,Xij_batch)


                    
                    # ==============================
                    # Gamma/distance
                    # ==============================
                    n_einsum_rows = self.temp_index_jj[i].shape[0]

                    # NOTE Distance This is also an torch.einsum('ijkk->ij', einsum), but slower
                    coeff = cupy.sum(cupy.diagonal(einsum, offset=0, axis1=2, axis2=3),axis=2)
                    gamma_mask = cupy.greater(coeff, self.rc_Gamma**2)
                    n_einsum_cols = gamma_mask.shape[1]
                    gamma_mask = cupy.logical_or(gamma_mask, cupy.equal(coeff,0))
                    
                    coeff = cupy.reciprocal(coeff) * -1
                    cupy.putmask(coeff, gamma_mask, 0)
                    coeff = cupy.expand_dims(coeff, 2)
                    coeff = cupy.expand_dims(coeff, 2)
                    
                    # ================================
                    # Broadcast
                    # ================================
                    # Broadcast constant and zero.
                    einsum *= coeff

                    # NOTE Remove Fill-ins just in case
                    #      NOTE I decided not to remove it 
                    #einsum[fillin_index[0],fillin_index[1],fillin_index[2],:] = 0
                    #einsum[fillin_index[0],fillin_index[1],:,fillin_index[2]] = 0
                    
                    # NOTE cupy 11 put does not work when the to be put is a matrix. 
                    #      i.e. putting matrix to tensor.
                    row_sum = (-1* cupy.sum(einsum,axis = 1))
                    #print(row_sum[0:2])
                    """
                    for i_row in range(einsum.shape[0]):
                        einsum[
                            self.temp_index_ii[self.temp_index_jj[i].shape[0]][i_row],
                            self.temp_index_jj[i][i_row], 
                            0:3,0:3] = row_sum[i_row]
                        if self.temp_index_ii[self.temp_index_jj[i].shape[0]][i_row] == 62571:
                            print("LOOK", row_sum[i_row])
                            sys.exit()
                    """



                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,:,:] = row_sum
                    #if self.batch_head[i]*3 > 2000:
                    #    print(self.temp_index_ii[n_einsum_rows])
                    #    sys.exit()


                    #if self.batch_head[i]*3 > 60000:
                    #    print(einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i],:,:])
                    #    time.sleep(1)

                    # NOTE The A + I condition number trick
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,0,0] += self.User_PlusI
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,1,1] += self.User_PlusI
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,2,2] += self.User_PlusI
                    


                    # =========================
                    # Multiplicatino in batch
                    # =========================
                    einsum = cupy.ascontiguousarray(einsum)
                    einsum = cupy.transpose(einsum, axes=(0,2,1,3))
                    #einsum = cupy.moveaxis(einsum, (0,1,2,3), (0,2,1,3))
                    einsum = cupy.ascontiguousarray(einsum)
                    einsum_dim0 = einsum.shape[0]
                    einsum_dim1 = einsum.shape[1]
                    einsum_dim2 = einsum.shape[2]
                    einsum_dim3 = einsum.shape[3]

                    # NOTE reshape is unsafe??

                    einsum = cupy.reshape(einsum, (einsum_dim0,einsum_dim1, einsum_dim2*einsum_dim3), order='C')
                    einsum = cupy.reshape(einsum, (einsum_dim0 * einsum_dim1, einsum_dim2*einsum_dim3), order='C')
                    #if self.batch_head[i]*3 > 60000:
                    #    print(einsum[:10,:10])
                    #batchtotalnnz = cupy.sum((cupy.abs(einsum) > 0) )

                    
                    """
                    print('min at segment %s > 1e-6 %s out of %s nnz'%(
                        cupy.min(cupy.abs(einsum)[cupy.abs(einsum) > 0]),
                        cupy.sum((cupy.abs(einsum) > 0) & (cupy.abs(einsum) < 1e-6)),
                        cupy.sum((cupy.abs(einsum) > 0) )
                        ))
                    """
                    """
                    print('min at segment %s > 1e-7 %s out of %s nnz'%(
                        cupy.min(cupy.abs(einsum)[cupy.abs(einsum) > 0]),
                        cupy.sum((cupy.abs(einsum) > 0) & (cupy.abs(einsum) < 1e-7)),
                        cupy.sum((cupy.abs(einsum) > 0) )
                        ))
                    for i_power in [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4]:
                        CumulativeStat.append(["batch%s" %(i), 
                        float(i_power),
                        float(cupy.sum((cupy.abs(einsum) > 0) & (cupy.abs(einsum) < 10**i_power)) / batchtotalnnz),
                        ])
                    """
                    #print(cupy.max(cupy.abs(einsum)[cupy.abs(einsum) > 0]))
                    # TODO Assume pdb format 3 digit decimal (x_i - x_j) (y_i -y_j) / Rij^2
                    #      Any number below 1e-3*1e-3/8^2 = 1.5 * 1e-8 are fill-ins.
                    #      but I will defer this removal 
                    """
                    cupy.around(einsum, decimals=7, out=einsum)
                    einsum[cupy.abs(einsum) < 1e-7] = 0
                    #print(cupy.max(cupy.abs(einsum)[cupy.abs(einsum) > 0]), cupy.min(cupy.abs(einsum)[cupy.abs(einsum) > 0]))
                    """
                    einsum = cupy.nan_to_num(einsum, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
                    einsum = cupysparse.coo_matrix(einsum)
                    einsum.eliminate_zeros()

                    #compress = cupy.cusparse.csr2csr_compress(einsum, tol = 1e-7)
                    #einsum.data = compress.data
                    #einsum.indices = compress.indices
                    #einsum.indptr = compress.indptr

                    
                    # NOTE ISSUE https://github.com/cupy/cupy/issues/3223 
                    compute_stream.synchronize()
                    A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                    self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ] = einsum
                    
                    PARTZZZ_CheckCorrect = False
                    if PARTZZZ_CheckCorrect:
                        """
                        print( 'einsum4 dims, batch index', einsum_dim0, einsum_dim1, einsum_dim2, einsum_dim3, i)
                        print('A.shape >? bbbatch gead [i] *3, [i+1]*3' , A.shape, self.batch_head[i]*3, self.batch_head[i+1]*3)
                        print('A.shape >? leftright nnz bound', self.LeftRightNnzBound[i][0]*3,self.LeftRightNnzBound[i][1]*3)
                        """
                        evidence = ~(cupy.allclose( A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                    self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray(), einsum.toarray(), rtol=1e-05, atol=1e-08, equal_nan=False))
                        if evidence:
                            """
                            print('EEEEEEEEevidenccce %s' %(i), cupy.allclose( A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray(), einsum.toarray(), rtol=1e-05, atol=1e-08, equal_nan=False))
                            print(cupy.where(cupy.abs(A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray() - einsum.toarray()) > 1e-8), cupy.where(cupy.abs(A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray() - einsum.toarray()) > 1e-8)[0].shape)
                            print(self.batch_head[i]*3)
                            """
                            xbound = cupy.where(cupy.abs(A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray() - einsum.toarray()) > 1e-8)[1]
                            
                            print('EEEEEEEEevidenccce %s' %(i), cupy.allclose( A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray(), einsum.toarray(), rtol=1e-05, atol=1e-08, equal_nan=False))
                            plotarray = cupy.asnumpy(cupy.abs(A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray() - einsum.toarray()) > 1e-8)
                            import matplotlib.pyplot as plt
                            plt.figure(figsize = (30,30))
                            plt.imshow(plotarray,  vmax=None, vmin=-1e-18, aspect='equal')
                            plt.xlim((xbound.min(), xbound.max()))
                            plt.show()
                        """
                        while evidence:
                            A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                            ] = einsum
                            print()
                            evidence = ~(cupy.allclose( A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                    self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray(), einsum.toarray(), rtol=1e-05, atol=1e-08, equal_nan=False))
                            print(evidence)
                        """
                    # ==========================
                    # Memory cleansing
                    # ============================
                    coeff = None
                    gamma_mask = None
                    einsum = None
                    row_sum  = None
                    Xij_batch = None
                    fillin_index = None
                    compress = None
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
            compute_stream.synchronize()
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        print("WARNING. Output NNZ %s and it consumes %s GB. Okay?" %(A.count_nonzero(),mempool.total_bytes()/1024/1024/1024))
        #print(mempool.used_bytes()/1024/1024/1024)              # 0
        #print(mempool.total_bytes()/1024/1024/1024)             # 0
        #print(pinned_mempool.n_free_blocks())    # 0
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        df = pd.DataFrame(CumulativeStat, columns=['Batch', 'Power', 'LessThanCount'])
        print(df.loc[df['Power'] <= -6].groupby(by='Power').mean())
        sns.relplot(data=df, x='Power', y = 'LessThanCount',kind="line")
        plt.show()
        """
        return A

    def ReturnCupyHLowerTriangle(self, 
                User_MaxHalfNnzBufferSize = 1e8):
        """
        if help:
            # NOTE This will make the LowerTriangle (including the main diagonal)
            The coeff gamma/distance is also synthesised on the fly. 
            ultimately reducing the product memery footprint from O(n_atom ^2 ) to O(n_atom , leaf size)
            Hq = b
            q is a flat vector of size (3 n_atoms)
            b w/ the same shape is the product
        """

        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        # NOTE I confirm that this makes slower and it pull more resource to copying...
        #mempool = cupy.cuda.MemoryPool(cupy.cuda.memory.malloc_managed) # get unified pool
        #cupy.cuda.set_allocator(mempool.malloc)
        # NOTE At the end I chose a c style way to reconstruct it
        #A = cupysparse.csr_matrix((self.n_atoms * 3, self.n_atoms * 3), dtype=cupy.float64) #cupysparse.eye(self.n_atoms * 3, dtype=np.float64, format='csc') # NOTE 32 easily produce nan!

        User_MaxHalfNnzBufferSize = int(User_MaxHalfNnzBufferSize)
        # NOTE These are preassigned contig block.
        A_indices = cupy.empty(User_MaxHalfNnzBufferSize +1, dtype=cupy.int32) 
        A_indptr = cupy.empty(self.n_atoms * 3 + 1, dtype=cupy.int32)
        A_data = cupy.empty(User_MaxHalfNnzBufferSize +1, dtype=cupy.float64)
        data_startindex = 0
        indptr_startindex = 0
        indices_startindex = 0


        CumulativeStat = []
        compute_stream = cupy.cuda.stream.Stream(non_blocking=False)
        with compute_stream:
            for i in tqdm.tqdm(range(self.n_batch_min1)[:]):

                # ==============================================
                # Differences 
                # ==============================================
                # Batching 
                # NOTE While this is also pseudo linear bound considering the zeroing by coeff
                # NOTE This is a broadcasted tensor
                #      (m,n,3)    =    (n,3) - (m,1,3) 
                #      I will denote the inter-point index as i and j 
                #                    the inter-point generalised coordinate as pq
                # NOTE Clearly the trace of each (i,j) block i.e. p==q gives the distance!
                #      torch.diagonal(x, offset=0, dim1=0, dim2=1)
                #print(self.rc_Gamma**2)
                Xij_batch = self.X[self.ungapped_column_indices[i], :] - self.X_unsqueezed[self.batch_head[i]:self.batch_head[i+1], :,:]
                # NOTE PDB format has 3 decimal digits
                #      These are the fill-ins that will persist! 
                #      (i,j,p) ~ 0.2 percent at this stage, but will propagate into the einsum!
                fillin_index = cupy.where(cupy.abs(Xij_batch) < 1e-4)
                einsum = cupy.einsum('ijp,ijq->ijpq', Xij_batch,Xij_batch)


                
                # ==============================
                # Gamma/distance
                # ==============================
                

                # NOTE Distance This is also an torch.einsum('ijkk->ij', einsum), but slower
                coeff = cupy.sum(cupy.diagonal(einsum, offset=0, axis1=2, axis2=3),axis=2)
                gamma_mask = cupy.greater(coeff, self.rc_Gamma**2)
                gamma_mask = cupy.logical_or(gamma_mask, cupy.equal(coeff,0))
                



                coeff = cupy.reciprocal(coeff) * -1
                cupy.putmask(coeff, gamma_mask, 0)
                coeff = cupy.expand_dims(coeff, 2)
                coeff = cupy.expand_dims(coeff, 2)
                
                # ================================
                # Broadcast
                # ================================
                # Broadcast constant and zero.
                einsum *= coeff
                #print(einsum)
                # NOTE Remove Fill-ins just in case
                #einsum[fillin_index[0],fillin_index[1],fillin_index[2],:] = 0
                #einsum[fillin_index[0],fillin_index[1],:,fillin_index[2]] = 0




                # NOTE The idea to handle PBC is to do it for 
                if self.User_DictCharmmGuiPbc is not None:
                    # 
                    # NOTE Check if any point is at boundary
                    check_xp = cupy.sum(self.X[self.ungapped_column_indices[i], :][:,0] > (self.User_DictCharmmGuiPbc["X"][1] - self.rc_Gamma))
                    check_xm = cupy.sum(self.X[self.ungapped_column_indices[i], :][:,0] < (self.User_DictCharmmGuiPbc["X"][0] + self.rc_Gamma))
                    check_yp = cupy.sum(self.X[self.ungapped_column_indices[i], :][:,1] > (self.User_DictCharmmGuiPbc["Y"][1] - self.rc_Gamma))
                    check_ym = cupy.sum(self.X[self.ungapped_column_indices[i], :][:,1] < (self.User_DictCharmmGuiPbc["Y"][0] + self.rc_Gamma))
                    if cupy.sum(check_xp + check_xm + check_yp + check_ym) == 0:
                        # NOTE There are no points at boundary. Fine, we need to do nothing!
                        pass
                    else:
                        #print("PBC called in batch %s" %(i))
                        # TODO This treatment only works for cases where rcgamma is smaller than the box size.
                        # NOTE There are some points at boundary check 
                        #      I have skipped the first instruction which is formthe central image
                        for i_instruction in range(len(self.PbcXyInstruction))[1:]:

                            # Reset the batch and coeff
                            Xij_batch = None
                            fillin_index = None
                            gamma_mask = None
                            coeff = None

                            Xij_batch = self.X[self.ungapped_column_indices[i], :] + (self.BoxsizeVector * self.PbcXyInstruction[i_instruction]) - self.X_unsqueezed[self.batch_head[i]:self.batch_head[i+1], :,:]
                            # NOTE PDB format has 3 decimal digits i.e. 0.001 nm
                            #      These are the fill-ins that will persist! 
                            #      (i,j,p) ~ 0.2 percent at this stage, but will propagate into the einsum!
                            fillin_index = cupy.where(cupy.abs(Xij_batch) < 1e-4)
                            einsum_temp = cupy.einsum('ijp,ijq->ijpq', Xij_batch,Xij_batch)


                            
                            # ==============================
                            # Gamma/distance
                            # ==============================
                            # NOTE Distance This is also an torch.einsum('ijkk->ij', einsum), but slower
                            coeff = cupy.sum(cupy.diagonal(einsum_temp, offset=0, axis1=2, axis2=3),axis=2)
                            #print("PBC called in batch %s, but image %s with coeff " %(i, i_instruction), cupy.sum(coeff) )


                            gamma_mask = cupy.greater(coeff, self.rc_Gamma**2)
                            gamma_mask = cupy.logical_or(gamma_mask, cupy.equal(coeff,0))
                            
                            coeff = cupy.reciprocal(coeff) * -1
                            cupy.putmask(coeff, gamma_mask, 0)


                            if cupy.sum(cupy.abs(coeff)) < 1e-5:
                                #print(cupy.sum(cupy.abs(coeff)))
                                # Then it means it is not at neighborhood at all
                                # And we will skip it to avoid unnecessary steps
                                Xij_batch = None
                                fillin_index = None
                                gamma_mask = None
                                coeff = None
                                continue


                            #print("PBC called in batch %s and image %s" %(i, i_instruction), cupy.sum(cupy.abs(coeff)) )
                            coeff = cupy.expand_dims(coeff, 2)
                            coeff = cupy.expand_dims(coeff, 2)
                            
                            # ================================
                            # Broadcast
                            # ================================
                            # Broadcast constant and zero.
                            einsum_temp *= coeff
                            #print(einsum)
                            # NOTE Remove Fill-ins just in case
                            einsum_temp[fillin_index[0],fillin_index[1],fillin_index[2],:] = 0
                            einsum_temp[fillin_index[0],fillin_index[1],:,fillin_index[2]] = 0

                            einsum += einsum_temp


                            Xij_batch = None
                            fillin_index = None
                            gamma_mask = None
                            coeff = None


                else:
                    pass
                



                # ======================================
                # Row sum
                # ======================================

                # NOTE cupy 11 put does not work when the to be put is a matrix. 
                #      i.e. putting matrix to tensor.
                row_sum = (-1* cupy.sum(einsum,axis = 1))
                #print(row_sum)
                #sys.exit()


                n_einsum_rows = self.temp_index_jj[i].shape[0]
                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item(),:,:] = row_sum

                # NOTE The A + I condition number trick
                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item() ,0,0] += self.User_PlusI
                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item(),1,1] += self.User_PlusI
                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item(),2,2] += self.User_PlusI
                

                # =========================
                # Multiplicatino in batch
                # =========================
                einsum = cupy.transpose(einsum, axes=(0,2,1,3))
                einsum_dim0 = einsum.shape[0]
                einsum_dim1 = einsum.shape[1]
                einsum_dim2 = einsum.shape[2]
                einsum_dim3 = einsum.shape[3]

                # NOTE reshape is unsafe??
                einsum = cupy.reshape(einsum, (einsum_dim0,einsum_dim1, einsum_dim2*einsum_dim3), order='C')
                einsum = cupy.reshape(einsum, (einsum_dim0 * einsum_dim1, einsum_dim2*einsum_dim3), order='C')

                # TODO Assume pdb format 3 digit decimal (x_i - x_j) (y_i -y_j) / Rij^2
                #      Any number below 1e-3*1e-3/8^2 = 1.5 * 1e-8 are fill-ins.
                #      but I will defer this removal 

                einsum = cupy.nan_to_num(einsum, copy=True, nan=0.0, posinf=0.0, neginf=0.0) 
 
                einsum = cupysparse.tril(einsum, 
                                            k = (
                                                (self.batch_head[i]*3
                                                ) - self.LeftRightNnzBound[i][0][0]*3 - self.frontal_gap_offset[i].item()*3).item(),
                                            format = 'csr')
               # print((-1* cupy.sum(einsum.data)))
                        
                #einsum.eliminate_zeros() # NOTE This line ahas a bug revealed in Linux
                #print((-1* cupy.sum(einsum.data)))
                # NOTE The upper triu can be removed in coo
                # NOTE ISSUE https://github.com/cupy/cupy/issues/3223 
                compute_stream.synchronize()


                #print(einsum)
                #sys.exit()


                # =========================================
                # packing
                # =========================================
                # NOTE CPU SYNCHRONIZED as it is on numpy; we just run it once and for all.
                ungapped_indexing = np.repeat(self.ungapped_column_indices[i], 3).reshape(self.ungapped_column_indices[i].shape[0],3)
                ungapped_indexing *= 3
                ungapped_indexing[:,1] += 1
                ungapped_indexing[:,2] += 2
                ungapped_indexing = ungapped_indexing.flatten()

                # NOTE This correspond to einsum's column indexing one one.
                gapped_col_indexing = ungapped_indexing[einsum.indices.get()]

                # NOTE Version 2
                # NOTE CSR data
                cupy.put(A_data, 
                            cupy.arange(
                                data_startindex,
                                data_startindex + einsum.data.shape[0],
                                1, 
                                dtype=cupy.int32), 
                                einsum.data, mode='raise')
                data_startindex += einsum.data.shape[0]

                # NOTE CSR indices
                cupy.put(A_indices, 
                            cupy.arange(
                                indices_startindex,
                                indices_startindex + gapped_col_indexing.shape[0],
                                1, 
                                dtype=cupy.int32), 
                                cupy.array(gapped_col_indexing, dtype= cupy.int32), mode='raise')
                indices_startindex += gapped_col_indexing.shape[0]

                # NOTE CSR index pointer
                if indptr_startindex == 0:
                    lastindtr = 0
                else:
                    lastindtr = A_indptr[indptr_startindex]
                cupy.put(A_indptr, 
                        cupy.arange(
                            indptr_startindex,
                            indptr_startindex + einsum.indptr.shape[0],
                            1, 
                            dtype=cupy.int32), 
                            lastindtr  + einsum.indptr, mode='raise')
                indptr_startindex += einsum.indptr.shape[0] -1 
                #print(einsum.data.shape[0], gapped_col_indexing.shape[0], einsum.indptr.shape[0])
                #sys.exit()
                


                """
                # NOTE Version 0 
                #      THis requre einsum beung cooy and is mem demanding
                A[      self.batch_head[i]*3:self.batch_head[i+1]*3,
                            ungapped_indexing
                                        ] = einsum
                """
                """
                # NOTE Version 1
                #      This append requires copying. While it works the memory still oscillates...
                A.data = cupy.append(A.data, einsum.data )
                A.indices = cupy.append(A.indices, cupy.array(gapped_col_indexing, dtype= cupy.int32))
                lastindtr = A.indptr[-1]
                if i == 0:
                    A.indptr = einsum.indptr
                else:
                    A.indptr = cupy.append(A.indptr[:-1], lastindtr + einsum.indptr)
                """
                # ==========================
                # Memory cleansing
                # ============================
                coeff = None
                gamma_mask = None
                einsum = None
                einsum_ = None
                row_sum  = None
                Xij_batch = None
                fillin_index = None
                compress = None
                del coeff, gamma_mask, einsum, einsum_, row_sum, Xij_batch, 
                del fillin_index, ungapped_indexing
                #gc.collect() # NOTE Slow..
                cupy.get_default_memory_pool().free_all_blocks()
                cupy.get_default_pinned_memory_pool().free_all_blocks()
                mempool.free_all_blocks()

                torch.cuda.empty_cache()    
                """
                torch.cuda.reset_peak_memory_stats(0)
                torch.cuda.memory_allocated(0)
                torch.cuda.max_memory_allocated(0)
                """
            compute_stream.synchronize()

        gc.collect()

       
        # ===========================
        # Host/GPU comm
        # =============================
        # NOTE Send all back to host so that we can close memory correctly?
        #      NOTE OBSOLETE. It does not help much
        """
        B_data = cupy.asnumpy(A_data[:data_startindex])
        B_indices = cupy.asnumpy(A_indices[:indices_startindex])
        B_indptr = cupy.asnumpy( A_indptr[:indptr_startindex+1])

        A_data = None
        A_indices = None
        A_indptr = None

        del A_data, A_indices, A_indptr
        """

        #print(mempool.used_bytes())              # 0
        #print(mempool.total_bytes())             # 512
        #print(pinned_mempool.n_free_blocks())


        self.X = None
        self.X_unsqueezed = None
        del self.X, self.X_unsqueezed
        #print("WARNING. Output NNZ %s and the mempool consumed %s GB. Okay?" %(
        #                    data_startindex,
        #                    mempool.total_bytes()/1024/1024/1024))
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        mempool.free_all_blocks()

        torch.cuda.empty_cache()    
        """
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.memory_allocated(0)
        torch.cuda.max_memory_allocated(0)
        """
        #print(A.indices, A.indptr, A.indices.shape, A.indptr.shape)
        #print("WARNING. Freed %s GB from mempool." %(mempool.total_bytes()/1024/1024/1024))
        #print("WARNING. Expect %s GB to store the matrix. Okay?" %(
        #    (data_startindex*8/1024/1024/1024) + (indices_startindex*4/1024/1024/1024) + (indptr_startindex*4/1024/1024/1024)
        #))






        # NOTE Version 2
        #print('data', A_data[-100:], data_startindex, A_data.shape, A_data[data_startindex+2-10:data_startindex+2])
        #print('inidces', A_indices[-100:], indices_startindex, A_indices.shape, A_indices[indices_startindex+2-10:indices_startindex+2])
        #print('indptr', A_indptr[-100:], indptr_startindex, A_indptr.shape, A_indptr[indptr_startindex+1+2-10:indptr_startindex+1+2])
        """
        return cupysparse.csr_matrix(
            (   cupy.array(B_data), 
                cupy.array(B_indices), 
               cupy.array( B_indptr)), 
            shape = (self.n_atoms * 3, self.n_atoms * 3), 
            dtype=cupy.float64 )

        """



        #B_data = cupy.asnumpy(A_data[:data_startindex])
        #B_indices = cupy.asnumpy(A_indices[:indices_startindex])
        #B_indptr = cupy.asnumpy( A_indptr[:indptr_startindex+1])
        return cupysparse.csr_matrix(
            (   A_data[:data_startindex], 
                A_indices[:indices_startindex], 
                A_indptr[:indptr_startindex+1]), 
            shape = (self.n_atoms * 3, self.n_atoms * 3), 
            dtype=cupy.float64 )



    # NOTE This is Version 1 with fluctuating memory due to copy
    def ReturnCupyHLowerTriangle_ARCHIVED(self, 
                User_MaxNnzBufferSize = 1e8):
        """
        if help:
            # NOTE This will make the LowerTriangle (including the main diagonal)
            The coeff gamma/distance is also synthesised on demand. 
            ultimately reducing the product memery footprint from O(n_atom ^2 ) to O(n_atom , leaf size)
            Hq = b
            q is a flat vector of size (3 n_atoms)
            b w/ the same shape is the product
        """

        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        # NOTE I confirm that this makes slower and it pull more resource to copying...
        #mempool = cupy.cuda.MemoryPool(cupy.cuda.memory.malloc_managed) # get unified pool
        #cupy.cuda.set_allocator(mempool.malloc)
        # NOTE At the end I chose a c style way to reconstruct it
        A = cupysparse.csr_matrix((self.n_atoms * 3, self.n_atoms * 3), dtype=cupy.float64) #cupysparse.eye(self.n_atoms * 3, dtype=np.float64, format='csc') # NOTE 32 easily produce nan!
        User_MaxNnzBufferSize = int(User_MaxNnzBufferSize)
        # NOTE These are preassigned contig block.
        A_indices = cupy.empty(User_MaxNnzBufferSize, dtype=cupy.int32)
        A_indptr = cupy.empty(User_MaxNnzBufferSize, dtype=cupy.int32)
        A_data = cupy.empty(User_MaxNnzBufferSize, dtype=cupy.float64)
        data_startindex = 0
        indptr_startindex = 0
        indices_startindex = 0


        CumulativeStat = []
        compute_stream = cupy.cuda.stream.Stream(non_blocking=False)
        with compute_stream:
            for i in tqdm.tqdm(range(self.n_batch_min1)[:]):
                #print(self.ungapped_column_indices[i])
                #continue

                # ==============================================
                # Differences 
                # ==============================================
                # Batching 
                # NOTE While this is also pseudo linear bound considering the zeroing by coeff, 
                #      it has a O(9bE[N]) with notorious coeff 9! unrealistic to store a (9*1000)*N_atom* 4 byte matrix...
                # NOTE This is a broadcasted tensor
                #      (m,n,3)    =    (n,3) - (m,1,3) 
                #      I will denote the inter-point index as i and j 
                #                    the inter-point generalised coordinate as pq
                # NOTE Clearly the trace of each (i,j) block i.e. p==q gives the distance!
                #      torch.diagonal(x, offset=0, dim1=0, dim2=1)
                #print(self.rc_Gamma**2)

                # NOTE Many of these will be zeroed.
                #Xij_batch = self.X[self.LeftRightNnzBound[i][0]:self.LeftRightNnzBound[i][1], :] - self.X_unsqueezed[self.batch_head[i]:self.batch_head[i+1], :,:]
                Xij_batch = self.X[self.ungapped_column_indices[i], :] - self.X_unsqueezed[self.batch_head[i]:self.batch_head[i+1], :,:]
                # NOTE PDB format has 3 decimal digits
                #      These are the fill-ins that will persist! 
                #      (i,j,p) ~ 0.2 percent at this stage, but will propagate into the einsum!
                fillin_index = cupy.where(cupy.abs(Xij_batch) < 1e-4)
                einsum = cupy.einsum('ijp,ijq->ijpq', Xij_batch,Xij_batch)


                
                # ==============================
                # Gamma/distance
                # ==============================
                n_einsum_rows = self.temp_index_jj[i].shape[0]

                # NOTE Distance This is also an torch.einsum('ijkk->ij', einsum), but slower
                coeff = cupy.sum(cupy.diagonal(einsum, offset=0, axis1=2, axis2=3),axis=2)
                gamma_mask = cupy.greater(coeff, self.rc_Gamma**2)
                n_einsum_cols = gamma_mask.shape[1]
                gamma_mask = cupy.logical_or(gamma_mask, cupy.equal(coeff,0))
                
                coeff = cupy.reciprocal(coeff) * -1
                cupy.putmask(coeff, gamma_mask, 0)
                coeff = cupy.expand_dims(coeff, 2)
                coeff = cupy.expand_dims(coeff, 2)
                
                # ================================
                # Broadcast
                # ================================
                # Broadcast constant and zero.
                einsum *= coeff

                # NOTE Remove Fill-ins just in case
                einsum[fillin_index[0],fillin_index[1],fillin_index[2],:] = 0
                einsum[fillin_index[0],fillin_index[1],:,fillin_index[2]] = 0
                
                # NOTE cupy 11 put does not work when the to be put is a matrix. 
                #      i.e. putting matrix to tensor.
                row_sum = (-1* cupy.sum(einsum,axis = 1))
                



                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item(),:,:] = row_sum

                # NOTE The A + I condition number trick
                #einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,0,0] += self.User_PlusI
                #einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,1,1] += self.User_PlusI
                #einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,2,2] += self.User_PlusI
                
                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item() ,0,0] += self.User_PlusI
                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item(),1,1] += self.User_PlusI
                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item(),2,2] += self.User_PlusI
                

                # =========================
                # Multiplicatino in batch
                # =========================
                #einsum = cupy.ascontiguousarray(einsum)
                einsum = cupy.transpose(einsum, axes=(0,2,1,3))
                #einsum = cupy.ascontiguousarray(einsum)
                einsum_dim0 = einsum.shape[0]
                einsum_dim1 = einsum.shape[1]
                einsum_dim2 = einsum.shape[2]
                einsum_dim3 = einsum.shape[3]

                # NOTE reshape is unsafe??
                einsum = cupy.reshape(einsum, (einsum_dim0,einsum_dim1, einsum_dim2*einsum_dim3), order='C')
                einsum = cupy.reshape(einsum, (einsum_dim0 * einsum_dim1, einsum_dim2*einsum_dim3), order='C')

                # TODO Assume pdb format 3 digit decimal (x_i - x_j) (y_i -y_j) / Rij^2
                #      Any number below 1e-3*1e-3/8^2 = 1.5 * 1e-8 are fill-ins.
                #      but I will defer this removal 
                
                einsum = cupy.nan_to_num(einsum, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
                einsum = cupysparse.coo_matrix(einsum)
                

                #compress = cupy.cusparse.csr2csr_compress(einsum, tol = 1e-7)
                #einsum.data = compress.data
                #einsum.indices = compress.indices
                #einsum.indptr = compress.indptr
                #print(((self.batch_head[i]*3) - self.LeftRightNnzBound[i][0]*3).item())
                
                einsum = cupysparse.tril(einsum, 
                                            k = (
                                                (self.batch_head[i]*3
                                                ) - self.LeftRightNnzBound[i][0][0]*3 - self.frontal_gap_offset[i].item()*3).item(),
                                            format = 'csr')
                
                einsum.eliminate_zeros()
                # NOTE The upper triu can be removed in coo
                # NOTE ISSUE https://github.com/cupy/cupy/issues/3223 
                compute_stream.synchronize()

                # NOTE CPU SYNCHRONIZED as it is on numpy; we just run it once and for all.
                ungapped_indexing = np.repeat(self.ungapped_column_indices[i], 3).reshape(self.ungapped_column_indices[i].shape[0],3)
                ungapped_indexing *= 3
                ungapped_indexing[:,1] += 1
                ungapped_indexing[:,2] += 2
                ungapped_indexing = ungapped_indexing.flatten()

                # NOTE This correspond to einsum's column indexing one one.

                gapped_col_indexing = ungapped_indexing[einsum.indices.get()]

                #print(ungapped_indexing.flatten())
                #sys.exit()
                #print(ungapped_indexing.flatten())
                #sys.exit()
                """
                # NOTE Version 2
                # NOTE CSR data
                #A_data[data_startindex:data_startindex + einsum.data.shape[0]] = einsum.data
                cupy.put(A_data, 
                            cupy.arange(
                                data_startindex,
                                data_startindex + einsum.data.shape[0],
                                1, 
                                dtype=cupy.int32), 
                                einsum.data, mode='raise')
                data_startindex += einsum.data.shape[0]
                # NOTE CSR indices
                cupy.put(A_indices, 
                            cupy.arange(
                                indices_startindex,
                                indices_startindex + gapped_col_indexing.shape[0],
                                1, 
                                dtype=cupy.int32), 
                                cupy.array(gapped_col_indexing, dtype= cupy.int32), mode='raise')
                indices_startindex += gapped_col_indexing.shape[0]
                #A.indices = cupy.append(A.indices, cupy.array(gapped_col_indexing, dtype= cupy.int32))


                # NOTE CSR index pointer
                lastindtr = A_indptr[-1]
                cupy.put(A_indptr, 
                        cupy.arange(
                            indptr_startindex,
                            indptr_startindex + einsum.indptr.shape[0],
                            1, 
                            dtype=cupy.int32), 
                            lastindtr  + einsum.indptr, mode='raise')
                indptr_startindex += einsum.indptr.shape[0]-1
                """


                """
                # NOTE Version 0 
                #      THis requre einsum beung cooy and is mem demanding
                A[      self.batch_head[i]*3:self.batch_head[i+1]*3,
                            ungapped_indexing
                                        ] = einsum
                """
                
                # NOTE Version 1
                #      This append requires copying. While it works the memory still oscillates...
                A.data = cupy.append(A.data, einsum.data )
                A.indices = cupy.append(A.indices, cupy.array(gapped_col_indexing, dtype= cupy.int32))
                lastindtr = A.indptr[-1]
                if i == 0:
                    A.indptr = einsum.indptr
                else:
                    A.indptr = cupy.append(A.indptr[:-1], lastindtr + einsum.indptr)
                
                # ==========================
                # Memory cleansing
                # ============================
                coeff = None
                gamma_mask = None
                einsum = None
                einsum_ = None
                row_sum  = None
                Xij_batch = None
                fillin_index = None
                compress = None
                del coeff, gamma_mask, einsum, einsum_, row_sum, Xij_batch, 
                del fillin_index, ungapped_indexing

                cupy.get_default_memory_pool().free_all_blocks()
                cupy.get_default_pinned_memory_pool().free_all_blocks()
                mempool.free_all_blocks()

                torch.cuda.empty_cache()    
                torch.cuda.reset_peak_memory_stats(0)
                torch.cuda.memory_allocated(0)
                torch.cuda.max_memory_allocated(0)

            compute_stream.synchronize()

        print("WARNING. Output NNZ %s and it consumes %s GB. Okay?" %(A.count_nonzero(),mempool.total_bytes()/1024/1024/1024))
        #print(mempool.used_bytes()/1024/1024/1024)              # 0
        #print(mempool.total_bytes()/1024/1024/1024)             # 0
        #print(pinned_mempool.n_free_blocks())    # 0

        mempool.free_all_blocks()

        torch.cuda.empty_cache()    
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.memory_allocated(0)
        torch.cuda.max_memory_allocated(0)
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        df = pd.DataFrame(CumulativeStat, columns=['Batch', 'Power', 'LessThanCount'])
        print(df.loc[df['Power'] <= -6].groupby(by='Power').mean())
        sns.relplot(data=df, x='Power', y = 'LessThanCount',kind="line")
        plt.show()
        """
        #print(A.indices, A.indptr, A.indices.shape, A.indptr.shape)
        # NOTE Version 0
        # return A

        # NOTE Version 1
        return cupysparse.csr_matrix(
            (A.data, A.indices, A.indptr), 
            shape = (self.n_atoms * 3, self.n_atoms * 3), 
            dtype=cupy.float64 )





# ==========================
# OBOSOLETE
# ==========================




class OBOSOLETE_Xnumpy_SparseCupyMatrixUngappped():
    def __init__(self, X, 
        batch_head = None, 
        maxleafsize = 100, rc_Gamma = 8.0,
        device  = torch.device(0), 
        User_PlusI = 1.0,
        dtype_temp = torch.float64, 
        X_precision = torch.cuda.DoubleTensor,
        NnzMinMaxDict = None,

        ):
        super().__init__()
        
        #InchingLite.util.TorchMakePrecision(Precision = str(dtype_temp))
        #InchingLite.util.TorchEmptyCache()


        self.device = device
        self.dtype_temp = dtype_temp
        self.nan = torch.finfo(dtype_temp).eps
        self.dtype_orig = X.dtype    
        self.n_atoms = X.shape[0]
        self.rc_Gamma = rc_Gamma / 10.0
        self.dof = int(3* self.n_atoms)
        self.User_PlusI = User_PlusI

        #sys.exit()
        # NOTE Now rc_gamma is supposed nm
        #print(self.rc_Gamma)
        """
        X = X.type(X_precision)
        self.X = to_dlpack(X)
        self.X = cupy.from_dlpack(self.X)
        self.X_unsqueezed = cupy.expand_dims(self.X, 1)
        #print(self.X_unsqueezed)
        """
        self.X = cupy.array(X, dtype = self.dtype_orig )
        self.X_unsqueezed = cupy.expand_dims(self.X, 1)
        # NOTE DLPACK optimize
        """
        Xtemp = torch.tensor(X, dtype= torch.float64, requires_grad=False)
        self.X = cupy.from_dlpack(to_dlpack(Xtemp))
        self.X_unsqueezed = cupy.expand_dims(self.X, 1)
        Xtemp = None
        del Xtemp

        """
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        # =======================
        # Size of batch
        # =======================
        if batch_head is None:
            batch_head = []
            PartitionTree = InchingLite.util.GetPartitionTree(range(self.n_atoms), maxleafsize = maxleafsize)
            FlattenPartitionTree_generator = InchingLite.util.FlattenPartitionTree(PartitionTree)
            batch_head = [0]
            # NOTE THe sorted here is necessary as it promote preallocation fo memory
            for i in sorted(FlattenPartitionTree_generator)[::-1]:
                batch_head.append(batch_head[-1] + i)
            batch_head = torch.LongTensor(batch_head)

            del PartitionTree, FlattenPartitionTree_generator
            gc.collect()
        self.batch_head = batch_head
        self.n_batch_min1 = self.batch_head.shape[0] -1



        if NnzMinMaxDict is None:
            self.LeftRightNnzBound = InchingLite.Fuel.Coordinate.T1.X_KdUngappedMinMaxNeighbor(X.detach().cpu().numpy(), 
                                                        rc_Gamma=rc_Gamma, maxleafsize = maxleafsize,
                                                        CollectStat = False, SliceForm = True )
        else:
            self.LeftRightNnzBound = NnzMinMaxDict




        # =======================================
        # Make some range vectors before hand
        # =========================================
        self.temp_index_ii = {} # called by size of einsum_rows
        #self.temp_index_ii3 = {}
        self.temp_index_jj = {} # Called by batch index
        #self.temp_index_kk = {} # Called by batch index
        #self.temp_b = {}
        for i in range(self.n_batch_min1):
            # NOTE This will need to be left right bounded
            self.temp_index_jj[i] = np.arange(self.batch_head[i], self.batch_head[i+1], dtype= np.int64)  - self.LeftRightNnzBound[i][0][0]
            #self.temp_index_kk[i] = np.arange(self.batch_head[i]*3,self.batch_head[i+1]*3, dtype= np.int64) 

            # NOTE Unchanged
            n_einsum_rows = self.temp_index_jj[i].shape[0]
            if n_einsum_rows not in self.temp_index_ii.keys():
                self.temp_index_ii[n_einsum_rows] = np.arange(n_einsum_rows, dtype= np.int64) 
                #self.temp_index_ii3[n_einsum_rows] = torch.arange(n_einsum_rows*3, dtype= torch.long, device= device)
                #self.temp_b[n_einsum_rows] = torch.zeros(
                #                            n_einsum_rows*3, 
                #                            device= device, dtype=dtype_temp)
            #print(self.temp_index_kk[i],self.LeftRightNnzBound[i][0] )
        #sys.exit()

        # =========================
        # Make Ungapped on CPU
        # =========================
        self.frontal_gap_offset = {} 
        self.ungapped_column_indices = {} 
        for i in tqdm.tqdm(range(self.n_batch_min1)):
            # TODO Move to init and save it as a dictionary
            total_column_indices = torch.arange(self.LeftRightNnzBound[i][0][0],self.LeftRightNnzBound[i][-1][1], device='cpu')
            n_bounds = len(self.LeftRightNnzBound[i])
            if n_bounds == 1:
                temp_mask =  torch.ones_like(total_column_indices, dtype=torch.bool, device='cpu')
                self.frontal_gap_offset[i] = torch.tensor(0,dtype=torch.int32, device='cpu')


            else:
                temp_mask = torch.zeros_like(total_column_indices, dtype=torch.bool, device='cpu')
                
                first_frontal_record = torch.ones(1, dtype=torch.bool, device='cpu')#, device='cpu')
                last_band = 0
                for i_boundrange in range(len(self.LeftRightNnzBound[i])):
                    temp_mask[torch.arange( self.LeftRightNnzBound[i][i_boundrange][0]- self.LeftRightNnzBound[i][0][0],
                                            self.LeftRightNnzBound[i][i_boundrange][1]- self.LeftRightNnzBound[i][0][0], device='cpu')] = True
                    #print(first_frontal_record.device,  self.batch_head[i].device, )
                    if (self.LeftRightNnzBound[i][i_boundrange][1] >= self.batch_head[i]) & first_frontal_record:
                        first_frontal_record = torch.zeros(1, dtype=torch.bool, device='cpu')
                        last_band = self.LeftRightNnzBound[i][i_boundrange][1]- self.LeftRightNnzBound[i][0][0]

                frontal_gap_offset =  torch.sum(~temp_mask[:last_band])
                self.frontal_gap_offset[i] = torch.tensor(frontal_gap_offset,dtype=torch.int32, device='cpu')#.clone().detach().cpu().requires_grad_(False) #hare_memory_()
            self.ungapped_column_indices[i] = torch.masked_select(total_column_indices, temp_mask).numpy()#.clone().detach().cpu().requires_grad_(False).numpy() #.share_memory_()
        print("\nFinished Initialise Sparse Ungapped.\n")  
        print("\n\n\n")


    def ReturnNumberTotalBatch(self):
        return self.n_batch_min1 + 1

    def ReturnCupyH(self): # NOTE This is ARCHIVED
        """
        if help:
            This is a on-demand memory Hessian Matrix-vector product.
            The coeff gamma/distance is also synthesised on demand. 
            ultimately reducing the product memery footprint from O(n_atom ^2 ) to O(n_atom , leaf size)
            Hq = b
            q is a flat vector of size (3 n_atoms)
            b w/ the same shape is the product
        """
        
        #return
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        A = cupysparse.csr_matrix((self.n_atoms * 3, self.n_atoms * 3), dtype=cupy.float64) #cupysparse.eye(self.n_atoms * 3, dtype=np.float64, format='csc') # NOTE 32 easily produce nan!
        #return
        CumulativeStat = []
        #compute_stream = cupy.cuda.stream.Stream(non_blocking=False)
        #with compute_stream:
        for i in tqdm.tqdm(range(self.n_batch_min1)):

                    # ==============================================
                    # Differences 
                    # ==============================================
                    # Batching 
                    # NOTE While this is also pseudo linear bound considering the zeroing by coeff, 
                    #      it has a O(9bE[N]) with notorious coeff 9! unrealistic to store a (9*1000)*N_atom* 4 byte matrix...
                    # NOTE This is a broadcasted tensor
                    #      (m,n,3)    =    (n,3) - (m,1,3) 
                    #      I will denote the inter-point index as i and j 
                    #                    the inter-point generalised coordinate as pq
                    # NOTE Clearly the trace of each (i,j) block i.e. p==q gives the distance!
                    #      torch.diagonal(x, offset=0, dim1=0, dim2=1)
                    #print(self.rc_Gamma**2)


                    # TODO PDB format
                    Xij_batch = self.X[self.LeftRightNnzBound[i][0]:self.LeftRightNnzBound[i][1], :] - self.X_unsqueezed[self.batch_head[i]:self.batch_head[i+1], :,:]

                    # NOTE PDB format has 3 decimal digits
                    #      These are the fill-ins that will persist! 
                    #      (i,j,p) ~ 0.2 percent at this stage, but will propagate into the einsum!
                    fillin_index = cupy.where(cupy.abs(Xij_batch) < 1e-4)
                    einsum = cupy.einsum('ijp,ijq->ijpq', Xij_batch,Xij_batch)


                    
                    # ==============================
                    # Gamma/distance
                    # ==============================
                    n_einsum_rows = self.temp_index_jj[i].shape[0]

                    # NOTE Distance This is also an torch.einsum('ijkk->ij', einsum), but slower
                    coeff = cupy.sum(cupy.diagonal(einsum, offset=0, axis1=2, axis2=3),axis=2)
                    gamma_mask = cupy.greater(coeff, self.rc_Gamma**2)
                    n_einsum_cols = gamma_mask.shape[1]
                    gamma_mask = cupy.logical_or(gamma_mask, cupy.equal(coeff,0))
                    
                    coeff = cupy.reciprocal(coeff) * -1
                    cupy.putmask(coeff, gamma_mask, 0)
                    coeff = cupy.expand_dims(coeff, 2)
                    coeff = cupy.expand_dims(coeff, 2)
                    
                    # ================================
                    # Broadcast
                    # ================================
                    # Broadcast constant and zero.
                    einsum *= coeff

                    # NOTE Remove Fill-ins just in case
                    einsum[fillin_index[0],fillin_index[1],fillin_index[2],:] = 0
                    einsum[fillin_index[0],fillin_index[1],:,fillin_index[2]] = 0
                    
                    # NOTE cupy 11 put does not work when the to be put is a matrix. 
                    #      i.e. putting matrix to tensor.
                    row_sum = (-1* cupy.sum(einsum,axis = 1))
                    #print(row_sum[0:2])
                    """
                    for i_row in range(einsum.shape[0]):
                        einsum[
                            self.temp_index_ii[self.temp_index_jj[i].shape[0]][i_row],
                            self.temp_index_jj[i][i_row], 
                            0:3,0:3] = row_sum[i_row]
                        if self.temp_index_ii[self.temp_index_jj[i].shape[0]][i_row] == 62571:
                            print("LOOK", row_sum[i_row])
                            sys.exit()
                    """



                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,:,:] = row_sum
                    #if self.batch_head[i]*3 > 2000:
                    #    print(self.temp_index_ii[n_einsum_rows])
                    #    sys.exit()


                    #if self.batch_head[i]*3 > 60000:
                    #    print(einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i],:,:])
                    #    time.sleep(1)

                    # NOTE The A + I condition number trick
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,0,0] += self.User_PlusI
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,1,1] += self.User_PlusI
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,2,2] += self.User_PlusI
                    


                    # =========================
                    # Multiplicatino in batch
                    # =========================
                    einsum = cupy.ascontiguousarray(einsum)
                    einsum = cupy.transpose(einsum, axes=(0,2,1,3))
                    #einsum = cupy.moveaxis(einsum, (0,1,2,3), (0,2,1,3))
                    einsum = cupy.ascontiguousarray(einsum)
                    einsum_dim0 = einsum.shape[0]
                    einsum_dim1 = einsum.shape[1]
                    einsum_dim2 = einsum.shape[2]
                    einsum_dim3 = einsum.shape[3]

                    # NOTE reshape is unsafe??

                    einsum = cupy.reshape(einsum, (einsum_dim0,einsum_dim1, einsum_dim2*einsum_dim3), order='C')
                    einsum = cupy.reshape(einsum, (einsum_dim0 * einsum_dim1, einsum_dim2*einsum_dim3), order='C')
                    #if self.batch_head[i]*3 > 60000:
                    #    print(einsum[:10,:10])
                    #batchtotalnnz = cupy.sum((cupy.abs(einsum) > 0) )

                    
                    """
                    print('min at segment %s > 1e-6 %s out of %s nnz'%(
                        cupy.min(cupy.abs(einsum)[cupy.abs(einsum) > 0]),
                        cupy.sum((cupy.abs(einsum) > 0) & (cupy.abs(einsum) < 1e-6)),
                        cupy.sum((cupy.abs(einsum) > 0) )
                        ))
                    """
                    """
                    print('min at segment %s > 1e-7 %s out of %s nnz'%(
                        cupy.min(cupy.abs(einsum)[cupy.abs(einsum) > 0]),
                        cupy.sum((cupy.abs(einsum) > 0) & (cupy.abs(einsum) < 1e-7)),
                        cupy.sum((cupy.abs(einsum) > 0) )
                        ))
                    for i_power in [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4]:
                        CumulativeStat.append(["batch%s" %(i), 
                        float(i_power),
                        float(cupy.sum((cupy.abs(einsum) > 0) & (cupy.abs(einsum) < 10**i_power)) / batchtotalnnz),
                        ])
                    """
                    #print(cupy.max(cupy.abs(einsum)[cupy.abs(einsum) > 0]))
                    # TODO Assume pdb format 3 digit decimal (x_i - x_j) (y_i -y_j) / Rij^2
                    #      Any number below 1e-3*1e-3/8^2 = 1.5 * 1e-8 are fill-ins.
                    #      but I will defer this removal 
                    """
                    cupy.around(einsum, decimals=7, out=einsum)
                    einsum[cupy.abs(einsum) < 1e-7] = 0
                    #print(cupy.max(cupy.abs(einsum)[cupy.abs(einsum) > 0]), cupy.min(cupy.abs(einsum)[cupy.abs(einsum) > 0]))
                    """
                    einsum = cupy.nan_to_num(einsum, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
                    einsum = cupysparse.coo_matrix(einsum)
                    einsum.eliminate_zeros()

                    #compress = cupy.cusparse.csr2csr_compress(einsum, tol = 1e-7)
                    #einsum.data = compress.data
                    #einsum.indices = compress.indices
                    #einsum.indptr = compress.indptr

                    
                    # NOTE ISSUE https://github.com/cupy/cupy/issues/3223 
                    #compute_stream.synchronize()
                    A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                    self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ] = einsum
                    
                    PARTZZZ_CheckCorrect = False
                    if PARTZZZ_CheckCorrect:
                        """
                        print( 'einsum4 dims, batch index', einsum_dim0, einsum_dim1, einsum_dim2, einsum_dim3, i)
                        print('A.shape >? bbbatch gead [i] *3, [i+1]*3' , A.shape, self.batch_head[i]*3, self.batch_head[i+1]*3)
                        print('A.shape >? leftright nnz bound', self.LeftRightNnzBound[i][0]*3,self.LeftRightNnzBound[i][1]*3)
                        """
                        evidence = ~(cupy.allclose( A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                    self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray(), einsum.toarray(), rtol=1e-05, atol=1e-08, equal_nan=False))
                        if evidence:
                            """
                            print('EEEEEEEEevidenccce %s' %(i), cupy.allclose( A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray(), einsum.toarray(), rtol=1e-05, atol=1e-08, equal_nan=False))
                            print(cupy.where(cupy.abs(A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray() - einsum.toarray()) > 1e-8), cupy.where(cupy.abs(A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray() - einsum.toarray()) > 1e-8)[0].shape)
                            print(self.batch_head[i]*3)
                            """
                            xbound = cupy.where(cupy.abs(A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray() - einsum.toarray()) > 1e-8)[1]
                            
                            print('EEEEEEEEevidenccce %s' %(i), cupy.allclose( A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray(), einsum.toarray(), rtol=1e-05, atol=1e-08, equal_nan=False))
                            plotarray = cupy.asnumpy(cupy.abs(A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray() - einsum.toarray()) > 1e-8)
                            import matplotlib.pyplot as plt
                            plt.figure(figsize = (30,30))
                            plt.imshow(plotarray,  vmax=None, vmin=-1e-18, aspect='equal')
                            plt.xlim((xbound.min(), xbound.max()))
                            plt.show()
                        """
                        while evidence:
                            A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                            ] = einsum
                            print()
                            evidence = ~(cupy.allclose( A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                    self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray(), einsum.toarray(), rtol=1e-05, atol=1e-08, equal_nan=False))
                            print(evidence)
                        """
                    # ==========================
                    # Memory cleansing
                    # ============================
                    coeff = None
                    gamma_mask = None
                    einsum = None
                    row_sum  = None
                    Xij_batch = None
                    fillin_index = None
                    compress = None
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
            #compute_stream.synchronize()
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        print("WARNING. Output NNZ %s and it consumes %s GB. Okay?" %(A.count_nonzero(),mempool.total_bytes()/1024/1024/1024))
        #print(mempool.used_bytes()/1024/1024/1024)              # 0
        #print(mempool.total_bytes()/1024/1024/1024)             # 0
        #print(pinned_mempool.n_free_blocks())    # 0
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        df = pd.DataFrame(CumulativeStat, columns=['Batch', 'Power', 'LessThanCount'])
        print(df.loc[df['Power'] <= -6].groupby(by='Power').mean())
        sns.relplot(data=df, x='Power', y = 'LessThanCount',kind="line")
        plt.show()
        """
        return A

    def ReturnCupyHLowerTriangle(self, 
                User_MaxHalfNnzBufferSize = 1e8):
        """
        if help:
            # NOTE This will make the LowerTriangle (including the main diagonal)
            The coeff gamma/distance is also synthesised on demand. 
            ultimately reducing the product memery footprint from O(n_atom ^2 ) to O(n_atom , leaf size)
            Hq = b
            q is a flat vector of size (3 n_atoms)
            b w/ the same shape is the product
        """

        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        # NOTE I confirm that this makes slower and it pull more resource to copying...
        #mempool = cupy.cuda.MemoryPool(cupy.cuda.memory.malloc_managed) # get unified pool
        #cupy.cuda.set_allocator(mempool.malloc)
        # NOTE At the end I chose a c style way to reconstruct it
        #A = cupysparse.csr_matrix((self.n_atoms * 3, self.n_atoms * 3), dtype=cupy.float64) #cupysparse.eye(self.n_atoms * 3, dtype=np.float64, format='csc') # NOTE 32 easily produce nan!

        User_MaxHalfNnzBufferSize = int(User_MaxHalfNnzBufferSize)
        # NOTE These are preassigned contig block.
        A_indices = cupy.empty(User_MaxHalfNnzBufferSize +1, dtype=cupy.int32) 
        A_indptr = cupy.empty(self.n_atoms * 3 + 1, dtype=cupy.int32)
        A_data = cupy.empty(User_MaxHalfNnzBufferSize +1, dtype=cupy.float64)
        data_startindex = 0
        indptr_startindex = 0
        indices_startindex = 0
        print("\n\n\n")
        print("Start to build matrix\n")
        print("\n\n\n")
        CumulativeStat = []
        #compute_stream = cupy.cuda.stream.Stream(non_blocking=False)
        #with compute_stream:
        # NOTE Added tqdm here 202303
        for i in tqdm.tqdm(range(self.n_batch_min1)[:]):
                #print(self.ungapped_column_indices[i])
                #continue

                # ==============================================
                # Differences 
                # ==============================================
                # Batching 
                # NOTE While this is also pseudo linear bound considering the zeroing by coeff
                # NOTE This is a broadcasted tensor
                #      (m,n,3)    =    (n,3) - (m,1,3) 
                #      I will denote the inter-point index as i and j 
                #                    the inter-point generalised coordinate as pq
                # NOTE Clearly the trace of each (i,j) block i.e. p==q gives the distance!
                #      torch.diagonal(x, offset=0, dim1=0, dim2=1)
                #print(self.rc_Gamma**2)

                # NOTE Many of these will be zeroed.
                #Xij_batch = self.X[self.LeftRightNnzBound[i][0]:self.LeftRightNnzBound[i][1], :] - self.X_unsqueezed[self.batch_head[i]:self.batch_head[i+1], :,:]
                Xij_batch = self.X[self.ungapped_column_indices[i], :] - self.X_unsqueezed[self.batch_head[i]:self.batch_head[i+1], :,:]
                # NOTE PDB format has 3 decimal digits
                #      These are the fill-ins that will persist! 
                #      (i,j,p) ~ 0.2 percent at this stage, but will propagate into the einsum!
                fillin_index = cupy.where(cupy.abs(Xij_batch) < 1e-4)
                einsum = cupy.einsum('ijp,ijq->ijpq', Xij_batch,Xij_batch)


                
                # ==============================
                # Gamma/distance
                # ==============================
                n_einsum_rows = self.temp_index_jj[i].shape[0]

                # NOTE Distance This is also an torch.einsum('ijkk->ij', einsum), but slower
                coeff = cupy.sum(cupy.diagonal(einsum, offset=0, axis1=2, axis2=3),axis=2)
                gamma_mask = cupy.greater(coeff, self.rc_Gamma**2)
                n_einsum_cols = gamma_mask.shape[1]
                gamma_mask = cupy.logical_or(gamma_mask, cupy.equal(coeff,0))
                
                coeff = cupy.reciprocal(coeff) * -1
                cupy.putmask(coeff, gamma_mask, 0)
                coeff = cupy.expand_dims(coeff, 2)
                coeff = cupy.expand_dims(coeff, 2)
                
                # ================================
                # Broadcast
                # ================================
                # Broadcast constant and zero.
                einsum *= coeff
                #print(einsum)

                # NOTE Remove Fill-ins just in case
                einsum[fillin_index[0],fillin_index[1],fillin_index[2],:] = 0
                einsum[fillin_index[0],fillin_index[1],:,fillin_index[2]] = 0
                
                #print(einsum)


                # NOTE cupy 11 put does not work when the to be put is a matrix. 
                #      i.e. putting matrix to tensor.
                row_sum = (-1* cupy.sum(einsum,axis = 1))
                
                

                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item(),:,:] = row_sum

                # NOTE The A + I condition number trick
                #einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,0,0] += self.User_PlusI
                #einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,1,1] += self.User_PlusI
                #einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,2,2] += self.User_PlusI
                
                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item() ,0,0] += self.User_PlusI
                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item(),1,1] += self.User_PlusI
                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item(),2,2] += self.User_PlusI
                

                
                # =========================
                # Multiplicatino in batch
                # =========================
                #einsum = cupy.ascontiguousarray(einsum)
                einsum = cupy.transpose(einsum, axes=(0,2,1,3))
                #einsum = cupy.ascontiguousarray(einsum)
                einsum_dim0 = einsum.shape[0]
                einsum_dim1 = einsum.shape[1]
                einsum_dim2 = einsum.shape[2]
                einsum_dim3 = einsum.shape[3]

                # NOTE reshape is unsafe??
                einsum = cupy.reshape(einsum, (einsum_dim0,einsum_dim1, einsum_dim2*einsum_dim3), order='C')
                einsum = cupy.reshape(einsum, (einsum_dim0 * einsum_dim1, einsum_dim2*einsum_dim3), order='C')

                # TODO Assume pdb format 3 digit decimal (x_i - x_j) (y_i -y_j) / Rij^2
                #      Any number below 1e-3*1e-3/8^2 = 1.5 * 1e-8 are fill-ins.
                #      but I will defer this removal 
                
                einsum = cupy.nan_to_num(einsum, copy=True, nan=0.0, posinf=0.0, neginf=0.0)


                #einsum = cupysparse.coo_matrix(einsum)                
                einsum = cupysparse.tril(einsum, 
                                            k = (
                                                (self.batch_head[i]*3
                                                ) - self.LeftRightNnzBound[i][0][0]*3 - self.frontal_gap_offset[i].item()*3).item(),
                                            format = 'csr')
                #print((-1* cupy.sum(einsum.data)))           
                # einsum.eliminate_zeros() # NOTE This line has a bug revealed in Linux
                #print((-1* cupy.sum(einsum.data)))
                # NOTE The upper triu can be removed in coo
                # NOTE ISSUE https://github.com/cupy/cupy/issues/3223 
                #compute_stream.synchronize()


                #print(einsum)
                #sys.exit()


                # =========================================
                # packing
                # =========================================
                # NOTE CPU SYNCHRONIZED as it is on numpy; we just run it once and for all.
                ungapped_indexing = np.repeat(self.ungapped_column_indices[i], 3).reshape(self.ungapped_column_indices[i].shape[0],3)
                ungapped_indexing *= 3
                ungapped_indexing[:,1] += 1
                ungapped_indexing[:,2] += 2
                ungapped_indexing = ungapped_indexing.flatten()

                # NOTE This correspond to einsum's column indexing one one.
                gapped_col_indexing = ungapped_indexing[einsum.indices.get()]

                # NOTE Version 2
                # NOTE CSR data
                #A_data[data_startindex:data_startindex + einsum.data.shape[0]] = einsum.data
                cupy.put(A_data, 
                            cupy.arange(
                                data_startindex,
                                data_startindex + einsum.data.shape[0],
                                1, 
                                dtype=cupy.int32), 
                                einsum.data, mode='raise')
                data_startindex += einsum.data.shape[0]

                # NOTE CSR indices
                cupy.put(A_indices, 
                            cupy.arange(
                                indices_startindex,
                                indices_startindex + gapped_col_indexing.shape[0],
                                1, 
                                dtype=cupy.int32), 
                                cupy.array(gapped_col_indexing, dtype= cupy.int32), mode='raise')
                indices_startindex += gapped_col_indexing.shape[0]

                # NOTE CSR index pointer
                if indptr_startindex == 0:
                    lastindtr = 0
                else:
                    lastindtr = A_indptr[indptr_startindex]
                cupy.put(A_indptr, 
                        cupy.arange(
                            indptr_startindex,
                            indptr_startindex + einsum.indptr.shape[0],
                            1, 
                            dtype=cupy.int32), 
                            lastindtr  + einsum.indptr, mode='raise')
                indptr_startindex += einsum.indptr.shape[0] -1 
                #print(einsum.data.shape[0], gapped_col_indexing.shape[0], einsum.indptr.shape[0])
                #sys.exit()
                #if i % 100 == 0:
                #    print("%s %s\n" %(i, indptr_startindex ))

                """
                # NOTE Version 0 
                #      THis requre einsum beung cooy and is mem demanding
                A[      self.batch_head[i]*3:self.batch_head[i+1]*3,
                            ungapped_indexing
                                        ] = einsum
                """
                """
                # NOTE Version 1
                #      This append requires copying. While it works the memory still oscillates...
                A.data = cupy.append(A.data, einsum.data )
                A.indices = cupy.append(A.indices, cupy.array(gapped_col_indexing, dtype= cupy.int32))
                lastindtr = A.indptr[-1]
                if i == 0:
                    A.indptr = einsum.indptr
                else:
                    A.indptr = cupy.append(A.indptr[:-1], lastindtr + einsum.indptr)
                """
                # ==========================
                # Memory cleansing
                # ============================
                coeff = None
                gamma_mask = None
                einsum = None
                einsum_ = None
                row_sum  = None
                Xij_batch = None
                fillin_index = None
                compress = None
                del coeff, gamma_mask, einsum, einsum_, row_sum, Xij_batch, 
                del fillin_index, ungapped_indexing
                #gc.collect() # NOTE Slow..
                cupy.get_default_memory_pool().free_all_blocks()
                cupy.get_default_pinned_memory_pool().free_all_blocks()
                mempool.free_all_blocks()

                torch.cuda.empty_cache()    
                torch.cuda.reset_peak_memory_stats(0)
                torch.cuda.memory_allocated(0)
                torch.cuda.max_memory_allocated(0)

            #compute_stream.synchronize()

        gc.collect()

       
        # ===========================
        # Host/GPU comm
        # =============================
        # NOTE Send all back to host so that we can close memory correctly?
        #      NOTE OBSOLETE. It does not help much
        """
        B_data = cupy.asnumpy(A_data[:data_startindex])
        B_indices = cupy.asnumpy(A_indices[:indices_startindex])
        B_indptr = cupy.asnumpy( A_indptr[:indptr_startindex+1])

        A_data = None
        A_indices = None
        A_indptr = None

        del A_data, A_indices, A_indptr
        """

        #print(mempool.used_bytes())              # 0
        #print(mempool.total_bytes())             # 512
        #print(pinned_mempool.n_free_blocks())


        self.X = None
        self.X_unsqueezed = None
        del self.X, self.X_unsqueezed
        #print("WARNING. Output NNZ %s and the mempool consumed %s GB. Okay?" %(
        #                    data_startindex,
        #                    mempool.total_bytes()/1024/1024/1024))
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        mempool.free_all_blocks()

        torch.cuda.empty_cache()    
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.memory_allocated(0)
        torch.cuda.max_memory_allocated(0)

        #print(A.indices, A.indptr, A.indices.shape, A.indptr.shape)
        #print("WARNING. Freed %s GB from mempool." %(mempool.total_bytes()/1024/1024/1024))
        #print("WARNING. Expect %s GB to store the matrix. Okay?" %(
        #    (data_startindex*8/1024/1024/1024) + (indices_startindex*4/1024/1024/1024) + (indptr_startindex*4/1024/1024/1024)
        #))






        # NOTE Version 2
        #print('data', A_data[-100:], data_startindex, A_data.shape, A_data[data_startindex+2-10:data_startindex+2])
        #print('inidces', A_indices[-100:], indices_startindex, A_indices.shape, A_indices[indices_startindex+2-10:indices_startindex+2])
        #print('indptr', A_indptr[-100:], indptr_startindex, A_indptr.shape, A_indptr[indptr_startindex+1+2-10:indptr_startindex+1+2])
        """
        return cupysparse.csr_matrix(
            (   cupy.array(B_data), 
                cupy.array(B_indices), 
               cupy.array( B_indptr)), 
            shape = (self.n_atoms * 3, self.n_atoms * 3), 
            dtype=cupy.float64 )

        """
        print("\n\n\n")

        print("Pulling Hessian to GPU\n")
        print("\n\n\n")
        #B_data = cupy.asnumpy(A_data[:data_startindex])
        #B_indices = cupy.asnumpy(A_indices[:indices_startindex])
        #B_indptr = cupy.asnumpy( A_indptr[:indptr_startindex+1])
        return cupysparse.csr_matrix(
            (   A_data[:data_startindex], 
                A_indices[:indices_startindex], 
                A_indptr[:indptr_startindex+1]), 
            shape = (self.n_atoms * 3, self.n_atoms * 3), 
            dtype=cupy.float64 )



    # NOTE This is Version 1 with fluctuating memory due to copy
    def ReturnCupyHLowerTriangle_ARCHIVED(self, 
                User_MaxNnzBufferSize = 1e8):
        """
        if help:
            # NOTE This will make the LowerTriangle (including the main diagonal)
            The coeff gamma/distance is also synthesised on demand. 
            ultimately reducing the product memery footprint from O(n_atom ^2 ) to O(n_atom , leaf size)
            Hq = b
            q is a flat vector of size (3 n_atoms)
            b w/ the same shape is the product
        """

        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        # NOTE I confirm that this makes slower and it pull more resource to copying...
        #mempool = cupy.cuda.MemoryPool(cupy.cuda.memory.malloc_managed) # get unified pool
        #cupy.cuda.set_allocator(mempool.malloc)
        # NOTE At the end I chose a c style way to reconstruct it
        A = cupysparse.csr_matrix((self.n_atoms * 3, self.n_atoms * 3), dtype=cupy.float64) #cupysparse.eye(self.n_atoms * 3, dtype=np.float64, format='csc') # NOTE 32 easily produce nan!
        User_MaxNnzBufferSize = int(User_MaxNnzBufferSize)
        # NOTE These are preassigned contig block.
        A_indices = cupy.empty(User_MaxNnzBufferSize, dtype=cupy.int32)
        A_indptr = cupy.empty(User_MaxNnzBufferSize, dtype=cupy.int32)
        A_data = cupy.empty(User_MaxNnzBufferSize, dtype=cupy.float64)
        data_startindex = 0
        indptr_startindex = 0
        indices_startindex = 0


        CumulativeStat = []
        compute_stream = cupy.cuda.stream.Stream(non_blocking=False)
        with compute_stream:
            for i in tqdm.tqdm(range(self.n_batch_min1)[:]):
                #print(self.ungapped_column_indices[i])
                #continue

                # ==============================================
                # Differences 
                # ==============================================
                # Batching 
                # NOTE While this is also pseudo linear bound considering the zeroing by coeff, 
                #      it has a O(9bE[N]) with notorious coeff 9! unrealistic to store a (9*1000)*N_atom* 4 byte matrix...
                # NOTE This is a broadcasted tensor
                #      (m,n,3)    =    (n,3) - (m,1,3) 
                #      I will denote the inter-point index as i and j 
                #                    the inter-point generalised coordinate as pq
                # NOTE Clearly the trace of each (i,j) block i.e. p==q gives the distance!
                #      torch.diagonal(x, offset=0, dim1=0, dim2=1)
                #print(self.rc_Gamma**2)

                # NOTE Many of these will be zeroed.
                #Xij_batch = self.X[self.LeftRightNnzBound[i][0]:self.LeftRightNnzBound[i][1], :] - self.X_unsqueezed[self.batch_head[i]:self.batch_head[i+1], :,:]
                Xij_batch = self.X[self.ungapped_column_indices[i], :] - self.X_unsqueezed[self.batch_head[i]:self.batch_head[i+1], :,:]
                # NOTE PDB format has 3 decimal digits
                #      These are the fill-ins that will persist! 
                #      (i,j,p) ~ 0.2 percent at this stage, but will propagate into the einsum!
                fillin_index = cupy.where(cupy.abs(Xij_batch) < 1e-4)
                einsum = cupy.einsum('ijp,ijq->ijpq', Xij_batch,Xij_batch)


                
                # ==============================
                # Gamma/distance
                # ==============================
                n_einsum_rows = self.temp_index_jj[i].shape[0]

                # NOTE Distance This is also an torch.einsum('ijkk->ij', einsum), but slower
                coeff = cupy.sum(cupy.diagonal(einsum, offset=0, axis1=2, axis2=3),axis=2)
                gamma_mask = cupy.greater(coeff, self.rc_Gamma**2)
                n_einsum_cols = gamma_mask.shape[1]
                gamma_mask = cupy.logical_or(gamma_mask, cupy.equal(coeff,0))
                
                coeff = cupy.reciprocal(coeff) * -1
                cupy.putmask(coeff, gamma_mask, 0)
                coeff = cupy.expand_dims(coeff, 2)
                coeff = cupy.expand_dims(coeff, 2)
                
                # ================================
                # Broadcast
                # ================================
                # Broadcast constant and zero.
                einsum *= coeff

                # NOTE Remove Fill-ins just in case
                einsum[fillin_index[0],fillin_index[1],fillin_index[2],:] = 0
                einsum[fillin_index[0],fillin_index[1],:,fillin_index[2]] = 0
                
                # NOTE cupy 11 put does not work when the to be put is a matrix. 
                #      i.e. putting matrix to tensor.
                row_sum = (-1* cupy.sum(einsum,axis = 1))
                



                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item(),:,:] = row_sum

                # NOTE The A + I condition number trick
                #einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,0,0] += self.User_PlusI
                #einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,1,1] += self.User_PlusI
                #einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,2,2] += self.User_PlusI
                
                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item() ,0,0] += self.User_PlusI
                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item(),1,1] += self.User_PlusI
                einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] - self.frontal_gap_offset[i].item(),2,2] += self.User_PlusI
                

                # =========================
                # Multiplicatino in batch
                # =========================
                #einsum = cupy.ascontiguousarray(einsum)
                einsum = cupy.transpose(einsum, axes=(0,2,1,3))
                #einsum = cupy.ascontiguousarray(einsum)
                einsum_dim0 = einsum.shape[0]
                einsum_dim1 = einsum.shape[1]
                einsum_dim2 = einsum.shape[2]
                einsum_dim3 = einsum.shape[3]

                # NOTE reshape is unsafe??
                einsum = cupy.reshape(einsum, (einsum_dim0,einsum_dim1, einsum_dim2*einsum_dim3), order='C')
                einsum = cupy.reshape(einsum, (einsum_dim0 * einsum_dim1, einsum_dim2*einsum_dim3), order='C')

                # TODO Assume pdb format 3 digit decimal (x_i - x_j) (y_i -y_j) / Rij^2
                #      Any number below 1e-3*1e-3/8^2 = 1.5 * 1e-8 are fill-ins.
                #      but I will defer this removal 
                
                einsum = cupy.nan_to_num(einsum, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
                einsum = cupysparse.coo_matrix(einsum)
                

                #compress = cupy.cusparse.csr2csr_compress(einsum, tol = 1e-7)
                #einsum.data = compress.data
                #einsum.indices = compress.indices
                #einsum.indptr = compress.indptr
                #print(((self.batch_head[i]*3) - self.LeftRightNnzBound[i][0]*3).item())
                
                einsum = cupysparse.tril(einsum, 
                                            k = (
                                                (self.batch_head[i]*3
                                                ) - self.LeftRightNnzBound[i][0][0]*3 - self.frontal_gap_offset[i].item()*3).item(),
                                            format = 'csr')
                
                einsum.eliminate_zeros()
                # NOTE The upper triu can be removed in coo
                # NOTE ISSUE https://github.com/cupy/cupy/issues/3223 
                compute_stream.synchronize()

                # NOTE CPU SYNCHRONIZED as it is on numpy; we just run it once and for all.
                ungapped_indexing = np.repeat(self.ungapped_column_indices[i], 3).reshape(self.ungapped_column_indices[i].shape[0],3)
                ungapped_indexing *= 3
                ungapped_indexing[:,1] += 1
                ungapped_indexing[:,2] += 2
                ungapped_indexing = ungapped_indexing.flatten()

                # NOTE This correspond to einsum's column indexing one one.

                gapped_col_indexing = ungapped_indexing[einsum.indices.get()]

                #print(ungapped_indexing.flatten())
                #sys.exit()
                #print(ungapped_indexing.flatten())
                #sys.exit()
                """
                # NOTE Version 2
                # NOTE CSR data
                #A_data[data_startindex:data_startindex + einsum.data.shape[0]] = einsum.data
                cupy.put(A_data, 
                            cupy.arange(
                                data_startindex,
                                data_startindex + einsum.data.shape[0],
                                1, 
                                dtype=cupy.int32), 
                                einsum.data, mode='raise')
                data_startindex += einsum.data.shape[0]
                # NOTE CSR indices
                cupy.put(A_indices, 
                            cupy.arange(
                                indices_startindex,
                                indices_startindex + gapped_col_indexing.shape[0],
                                1, 
                                dtype=cupy.int32), 
                                cupy.array(gapped_col_indexing, dtype= cupy.int32), mode='raise')
                indices_startindex += gapped_col_indexing.shape[0]
                #A.indices = cupy.append(A.indices, cupy.array(gapped_col_indexing, dtype= cupy.int32))


                # NOTE CSR index pointer
                lastindtr = A_indptr[-1]
                cupy.put(A_indptr, 
                        cupy.arange(
                            indptr_startindex,
                            indptr_startindex + einsum.indptr.shape[0],
                            1, 
                            dtype=cupy.int32), 
                            lastindtr  + einsum.indptr, mode='raise')
                indptr_startindex += einsum.indptr.shape[0]-1
                """


                """
                # NOTE Version 0 
                #      THis requre einsum beung cooy and is mem demanding
                A[      self.batch_head[i]*3:self.batch_head[i+1]*3,
                            ungapped_indexing
                                        ] = einsum
                """
                
                # NOTE Version 1
                #      This append requires copying. While it works the memory still oscillates...
                A.data = cupy.append(A.data, einsum.data )
                A.indices = cupy.append(A.indices, cupy.array(gapped_col_indexing, dtype= cupy.int32))
                lastindtr = A.indptr[-1]
                if i == 0:
                    A.indptr = einsum.indptr
                else:
                    A.indptr = cupy.append(A.indptr[:-1], lastindtr + einsum.indptr)
                
                # ==========================
                # Memory cleansing
                # ============================
                coeff = None
                gamma_mask = None
                einsum = None
                einsum_ = None
                row_sum  = None
                Xij_batch = None
                fillin_index = None
                compress = None
                del coeff, gamma_mask, einsum, einsum_, row_sum, Xij_batch, 
                del fillin_index, ungapped_indexing

                cupy.get_default_memory_pool().free_all_blocks()
                cupy.get_default_pinned_memory_pool().free_all_blocks()
                mempool.free_all_blocks()

                torch.cuda.empty_cache()    
                torch.cuda.reset_peak_memory_stats(0)
                torch.cuda.memory_allocated(0)
                torch.cuda.max_memory_allocated(0)

            compute_stream.synchronize()

        print("WARNING. Output NNZ %s and it consumes %s GB. Okay?" %(A.count_nonzero(),mempool.total_bytes()/1024/1024/1024))
        #print(mempool.used_bytes()/1024/1024/1024)              # 0
        #print(mempool.total_bytes()/1024/1024/1024)             # 0
        #print(pinned_mempool.n_free_blocks())    # 0

        mempool.free_all_blocks()

        torch.cuda.empty_cache()    
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.memory_allocated(0)
        torch.cuda.max_memory_allocated(0)
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        df = pd.DataFrame(CumulativeStat, columns=['Batch', 'Power', 'LessThanCount'])
        print(df.loc[df['Power'] <= -6].groupby(by='Power').mean())
        sns.relplot(data=df, x='Power', y = 'LessThanCount',kind="line")
        plt.show()
        """
        #print(A.indices, A.indptr, A.indices.shape, A.indptr.shape)
        # NOTE Version 0
        # return A

        # NOTE Version 1
        return cupysparse.csr_matrix(
            (A.data, A.indices, A.indptr), 
            shape = (self.n_atoms * 3, self.n_atoms * 3), 
            dtype=cupy.float64 )




# NOTE This is the gapped version! Correct but Xij consumes and leak mem
@torch.no_grad()
class X_SparseCupyMatrix():
    def __init__(self, X, 
        batch_head = None, 
        maxleafsize = 100, rc_Gamma = 8.0,
        device  = torch.device(0), 
        User_PlusI = 1.0,
        dtype_temp = torch.float64, 
        X_precision = torch.cuda.DoubleTensor,
        NnzMinMaxDict = None,

        ):
        super().__init__()
        
        #InchingLite.util.TorchMakePrecision(Precision = str(dtype_temp))
        #InchingLite.util.TorchEmptyCache()


        self.device = device
        self.dtype_temp = dtype_temp
        self.nan = torch.finfo(dtype_temp).eps
        self.dtype_orig = X.dtype    
        self.n_atoms = X.shape[0]
        self.rc_Gamma = rc_Gamma / 10.0
        self.dof = int(3* self.n_atoms)
        self.User_PlusI = User_PlusI


        # NOTE Now rc_gamma is supposed nm
        #print(self.rc_Gamma)
        X = X.type(X_precision)
        self.X = to_dlpack(X)
        self.X = cupy.from_dlpack(self.X)
        self.X_unsqueezed = cupy.expand_dims(self.X, 1)
        #print(self.X_unsqueezed)



        # =======================
        # Size of batch
        # =======================
        if batch_head is None:
            batch_head = []
            PartitionTree = InchingLite.util.GetPartitionTree(range(self.n_atoms), maxleafsize = maxleafsize)
            FlattenPartitionTree_generator = InchingLite.util.FlattenPartitionTree(PartitionTree)
            batch_head = [0]
            # NOTE THe sorted here is necessary as it promote preallocation fo memory
            for i in sorted(FlattenPartitionTree_generator)[::-1]:
                batch_head.append(batch_head[-1] + i)
            batch_head = torch.LongTensor(batch_head)

            del PartitionTree, FlattenPartitionTree_generator
            gc.collect()
        self.batch_head = batch_head
        self.n_batch_min1 = self.batch_head.shape[0] -1



        if NnzMinMaxDict is None:
            self.LeftRightNnzBound = InchingLite.Fuel.Coordinate.T1.X_KdMinMaxNeighbor(X.detach().cpu().numpy(), 
                                                        rc_Gamma=rc_Gamma, maxleafsize = maxleafsize,
                                                        CollectStat = False, SliceForm = True )
        else:
            self.LeftRightNnzBound = NnzMinMaxDict




        # ==========================================
        # Hessian vector multiplication in batch
        # ============================================
        # NOTE as b will be index put rather than any operation we can do the following
        # NOTE Ax=b this is the storage of the product to be returned
        #self.Ax = torch.zeros([self.dof], dtype = self.dtype_temp, device = self.device)







        # ==============================
        # Some put tensor number on GPU
        # =================================
      

        # ================================================
        # Warm up with Gram by catching the nonzero entries
        # ================================================
        # NOTE The catch here is that in very large proteins
        #      Even the warm up will need very fine batches and takes time
        #      But the reward is that after catching these we can safely ignore the zero regions and also gaining precision due to removal of 0*0+++

        PART0_WARMUP = False
        if PART0_WARMUP:
            fine_batch_head = []
            PartitionTree = InchingLite.util.GetPartitionTree(range(self.n_atoms), maxleafsize = 50)
            FlattenPartitionTree_generator = InchingLite.util.FlattenPartitionTree(PartitionTree)
            fine_batch_head = [0]
            for i in sorted(FlattenPartitionTree_generator)[::-1]:
                    fine_batch_head.append(fine_batch_head[-1] + i)
            fine_batch_head = torch.LongTensor(fine_batch_head)
            fine_n_batch_min1 = fine_batch_head.shape[0] -1

            Fine_LeftRightNnzBound = {}
            g_1 = torch.sum(self.X * self.X, axis =1)
            for i in range(fine_n_batch_min1):

                R = g_1.repeat(fine_batch_head[i+1]-fine_batch_head[i], 1).T + \
                    g_1[fine_batch_head[i]:fine_batch_head[i+1]].repeat(self.n_atoms,1) - \
                    2* torch.einsum('bi,ai->ba', (self.X,self.X[fine_batch_head[i]:fine_batch_head[i+1],:]))
                
                R[R > self.rc_Gamma**2] = 0.0

                # NOTE Get the left write bound of this strip
                lrbound = torch.nonzero(R.sum(axis = 1), as_tuple= False)
                lbound = lrbound.min().item()
                rbound = lrbound.max().item() + 1                   # NOTE as we will use slice
                
                #print(fine_batch_head[i],fine_batch_head[i+1], lbound, rbound)
                Fine_LeftRightNnzBound[(fine_batch_head[i],fine_batch_head[i+1])] = (lbound,rbound)


            keys_Fine_LeftRightNnzBound = sorted(Fine_LeftRightNnzBound.keys())
            #print(keys_Fine_LeftRightNnzBound)
            self.LeftRightNnzBound = {}
            for i in range(self.n_batch_min1):

                applicable_lbound = []
                applicable_rbound = []

                for k in keys_Fine_LeftRightNnzBound:
                    if i == 0:
                        v = Fine_LeftRightNnzBound[keys_Fine_LeftRightNnzBound[0]]
                        applicable_lbound.append(v[0])
                        applicable_rbound.append(v[1])

                    if i == self.n_batch_min1 - 1:
                        v = Fine_LeftRightNnzBound[keys_Fine_LeftRightNnzBound[-1]]
                        applicable_lbound.append(v[0])
                        applicable_rbound.append(v[1])  

                    if ((self.batch_head[i] <= k[0]+50) & (self.batch_head[i+1] >= k[1]-50)):
                        v = Fine_LeftRightNnzBound[k]
                        applicable_lbound.append(v[0])
                        applicable_rbound.append(v[1])  

                minlbound = min(applicable_lbound)
                maxrbound = max(applicable_rbound)

                self.LeftRightNnzBound[i] = (minlbound, maxrbound)

            del g_1, R ,lrbound,fine_batch_head
            gc.collect()
            InchingLite.util.TorchEmptyCache()




        # =======================================
        # Make some range vectors before hand
        # =========================================
        self.temp_index_ii = {} # called by size of einsum_rows
        #self.temp_index_ii3 = {}
        self.temp_index_jj = {} # Called by batch index
        #self.temp_index_kk = {} # Called by batch index
        #self.temp_b = {}
        for i in range(self.n_batch_min1):
            # NOTE This will need to be left right bounded
            self.temp_index_jj[i] = np.arange(self.batch_head[i], self.batch_head[i+1], dtype= np.int64) - self.LeftRightNnzBound[i][0] 
            #self.temp_index_kk[i] = np.arange(self.batch_head[i]*3,self.batch_head[i+1]*3, dtype= np.int64) 

            # NOTE Unchanged
            n_einsum_rows = self.temp_index_jj[i].shape[0]
            if n_einsum_rows not in self.temp_index_ii.keys():
                self.temp_index_ii[n_einsum_rows] = np.arange(n_einsum_rows, dtype= np.int64) 
                #self.temp_index_ii3[n_einsum_rows] = torch.arange(n_einsum_rows*3, dtype= torch.long, device= device)
                #self.temp_b[n_einsum_rows] = torch.zeros(
                #                            n_einsum_rows*3, 
                #                            device= device, dtype=dtype_temp)
            #print(self.temp_index_kk[i],self.LeftRightNnzBound[i][0] )
        #sys.exit()

    def ReturnNumberTotalBatch(self):
        return self.n_batch_min1 + 1

    def ReturnCupyH(self): # NOTE This is ARCHIVED
        """
        if help:
            This is a on-demand memory Hessian Matrix-vector product.
            The coeff gamma/distance is also synthesised on demand. 
            ultimately reducing the product memery footprint from O(n_atom ^2 ) to O(n_atom , leaf size)
            Hq = b
            q is a flat vector of size (3 n_atoms)
            b w/ the same shape is the product
        """
        
        #return
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        A = cupysparse.csr_matrix((self.n_atoms * 3, self.n_atoms * 3), dtype=cupy.float64) #cupysparse.eye(self.n_atoms * 3, dtype=np.float64, format='csc') # NOTE 32 easily produce nan!
        #return
        CumulativeStat = []
        compute_stream = cupy.cuda.stream.Stream(non_blocking=False)
        with compute_stream:
            for i in tqdm.tqdm(range(self.n_batch_min1)):

                    # ==============================================
                    # Differences 
                    # ==============================================
                    # Batching 
                    # NOTE While this is also pseudo linear bound considering the zeroing by coeff, 
                    #      it has a O(9bE[N]) with notorious coeff 9! unrealistic to store a (9*1000)*N_atom* 4 byte matrix...
                    # NOTE This is a broadcasted tensor
                    #      (m,n,3)    =    (n,3) - (m,1,3) 
                    #      I will denote the inter-point index as i and j 
                    #                    the inter-point generalised coordinate as pq
                    # NOTE Clearly the trace of each (i,j) block i.e. p==q gives the distance!
                    #      torch.diagonal(x, offset=0, dim1=0, dim2=1)
                    #print(self.rc_Gamma**2)


                    # TODO PDB format
                    Xij_batch = self.X[self.LeftRightNnzBound[i][0]:self.LeftRightNnzBound[i][1], :] - self.X_unsqueezed[self.batch_head[i]:self.batch_head[i+1], :,:]

                    # NOTE PDB format has 3 decimal digits
                    #      These are the fill-ins that will persist! 
                    #      (i,j,p) ~ 0.2 percent at this stage, but will propagate into the einsum!
                    fillin_index = cupy.where(cupy.abs(Xij_batch) < 1e-4)
                    einsum = cupy.einsum('ijp,ijq->ijpq', Xij_batch,Xij_batch)


                    
                    # ==============================
                    # Gamma/distance
                    # ==============================
                    n_einsum_rows = self.temp_index_jj[i].shape[0]

                    # NOTE Distance This is also an torch.einsum('ijkk->ij', einsum), but slower
                    coeff = cupy.sum(cupy.diagonal(einsum, offset=0, axis1=2, axis2=3),axis=2)
                    gamma_mask = cupy.greater(coeff, self.rc_Gamma**2)
                    n_einsum_cols = gamma_mask.shape[1]
                    gamma_mask = cupy.logical_or(gamma_mask, cupy.equal(coeff,0))
                    
                    coeff = cupy.reciprocal(coeff) * -1
                    cupy.putmask(coeff, gamma_mask, 0)
                    coeff = cupy.expand_dims(coeff, 2)
                    coeff = cupy.expand_dims(coeff, 2)
                    
                    # ================================
                    # Broadcast
                    # ================================
                    # Broadcast constant and zero.
                    einsum *= coeff

                    # NOTE Remove Fill-ins just in case
                    einsum[fillin_index[0],fillin_index[1],fillin_index[2],:] = 0
                    einsum[fillin_index[0],fillin_index[1],:,fillin_index[2]] = 0
                    
                    # NOTE cupy 11 put does not work when the to be put is a matrix. 
                    #      i.e. putting matrix to tensor.
                    row_sum = (-1* cupy.sum(einsum,axis = 1))
                    #print(row_sum[0:2])
                    """
                    for i_row in range(einsum.shape[0]):
                        einsum[
                            self.temp_index_ii[self.temp_index_jj[i].shape[0]][i_row],
                            self.temp_index_jj[i][i_row], 
                            0:3,0:3] = row_sum[i_row]
                        if self.temp_index_ii[self.temp_index_jj[i].shape[0]][i_row] == 62571:
                            print("LOOK", row_sum[i_row])
                            sys.exit()
                    """



                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,:,:] = row_sum
                    #if self.batch_head[i]*3 > 2000:
                    #    print(self.temp_index_ii[n_einsum_rows])
                    #    sys.exit()


                    #if self.batch_head[i]*3 > 60000:
                    #    print(einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i],:,:])
                    #    time.sleep(1)

                    # NOTE The A + I condition number trick
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,0,0] += self.User_PlusI
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,1,1] += self.User_PlusI
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,2,2] += self.User_PlusI
                    


                    # =========================
                    # Multiplicatino in batch
                    # =========================
                    einsum = cupy.ascontiguousarray(einsum)
                    einsum = cupy.transpose(einsum, axes=(0,2,1,3))
                    #einsum = cupy.moveaxis(einsum, (0,1,2,3), (0,2,1,3))
                    einsum = cupy.ascontiguousarray(einsum)
                    einsum_dim0 = einsum.shape[0]
                    einsum_dim1 = einsum.shape[1]
                    einsum_dim2 = einsum.shape[2]
                    einsum_dim3 = einsum.shape[3]

                    # NOTE reshape is unsafe??

                    einsum = cupy.reshape(einsum, (einsum_dim0,einsum_dim1, einsum_dim2*einsum_dim3), order='C')
                    einsum = cupy.reshape(einsum, (einsum_dim0 * einsum_dim1, einsum_dim2*einsum_dim3), order='C')
                    #if self.batch_head[i]*3 > 60000:
                    #    print(einsum[:10,:10])
                    #batchtotalnnz = cupy.sum((cupy.abs(einsum) > 0) )

                    
                    """
                    print('min at segment %s > 1e-6 %s out of %s nnz'%(
                        cupy.min(cupy.abs(einsum)[cupy.abs(einsum) > 0]),
                        cupy.sum((cupy.abs(einsum) > 0) & (cupy.abs(einsum) < 1e-6)),
                        cupy.sum((cupy.abs(einsum) > 0) )
                        ))
                    """
                    """
                    print('min at segment %s > 1e-7 %s out of %s nnz'%(
                        cupy.min(cupy.abs(einsum)[cupy.abs(einsum) > 0]),
                        cupy.sum((cupy.abs(einsum) > 0) & (cupy.abs(einsum) < 1e-7)),
                        cupy.sum((cupy.abs(einsum) > 0) )
                        ))
                    for i_power in [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4]:
                        CumulativeStat.append(["batch%s" %(i), 
                        float(i_power),
                        float(cupy.sum((cupy.abs(einsum) > 0) & (cupy.abs(einsum) < 10**i_power)) / batchtotalnnz),
                        ])
                    """
                    #print(cupy.max(cupy.abs(einsum)[cupy.abs(einsum) > 0]))
                    # TODO Assume pdb format 3 digit decimal (x_i - x_j) (y_i -y_j) / Rij^2
                    #      Any number below 1e-3*1e-3/8^2 = 1.5 * 1e-8 are fill-ins.
                    #      but I will defer this removal 
                    """
                    cupy.around(einsum, decimals=7, out=einsum)
                    einsum[cupy.abs(einsum) < 1e-7] = 0
                    #print(cupy.max(cupy.abs(einsum)[cupy.abs(einsum) > 0]), cupy.min(cupy.abs(einsum)[cupy.abs(einsum) > 0]))
                    """
                    einsum = cupy.nan_to_num(einsum, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
                    einsum = cupysparse.coo_matrix(einsum)
                    einsum.eliminate_zeros()

                    #compress = cupy.cusparse.csr2csr_compress(einsum, tol = 1e-7)
                    #einsum.data = compress.data
                    #einsum.indices = compress.indices
                    #einsum.indptr = compress.indptr

                    
                    # NOTE ISSUE https://github.com/cupy/cupy/issues/3223 
                    compute_stream.synchronize()
                    A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                    self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ] = einsum
                    
                    PARTZZZ_CheckCorrect = False
                    if PARTZZZ_CheckCorrect:
                        """
                        print( 'einsum4 dims, batch index', einsum_dim0, einsum_dim1, einsum_dim2, einsum_dim3, i)
                        print('A.shape >? bbbatch gead [i] *3, [i+1]*3' , A.shape, self.batch_head[i]*3, self.batch_head[i+1]*3)
                        print('A.shape >? leftright nnz bound', self.LeftRightNnzBound[i][0]*3,self.LeftRightNnzBound[i][1]*3)
                        """
                        evidence = ~(cupy.allclose( A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                    self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray(), einsum.toarray(), rtol=1e-05, atol=1e-08, equal_nan=False))
                        if evidence:
                            """
                            print('EEEEEEEEevidenccce %s' %(i), cupy.allclose( A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray(), einsum.toarray(), rtol=1e-05, atol=1e-08, equal_nan=False))
                            print(cupy.where(cupy.abs(A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray() - einsum.toarray()) > 1e-8), cupy.where(cupy.abs(A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray() - einsum.toarray()) > 1e-8)[0].shape)
                            print(self.batch_head[i]*3)
                            """
                            xbound = cupy.where(cupy.abs(A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray() - einsum.toarray()) > 1e-8)[1]
                            
                            print('EEEEEEEEevidenccce %s' %(i), cupy.allclose( A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray(), einsum.toarray(), rtol=1e-05, atol=1e-08, equal_nan=False))
                            plotarray = cupy.asnumpy(cupy.abs(A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray() - einsum.toarray()) > 1e-8)
                            import matplotlib.pyplot as plt
                            plt.figure(figsize = (30,30))
                            plt.imshow(plotarray,  vmax=None, vmin=-1e-18, aspect='equal')
                            plt.xlim((xbound.min(), xbound.max()))
                            plt.show()
                        """
                        while evidence:
                            A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                        self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                            ] = einsum
                            print()
                            evidence = ~(cupy.allclose( A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                    self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ].toarray(), einsum.toarray(), rtol=1e-05, atol=1e-08, equal_nan=False))
                            print(evidence)
                        """
                    # ==========================
                    # Memory cleansing
                    # ============================
                    coeff = None
                    gamma_mask = None
                    einsum = None
                    row_sum  = None
                    Xij_batch = None
                    fillin_index = None
                    compress = None
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
            compute_stream.synchronize()
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        print("WARNING. Output NNZ %s and it consumes %s GB. Okay?" %(A.count_nonzero(),mempool.total_bytes()/1024/1024/1024))
        #print(mempool.used_bytes()/1024/1024/1024)              # 0
        #print(mempool.total_bytes()/1024/1024/1024)             # 0
        #print(pinned_mempool.n_free_blocks())    # 0
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        df = pd.DataFrame(CumulativeStat, columns=['Batch', 'Power', 'LessThanCount'])
        print(df.loc[df['Power'] <= -6].groupby(by='Power').mean())
        sns.relplot(data=df, x='Power', y = 'LessThanCount',kind="line")
        plt.show()
        """
        return A





    def ReturnCupyHLowerTriangle(self):
        """
        if help:
            # NOTE This will make the LowerTriangle (including the main diagonal)
            The coeff gamma/distance is also synthesised on demand. 
            ultimately reducing the product memery footprint from O(n_atom ^2 ) to O(n_atom , leaf size)
            Hq = b
            q is a flat vector of size (3 n_atoms)
            b w/ the same shape is the product
        """
        
        #return
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        A = cupysparse.csr_matrix((self.n_atoms * 3, self.n_atoms * 3), dtype=cupy.float64) #cupysparse.eye(self.n_atoms * 3, dtype=np.float64, format='csc') # NOTE 32 easily produce nan!
        #return
        CumulativeStat = []
        compute_stream = cupy.cuda.stream.Stream(non_blocking=False)
        with compute_stream:
            for i in tqdm.tqdm(range(self.n_batch_min1)[:]):

                    # ==============================================
                    # Differences 
                    # ==============================================
                    # Batching 
                    # NOTE While this is also pseudo linear bound considering the zeroing by coeff, 
                    #      it has a O(9bE[N]) with notorious coeff 9! unrealistic to store a (9*1000)*N_atom* 4 byte matrix...
                    # NOTE This is a broadcasted tensor
                    #      (m,n,3)    =    (n,3) - (m,1,3) 
                    #      I will denote the inter-point index as i and j 
                    #                    the inter-point generalised coordinate as pq
                    # NOTE Clearly the trace of each (i,j) block i.e. p==q gives the distance!
                    #      torch.diagonal(x, offset=0, dim1=0, dim2=1)
                    #print(self.rc_Gamma**2)

                    # NOTE Many of these will be zeroed.
                    Xij_batch = self.X[self.LeftRightNnzBound[i][0]:self.LeftRightNnzBound[i][1], :] - self.X_unsqueezed[self.batch_head[i]:self.batch_head[i+1], :,:]

                    # NOTE PDB format has 3 decimal digits
                    #      These are the fill-ins that will persist! 
                    #      (i,j,p) ~ 0.2 percent at this stage, but will propagate into the einsum!
                    fillin_index = cupy.where(cupy.abs(Xij_batch) < 1e-4)
                    einsum = cupy.einsum('ijp,ijq->ijpq', Xij_batch,Xij_batch)


                    
                    # ==============================
                    # Gamma/distance
                    # ==============================
                    n_einsum_rows = self.temp_index_jj[i].shape[0]

                    # NOTE Distance This is also an torch.einsum('ijkk->ij', einsum), but slower
                    coeff = cupy.sum(cupy.diagonal(einsum, offset=0, axis1=2, axis2=3),axis=2)
                    gamma_mask = cupy.greater(coeff, self.rc_Gamma**2)
                    n_einsum_cols = gamma_mask.shape[1]
                    gamma_mask = cupy.logical_or(gamma_mask, cupy.equal(coeff,0))
                    
                    coeff = cupy.reciprocal(coeff) * -1
                    cupy.putmask(coeff, gamma_mask, 0)
                    coeff = cupy.expand_dims(coeff, 2)
                    coeff = cupy.expand_dims(coeff, 2)
                    
                    # ================================
                    # Broadcast
                    # ================================
                    # Broadcast constant and zero.
                    einsum *= coeff

                    # NOTE Remove Fill-ins just in case
                    einsum[fillin_index[0],fillin_index[1],fillin_index[2],:] = 0
                    einsum[fillin_index[0],fillin_index[1],:,fillin_index[2]] = 0
                    
                    # NOTE cupy 11 put does not work when the to be put is a matrix. 
                    #      i.e. putting matrix to tensor.
                    row_sum = (-1* cupy.sum(einsum,axis = 1))
                   



                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,:,:] = row_sum

                    # NOTE The A + I condition number trick
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,0,0] += self.User_PlusI
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,1,1] += self.User_PlusI
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,2,2] += self.User_PlusI
                    


                    # =========================
                    # Multiplicatino in batch
                    # =========================
                    #einsum = cupy.ascontiguousarray(einsum)
                    einsum = cupy.transpose(einsum, axes=(0,2,1,3))
                    #einsum = cupy.ascontiguousarray(einsum)
                    einsum_dim0 = einsum.shape[0]
                    einsum_dim1 = einsum.shape[1]
                    einsum_dim2 = einsum.shape[2]
                    einsum_dim3 = einsum.shape[3]

                    # NOTE reshape is unsafe??
                    einsum = cupy.reshape(einsum, (einsum_dim0,einsum_dim1, einsum_dim2*einsum_dim3), order='C')
                    einsum = cupy.reshape(einsum, (einsum_dim0 * einsum_dim1, einsum_dim2*einsum_dim3), order='C')

                    # TODO Assume pdb format 3 digit decimal (x_i - x_j) (y_i -y_j) / Rij^2
                    #      Any number below 1e-3*1e-3/8^2 = 1.5 * 1e-8 are fill-ins.
                    #      but I will defer this removal 
                    
                    einsum = cupy.nan_to_num(einsum, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
                    einsum_ = cupysparse.coo_matrix(einsum)
                    einsum = None
                    einsum_.eliminate_zeros()

                    #compress = cupy.cusparse.csr2csr_compress(einsum, tol = 1e-7)
                    #einsum.data = compress.data
                    #einsum.indices = compress.indices
                    #einsum.indptr = compress.indptr
                    #print(((self.batch_head[i]*3) - self.LeftRightNnzBound[i][0]*3).item())
                    einsum__ = cupysparse.tril(einsum_, 
                                                k = ((self.batch_head[i]*3) - self.LeftRightNnzBound[i][0]*3).item(),
                                                format = 'coo')

                    einsum_ = None
                    # NOTE The upper triu can be removed in coo
                    # NOTE ISSUE https://github.com/cupy/cupy/issues/3223 
                    compute_stream.synchronize()
                    A[      self.batch_head[i]*3:self.batch_head[i+1]*3,
                            self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ] = einsum__
                    
                    # ==========================
                    # Memory cleansing
                    # ============================
                    coeff = None
                    gamma_mask = None
                    einsum = None
                    einsum_ = None
                    row_sum  = None
                    Xij_batch = None
                    fillin_index = None
                    compress = None
                    del coeff, gamma_mask, einsum, einsum_, row_sum, Xij_batch, fillin_index

                    cupy.get_default_memory_pool().free_all_blocks()
                    cupy.get_default_pinned_memory_pool().free_all_blocks()
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
            compute_stream.synchronize()
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        print("WARNING. Output NNZ %s and it consumes %s GB. Okay?" %(A.count_nonzero(),mempool.total_bytes()/1024/1024/1024))
        #print(mempool.used_bytes()/1024/1024/1024)              # 0
        #print(mempool.total_bytes()/1024/1024/1024)             # 0
        #print(pinned_mempool.n_free_blocks())    # 0
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        df = pd.DataFrame(CumulativeStat, columns=['Batch', 'Power', 'LessThanCount'])
        print(df.loc[df['Power'] <= -6].groupby(by='Power').mean())
        sns.relplot(data=df, x='Power', y = 'LessThanCount',kind="line")
        plt.show()
        """
        return A







# NOTE  OBSOLETE. Too slow
@torch.no_grad()
class OBSOLETE_X_SparseCupyMatrixSegment():
    def __init__(self, X, 
        batch_head = None, 
        maxleafsize = 100, rc_Gamma = 8.0,
        #device  = torch.device(0), 
        User_PlusI = 1.0,
        dtype_temp = cupy.float64, 
        X_precision = torch.cuda.DoubleTensor,
        NnzMinMaxDict = None,

        ):
        super().__init__()
        
        #InchingLite.util.TorchMakePrecision(Precision = str(dtype_temp))
        #InchingLite.util.TorchEmptyCache()


        #self.device = device
        self.dtype_temp = dtype_temp
        self.nan = cupy.finfo(dtype_temp).eps
        self.dtype_orig = X.dtype    
        self.n_atoms = X.shape[0]
        self.rc_Gamma = rc_Gamma / 10.0
        self.dof = int(3* self.n_atoms)
        self.User_PlusI = User_PlusI


        # NOTE Now rc_gamma is supposed nm
        #print(self.rc_Gamma)
        X = X.type(X_precision)
        self.X = to_dlpack(X)
        self.X = cupy.from_dlpack(self.X)
        self.X_unsqueezed = cupy.expand_dims(self.X, 1)
        #print(self.X_unsqueezed)



        # =======================
        # Size of batch
        # =======================
        if batch_head is None:
            batch_head = []
            PartitionTree = InchingLite.util.GetPartitionTree(range(self.n_atoms), maxleafsize = maxleafsize)
            FlattenPartitionTree_generator = InchingLite.util.FlattenPartitionTree(PartitionTree)
            batch_head = [0]
            # NOTE THe sorted here is necessary as it promote preallocation fo memory
            for i in sorted(FlattenPartitionTree_generator)[::-1]:
                batch_head.append(batch_head[-1] + i)
            batch_head = torch.LongTensor(batch_head)
            del PartitionTree, FlattenPartitionTree_generator
            gc.collect()
        self.batch_head = batch_head
        self.n_batch_min1 = self.batch_head.shape[0] -1



        if NnzMinMaxDict is None:
            self.LeftRightNnzBound = InchingLite.Fuel.Coordinate.T1.X_KdMinMaxNeighbor(X.detach().cpu().numpy(), 
                                                        rc_Gamma=rc_Gamma, maxleafsize = maxleafsize,
                                                        CollectStat = False, SliceForm = True )
        else:
            self.LeftRightNnzBound = NnzMinMaxDict

        
        # =======================================
        # Make some range vectors before hand
        # =========================================
        self.temp_index_ii = {} # called by size of einsum_rows
        #self.temp_index_ii3 = {}
        self.temp_index_jj = {} # Called by batch index
        self.temp_index_kk = {} # Called by batch index
        #self.temp_b = {}
        for i in range(self.n_batch_min1):
            # NOTE This will need to be left right bounded
            self.temp_index_jj[i] = np.arange(self.batch_head[i], self.batch_head[i+1], dtype= np.int64) - self.LeftRightNnzBound[i][0] 
            self.temp_index_kk[i] = np.arange(self.batch_head[i]*3,self.batch_head[i+1]*3, dtype= np.int64) 

            # NOTE Unchanged
            n_einsum_rows = self.temp_index_jj[i].shape[0]
            if n_einsum_rows not in self.temp_index_ii.keys():
                self.temp_index_ii[n_einsum_rows] = np.arange(n_einsum_rows, dtype= np.int64) 
                #self.temp_index_ii3[n_einsum_rows] = torch.arange(n_einsum_rows*3, dtype= torch.long, device= device)
                #self.temp_b[n_einsum_rows] = torch.zeros(
                #                            n_einsum_rows*3, 
                #                            device= device, dtype=dtype_temp)
            #print(self.temp_index_kk[i],self.LeftRightNnzBound[i][0] )
        #sys.exit()

    def ReturnCupyH(self,         
                    User_StartAtBatchI = None,
                    User_StopAtBatchI = None,

                    ):
        """
        if help:
            This is a on-demand memory Hessian Matrix-vector product.
            The coeff gamma/distance is also synthesised on demand. 
            ultimately reducing the product memery footprint from O(n_atom ^2 ) to O(n_atom , leaf size)
            Hq = b
            q is a flat vector of size (3 n_atoms)
            b w/ the same shape is the product
        """
        # ============================
        # Control Flow
        # ==============================

        if User_StopAtBatchI is None:
            self.User_StopAtBatchI = self.n_batch_min1 + 777
        else:
            self.User_StopAtBatchI = User_StopAtBatchI

        if User_StartAtBatchI is None:
            self.User_StartAtBatchI = 0
        else:
            self.User_StartAtBatchI = User_StartAtBatchI


        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        A = cupysparse.csr_matrix((self.n_atoms * 3, self.n_atoms * 3), dtype=cupy.float64) #cupysparse.eye(self.n_atoms * 3, dtype=np.float64, format='csc') # NOTE 32 easily produce nan!

        CumulativeStat = []
        compute_stream = cupy.cuda.stream.Stream(non_blocking=False)
        with compute_stream:
            for i in range(self.n_batch_min1):


                    # =====================
                    # Control flow
                    # ======================
                    if i < self.User_StartAtBatchI:
                        continue
                    if i > self.User_StopAtBatchI:
                        continue

                    # ==============================================
                    # Differences 
                    # ==============================================
                    # Batching 
                    # NOTE While this is also pseudo linear bound considering the zeroing by coeff, 
                    #      it has a O(9bE[N]) with notorious coeff 9! unrealistic to store a (9*1000)*N_atom* 4 byte matrix...
                    # NOTE This is a broadcasted tensor
                    #      (m,n,3)    =    (n,3) - (m,1,3) 
                    #      I will denote the inter-point index as i and j 
                    #                    the inter-point generalised coordinate as pq
                    # NOTE Clearly the trace of each (i,j) block i.e. p==q gives the distance!
                    #      torch.diagonal(x, offset=0, dim1=0, dim2=1)
                    #print(self.rc_Gamma**2)


                    # TODO PDB format
                    Xij_batch = self.X[self.LeftRightNnzBound[i][0]:self.LeftRightNnzBound[i][1], :] - self.X_unsqueezed[self.batch_head[i]:self.batch_head[i+1], :,:]

                    # NOTE PDB format has 3 decimal digits
                    #      These are the fill-ins that will persist! 
                    #      (i,j,p) ~ 0.2 percent at this stage, but will propagate into the einsum!
                    fillin_index = cupy.where(cupy.abs(Xij_batch) < 1e-4)
                    einsum = cupy.einsum('ijp,ijq->ijpq', Xij_batch,Xij_batch)


                    
                    # ==============================
                    # Gamma/distance
                    # ==============================
                    n_einsum_rows = self.temp_index_jj[i].shape[0]

                    # NOTE Distance This is also an torch.einsum('ijkk->ij', einsum), but slower
                    coeff = cupy.sum(cupy.diagonal(einsum, offset=0, axis1=2, axis2=3),axis=2)
                    gamma_mask = cupy.greater(coeff, self.rc_Gamma**2)
                    n_einsum_cols = gamma_mask.shape[1]
                    gamma_mask = cupy.logical_or(gamma_mask, cupy.equal(coeff,0))
                    
                    coeff = cupy.reciprocal(coeff) * -1
                    cupy.putmask(coeff, gamma_mask, 0)
                    coeff = cupy.expand_dims(coeff, 2)
                    coeff = cupy.expand_dims(coeff, 2)
                    
                    # ================================
                    # Broadcast
                    # ================================
                    # Broadcast constant and zero.
                    einsum *= coeff

                    # NOTE Remove Fill-ins just in case
                    einsum[fillin_index[0],fillin_index[1],fillin_index[2],:] = 0
                    einsum[fillin_index[0],fillin_index[1],:,fillin_index[2]] = 0
                    
                    # NOTE cupy 11 put does not work when the to be put is a matrix. 
                    #      i.e. putting matrix to tensor.
                    row_sum = (-1* cupy.sum(einsum,axis = 1))
                    #print(row_sum[0:2])
                    """
                    for i_row in range(einsum.shape[0]):
                        einsum[
                            self.temp_index_ii[self.temp_index_jj[i].shape[0]][i_row],
                            self.temp_index_jj[i][i_row], 
                            0:3,0:3] = row_sum[i_row]
                        if self.temp_index_ii[self.temp_index_jj[i].shape[0]][i_row] == 62571:
                            print("LOOK", row_sum[i_row])
                            sys.exit()
                    """



                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,:,:] = row_sum
                    #if self.batch_head[i]*3 > 2000:
                    #    print(self.temp_index_ii[n_einsum_rows])
                    #    sys.exit()


                    #if self.batch_head[i]*3 > 60000:
                    #    print(einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i],:,:])
                    #    time.sleep(1)

                    # NOTE The A + I condition number trick
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,0,0] += self.User_PlusI
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,1,1] += self.User_PlusI
                    einsum[self.temp_index_ii[n_einsum_rows], self.temp_index_jj[i] ,2,2] += self.User_PlusI
                    


                    # =========================
                    # Multiplicatino in batch
                    # =========================
                    einsum = cupy.ascontiguousarray(einsum)
                    einsum = cupy.transpose(einsum, axes=(0,2,1,3))
                    #einsum = cupy.moveaxis(einsum, (0,1,2,3), (0,2,1,3))
                    einsum = cupy.ascontiguousarray(einsum)
                    einsum_dim0 = einsum.shape[0]
                    einsum_dim1 = einsum.shape[1]
                    einsum_dim2 = einsum.shape[2]
                    einsum_dim3 = einsum.shape[3]

                    # NOTE reshape is unsafe??

                    einsum = cupy.reshape(einsum, (einsum_dim0,einsum_dim1, einsum_dim2*einsum_dim3), order='C')
                    einsum = cupy.reshape(einsum, (einsum_dim0 * einsum_dim1, einsum_dim2*einsum_dim3), order='C')
                    #if self.batch_head[i]*3 > 60000:
                    #    print(einsum[:10,:10])
                    batchtotalnnz = cupy.sum((cupy.abs(einsum) > 0) )

                    
                    """
                    print('min at segment %s > 1e-6 %s out of %s nnz'%(
                        cupy.min(cupy.abs(einsum)[cupy.abs(einsum) > 0]),
                        cupy.sum((cupy.abs(einsum) > 0) & (cupy.abs(einsum) < 1e-6)),
                        cupy.sum((cupy.abs(einsum) > 0) )
                        ))
                    """
                    """
                    print('min at segment %s > 1e-7 %s out of %s nnz'%(
                        cupy.min(cupy.abs(einsum)[cupy.abs(einsum) > 0]),
                        cupy.sum((cupy.abs(einsum) > 0) & (cupy.abs(einsum) < 1e-7)),
                        cupy.sum((cupy.abs(einsum) > 0) )
                        ))
                    for i_power in [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4]:
                        CumulativeStat.append(["batch%s" %(i), 
                        float(i_power),
                        float(cupy.sum((cupy.abs(einsum) > 0) & (cupy.abs(einsum) < 10**i_power)) / batchtotalnnz),
                        ])
                    """
                    #print(cupy.max(cupy.abs(einsum)[cupy.abs(einsum) > 0]))
                    # TODO Assume pdb format 3 digit decimal (x_i - x_j) (y_i -y_j) / Rij^2
                    #      Any number below 1e-3*1e-3/8^2 = 1.5 * 1e-8 are fill-ins.
                    #      but I will defer this removal 
                    """
                    cupy.around(einsum, decimals=7, out=einsum)
                    einsum[cupy.abs(einsum) < 1e-7] = 0
                    #print(cupy.max(cupy.abs(einsum)[cupy.abs(einsum) > 0]), cupy.min(cupy.abs(einsum)[cupy.abs(einsum) > 0]))
                    """
                    einsum = cupy.nan_to_num(einsum, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
                    einsum = cupysparse.coo_matrix(einsum)
                    einsum.eliminate_zeros()

                    #compress = cupy.cusparse.csr2csr_compress(einsum, tol = 1e-7)
                    #einsum.data = compress.data
                    #einsum.indices = compress.indices
                    #einsum.indptr = compress.indptr

                    
                    # NOTE ISSUE https://github.com/cupy/cupy/issues/3223 
                    compute_stream.synchronize()
                    A[self.batch_head[i]*3:self.batch_head[i+1]*3,
                    self.LeftRightNnzBound[i][0]*3:self.LeftRightNnzBound[i][1]*3
                                        ] = einsum
                    
                    # ==========================
                    # Memory cleansing
                    # ============================
                    coeff = None
                    gamma_mask = None
                    einsum = None
                    row_sum  = None
                    Xij_batch = None
                    fillin_index = None
                    compress = None
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
            compute_stream.synchronize()
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        #print("WARNING. Output NNZ %s and it consumes %s GB. Okay?" %(A.count_nonzero(),mempool.total_bytes()/1024/1024/1024))
        #print(mempool.used_bytes()/1024/1024/1024)              # 0
        #print(mempool.total_bytes()/1024/1024/1024)             # 0
        #print(pinned_mempool.n_free_blocks())    # 0
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        df = pd.DataFrame(CumulativeStat, columns=['Batch', 'Power', 'LessThanCount'])
        print(df.loc[df['Power'] <= -6].groupby(by='Power').mean())
        sns.relplot(data=df, x='Power', y = 'LessThanCount',kind="line")
        plt.show()
        """
        return A




    def ReturnNumberTotalBatch(self):
        return self.n_batch_min1 + 1

