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


import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg

import numpy
import cupy
import cupy as cp

from cupy import cublas
from cupy import cusparse
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse.linalg import _interface
import time

cupy.random.seed(seed = 0)
import tqdm 
import time
import sys

sys.path.append('../InchingLiteInteger/Burn/')

import InchingLiteInteger.Burn.Orthogonalization.T3
import InchingLiteInteger.Burn.Orthogonalization.T2
import InchingLiteInteger.Burn.Krylov.T3
import InchingLiteInteger.Burn.HermitianLanczos.T2

# =================================
# Tools
# ==================================
# NOTE Dense counterparts
def OOC2_A_x_ChebyshevFilterSlim(A,x
                                 ,polym=80,low=0.0,high=759.0):


    e = (high - low) / 2
    c = (high + low) / 2

    y = A@x
    y = (- y + (c*x)) / e
    for _ in range(polym):
        ynew = A@y
        ynew =(((- ynew + (c*y))*2) / e) - x
        #cp.copyto(x,y) # NOTE Do not change the x
        #cp.copyto(y,ynew)
        x = y
        y = ynew
    return y

def OOC_A_EstimateSpectrumBound(A):
    
    N = A.shape[0]
    jmin = min(100, int(N / 2 / 2))
    n_tridiag = max(jmin, int(N / 2))

    alpha_list = cp.zeros(n_tridiag + 1, dtype=A.dtype)
    beta_list = cp.zeros(n_tridiag, dtype=A.dtype)


    # NOTE f is the placeholder for Av product.

    f = cp.zeros(N, dtype=A.dtype)
    v0 = cp.zeros(N, dtype=A.dtype)

    v = cp.random.random(N) - 0.5
    v /= cp.linalg.norm(v, ord = 2)

    f = A @ v   # NOTE KrylovAv(A,v_,f_)
    alpha = cp.dot(v, f)

    f -= alpha * v
    alpha_list[0] = alpha

    for i in range(n_tridiag):
        beta = cp.linalg.norm(f, ord = 2)
        
        #v0 = v
        #v = f / beta
        cp.copyto(v0, v)
        cp.copyto(v, f / beta)

        f = A @ v
        f -= beta * v0

        alpha = cp.dot(f, v)
        f -= alpha * v

        alpha_list[i + 1] += alpha
        beta_list[i] += beta




    #T_  = numpy.diag(cupy.asnumpy(alpha_list_),0) + numpy.diag(cupy.asnumpy(beta_list_),-1) + numpy.diag(cupy.asnumpy(beta_list_),1)
    T = numpy.diag(alpha_list, 0) + numpy.diag(beta_list, -1) + numpy.diag(beta_list, 1)
    w, q = numpy.linalg.eigh(T)
    



    return numpy.min(w), numpy.max(w)

# NOTE we will initiate a vector inside because this is only run once. 
def OOC2_KrylovAv_A_EstimateSpectrumBound(KrylovAv, A):
    
    N = A.shape[0]
    jmin = min(100, int(N / 2 / 2))
    #n_tridiag = max(jmin, int(N / 2))
    n_tridiag = 300 # NOTE 300 enough

    alpha_list = cp.zeros(n_tridiag + 1, dtype=A.dtype)
    beta_list = cp.zeros(n_tridiag, dtype=A.dtype)


    # NOTE f is the placeholder for Av product.
    f = cp.zeros(N, dtype=A.dtype)
    v0 = cp.zeros(N, dtype=A.dtype)

    v = cp.random.random(N) - 0.5
    v /= cp.linalg.norm(v, ord = 2)

    
    KrylovAv(A,v,f)
    #f = A @ v   # NOTE KrylovAv(A,v_,f_)
    alpha = cp.dot(v, f)

    f -= alpha * v
    alpha_list[0] = alpha

    for i in tqdm.tqdm(range(n_tridiag)): # NOTE Classic question here. what if you change n_tridiag? This is a no-restart lanczos
        beta = cp.linalg.norm(f, ord = 2)
        
        #v0 = v
        #v = f / beta
        cp.copyto(v0, v)
        cp.copyto(v, f / beta)

        #f = A @ v
        KrylovAv(A,v,f)
        #f = A @ v
        f -= beta * v0

        alpha = cp.dot(f, v)
        f -= alpha * v

        alpha_list[i + 1] += alpha
        beta_list[i] += beta




    #T_  = numpy.diag(cupy.asnumpy(alpha_list_),0) + numpy.diag(cupy.asnumpy(beta_list_),-1) + numpy.diag(cupy.asnumpy(beta_list_),1)
    T = numpy.diag(alpha_list, 0) + numpy.diag(beta_list, -1) + numpy.diag(beta_list, 1)
    w, q = numpy.linalg.eigh(T)
    
    f = None
    v = None
    v0 = None
    del f,v,v0

    return numpy.min(w), numpy.max(w)


def OOC5_KrylovAv_A_y_ynew_x_ChebyshevFilterSlim(KrylovAv, A, y, ynew, x
                                 ,polym=80,low=0.0,high=759.0,

                                 ):

    z = cp.ravel(cp.copy(x))

    e = (high - low) / 2
    c = (high + low) / 2

    KrylovAv(A,z,y)
    #y = A@x


    y = (- y + (c*z)) / e

    for _ in range(polym):

        KrylovAv(A,cupy.ravel(y),ynew)

        #ynew = A@y
        ynew =(((- ynew + (c*y))*2) / e) - z

        z = cp.copy(y)
        y = cp.copy(ynew)
    z = None
    del z
    return y



# =============================
# Main
# ===================================
def S_HeigvalCDSIHD_HeigvecCDSIHD(A, A_diag,
        
        k=16,
        tol=1e-10,
        maxiter=1000000,

        
        
        User_WorkspaceSizeFactor = 2,
        User_ChebyshevDegree = 80, 

        User_HalfMemMode= True,
        
        #User_IntermediateConvergenceTol=1e-3,
        #User_GapEstimate=0.1,
        #User_FactoringToleranceOnCorrection = 1e-4,
        User_SpectrumBound = None,

        User_Q_HotellingDeflation = None,
        User_HotellingShift = 10.0
        ):
        

    assert User_SpectrumBound is not None, "ABORTED. The spectrum bound has to be initiated."
    assert User_WorkspaceSizeFactor > 1, "ABORTED. We require User_WorkspaceSizeFactor > 1 "
    assert User_Q_HotellingDeflation is None, "ABORTED. We do not support Hotelling at the moment (Actually we do not need it if we have a mid-pass filter.) "
    _dohotelling = False

    N = A.shape[0]
    jmin = min(N, k)
    jmax = User_WorkspaceSizeFactor * jmin

    dtype_temp = A.dtype


    if User_Q_HotellingDeflation is None:
        print("WARNING. Hotelling deflation not in use")
        _dohotelling = False
    else:
        _dohotelling = True

    if N > 2000000*3:
        printing_ = 1
    else:
        printing_ = 100



    

    polym = User_ChebyshevDegree # TODO Give option to determine polynomial degre. We will write another module.



    




    PART00_Initialization = True
    if PART00_Initialization:
        # ===================
        # Memory Management
        # ===================
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()


        eigval_converged = cp.zeros((jmin + 1, 1), dtype = dtype_temp)
        # ===================
        # Ritz
        # =====================
        V = cp.zeros((N, jmax), dtype = dtype_temp) 
        Q = cp.zeros((N, jmax), dtype = dtype_temp)
        G = cp.zeros((jmax, jmax), dtype = dtype_temp)

        # NOTE Placegolder
        Av = cp.empty((N,)).astype(dtype_temp)
        x = cp.empty((N,1)).astype(dtype_temp)


        # NOTE First ritz. Trivially Different from JDM in that the x will be updated
        x = cp.random.rand(N,1).astype(dtype_temp) 
        x = x / cp.sqrt((cp.multiply(x,x)).sum(axis=0))  #cp.linalg.norm(x, ord=2)
        
        # NOTE Placeholder for the vectors inside cheb
        y = cp.empty((N,)).astype(dtype_temp)
        ynew = cp.empty((N,)).astype(dtype_temp)
        # ===========================================
        # NOTE define krylov protocol to be used. 
        # ============================================
        if User_HalfMemMode:
            KrylovAv = InchingLiteInteger.Burn.Krylov.T3.OOC2_HalfMemS_v_KrylovAv_VOID(A, A_diag)
        else:
            KrylovAv = InchingLiteInteger.Burn.Krylov.T3.OOC2_FullMemS_v_KrylovAv_VOID(A, A_diag)






    # =========================
    # Estimate upper and lower boudn
    # ===========================
    # NOTE See the Block Chebyshev Davidson paper 
    #      The bound is adopted from Zhou, Saad, 

    thb = User_SpectrumBound[-1] + 1e-5           
    current_tha = User_SpectrumBound[0] - 1e-5






    # =================
    # Initialize with random vecgors
    # ================================
    KrylovAv(A,cupy.ravel(x),Av)


    V[:,:1] = x
    #print(Av.shape)
    #Av = A @ x 
    Q[:,0] = Av
    G[0,0]  = x.T.dot(Q[:,0]) 

    # NOTE The first lower bound proposed
    squeeze_tha = (thb + G[0,0]) / 2

    # NOTE Determine whether to move the squeeze_tha or not
    
    beta = max(1e-14,abs(G[0,0])) # NOTE Original CDSI from Zhou Saad
    tolr = tol * beta
    """
    beta = max(1e-14,abs(G[0,0]))
    tolr = tol
    """





    # =========================
    # Main loop
    # =========================

    n_RitzConverged=0
    n_RitzConverged1=1


    jcurrent_V = 1

    for i_iter in range(maxiter):

        # ========================
        # Cheb
        # =========================
        # NOTE Copy the lower bound as median of ritz value in the last round
        current_tha =  squeeze_tha
        Av = OOC5_KrylovAv_A_y_ynew_x_ChebyshevFilterSlim(KrylovAv, 
                                                          A, 
                                                          y, 
                                                          ynew, 
                                                          x ,
                                                          polym=polym,
                                                          low=current_tha,
                                                          high=thb, 
                                                          )
        

        # NOTE There is a bug with cp.linalg.norm(Av, ord = 2) making bananas
        Av = Av / cp.sqrt((cp.multiply(Av,Av)).sum(axis=0)) 




        Av = cp.ravel(InchingLiteInteger.Burn.Orthogonalization.T2.OOC2_qnext_Q_ICGSqnext(Av,V[:,cp.arange(1, jcurrent_V+1) -1])) 
        Av = Av / cp.sqrt(cp.sum(Av*Av))
        Av = cp.ravel(InchingLiteInteger.Burn.Orthogonalization.T2.OOC2_qnext_Q_MGSqnext(Av[:,None],V[:,:n_RitzConverged]))
        V[:, jcurrent_V] = Av

        # =================
        # The next 
        # =================
        KrylovAv(A,cupy.ravel(V[:,jcurrent_V]),Av)



        Q[:, jcurrent_V] = cp.ravel(Av) # NOTE touched only once



        # ====================
        # Update G
        # ====================
        # NOTE This is different JDM in that we never look back at those converged!
        #      Only the 'active part' i.e. the non-converged is involved. # NOTE This is a Mv
        result = V[:, n_RitzConverged:jcurrent_V+1].T.dot(Q[:,jcurrent_V])

        # NOTE since the eigh can take lower triangle no need to update this actually
        G[jcurrent_V, n_RitzConverged:jcurrent_V] = (result[:-1]).T
        G[n_RitzConverged:jcurrent_V+1, jcurrent_V] = result




        # ======================
        # Solving the RQ
        # ======================
        # NOTE Solving the smaller RQ matrix to find projection
        #      We can also do the largest if wanted, just turn on the following two lines.
        #      However, we all know that the largest is "almost effortless" in subspace iteration!
        # NOTE cp.linalg.eigh (a TRLM, not IRLM in np) indeed does not converged
        #S,W=cp.linalg.eigh(G[n_RitzConverged:jcurrent_V+1,n_RitzConverged:jcurrent_V+1], UPLO='L')
        G = (G.T + G)/2
        S,W=numpy.linalg.eigh(cp.asnumpy(G[n_RitzConverged:jcurrent_V+1,n_RitzConverged:jcurrent_V+1]), UPLO='L')
        S = cp.array(S, dtype = A.dtype)
        W = cp.array(W, dtype = A.dtype)
        """
        S = S[::-1]
        W = W[:,::-1]
        """

        
        # ========================
        # Restart
        # ========================
        # NOTE Restart if we are exceeding the workspace.
        if (jcurrent_V + 1 >= jmax):
            j_restart = max([n_RitzConverged1, jmin + 5, min(jmin + n_RitzConverged, jmax - 5)])
        else:
            j_restart = jcurrent_V + 1 
            
        n_RitzNotConverged = j_restart - n_RitzConverged1 + 1


        V[:,n_RitzConverged:j_restart] = V[:,n_RitzConverged:jcurrent_V+1].dot(W[:,:n_RitzNotConverged])
        Q[:,n_RitzConverged:j_restart] = Q[:,n_RitzConverged:jcurrent_V+1].dot(W[:,:n_RitzNotConverged])

        # NOTE Update the RQ
        G[n_RitzConverged:j_restart,n_RitzConverged:j_restart] = cp.diag(S[:n_RitzNotConverged])




              
        # NOTE Update tolerance flexibly, 
        #      While this allows the tolerance to be flexible (having higher than assigned tolerance), 
        #      the acuracy of the modes are then harder to control. 
        
        beta1 = max(abs(S))
        if (beta1 > beta):
            beta = beta1
            tolr = tol *beta  # NOTE orginal CDSI from Zhou Saad
            #tolr = min(tol *beta, tol) # NOTE An accuracy controlled CDSI. 
            #print(beta)


        # =======================
        # Next Ritz
        # =======================
        # NOTE Ritz value and vector
        theta=S[0]
        x=V[:,n_RitzConverged]

        # NOTE Residual # NOTE The Av after the rayleighritz
        r=Q[:,n_RitzConverged] - theta * x
        cur_tol = cp.sqrt((cp.multiply(r,r)).sum(axis=0))#cp.linalg.norm(r)

        #print(cur_tol, theta)

        # =======================
        # Check Convergence 
        # =======================
        if i_iter % printing_ == 0:
            print("%s, %s, %s,  %s" %(i_iter, cur_tol, theta, n_RitzConverged))
        
        swap = False
        if (cur_tol < tolr):
        

            # NOTE Record the eigval
            eigval_converged[n_RitzConverged]=theta
            #print(n_RitzConverged, theta)

            n_RitzConverged += 1
            n_RitzConverged1 = n_RitzConverged + 1
            """
            # ==================
            # Swap
            # ===================
            # NOTE We determine here whether the newly converged is accepted
            #      The significance of the ordering of the vectors is that 
            #      the next ritz vector is  given as x=V[:,n_RitzConverged], 
            #      and the filter is constantly being moved. If the order is not non-decreasing
            #      we may at risk of missing some eigpairs that are lower than the currently found.
            #      
            #      In the work of Zhou and Saad, they also allow checks to continue for 3 times after final convergence 
            #      to make sure no swap.
            if n_RitzConverged < 2 :
                pass
            else:
                swap = False



                # NOTE Current i.e. the freshly converged
                # NOTE The copy is important otherwise it is changed when swapped because it's an address!
                vtmp =  cp.copy(V[:,n_RitzConverged-1] )    
                
                for i in range(n_RitzConverged)[::-1]:      
                    
                    if (theta < eigval_converged[i-1]):
                        
                        swap = True
                        print("swap", theta, eigval_converged[i-1] )
                        eigval_converged[i] = eigval_converged[i-1]
                        eigval_converged[i-1] = theta
                        V[:,i]= V[:,i-1]

                        # NOTE Check if we need further swap by looking forward
                        if (i-1 > 0) & (theta < eigval_converged[i-2]):
                            pass

                        else:
                            # NOTE This put the freshly converged to the place where it is non-decreasing
                            V[:,i-1] = vtmp
                            break

                    else: # NOTE if the first one is smaller then we need not do anything
                        break

                if swap:
                    #print("SWAPPED")
                    x = V[:,n_RitzConverged]
                    continue
            """
            # =====================
            # All Done?
            # =====================
            # NOTE use the swap as a final decision summary "button"
            if (n_RitzConverged >= jmin+3 & swap): # NOTE Let's check 3 more times to make sure 
                swap = False


            if (n_RitzConverged >= jmin & (~swap) )| (n_RitzConverged >= jmin+3): # NOTE This check 3 more times if there are swap needed 
                print("DONE. We went through %s coarse iter, %s eigval converged" %(i_iter, n_RitzConverged))
                eigval_converged = cp.ravel(eigval_converged[:-1])
                idx = cp.argsort(eigval_converged)


                S = None
                Q = None
                W = None
                Av = None
                x = None
                del S,Q,W, Av, x
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()                
                return eigval_converged[idx], V[:,idx] 

            else:
                # NOTE if not swapped
                theta = S[1]
                x = V[:,n_RitzConverged - 1]




        # ========================
        # Update the lower boudn if converged
        # =========================
        jcurrent_V = j_restart
        #print(squeeze_tha)
        #if cp.median(S) < 1.0:
        #   print("why")
        squeeze_tha = cp.max (cp.array([cp.median(S).get(), User_SpectrumBound[0]]))
        #print(squeeze_tha)

    S = None
    Q = None
    W = None
    Av = None
    x = None
    del S,Q,W, Av, x
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()   


    print("ABORTED. It did not actually converged! We went through %s coarse iter and collected %s converged" %(i_iter, n_RitzConverged))
    idx = cp.argsort(eigval_converged)
    return eigval_converged[idx], V[:,idx]


