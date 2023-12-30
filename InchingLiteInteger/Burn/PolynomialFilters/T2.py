import numpy 
import numpy as np
import sys
import tqdm
import cupy as cp
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../InchingLiteInteger/Burn/')

# NOTE These are actually numpy processes
import InchingLiteInteger.Burn.PolynomialFilters.T0



# NOTE This one is on gpu 
class OOC2_A_Adiag_ChebyshevAv:
    
    def __init__(self, A, A_diag, User_PolynomialParams = None, User_HalfMemMode = True):
        assert User_PolynomialParams is not None, "ABORTED. You must initiate with a found Polynomial Param"
        self.User_PolynomialParams = User_PolynomialParams 
        self.temp_dtype = A.dtype
        self.N = A.shape[0]  # Assuming n is the length of v. v is a np array
        if User_HalfMemMode:
            self.KrylovAv = InchingLiteInteger.Burn.Krylov.T3.OOC2_HalfMemS_v_KrylovAv_VOID(A, A_diag)
        else:
            self.KrylovAv = InchingLiteInteger.Burn.Krylov.T3.OOC2_FullMemS_v_KrylovAv_VOID(A, A_diag)

    def ChebyshevAv(self, A, v, 
                    User_ReturnRho = False, 
                    ):




        

        mu = self.User_PolynomialParams.AdjustedCoefficients
        dd = self.User_PolynomialParams.dd
        cc = self.User_PolynomialParams.cc
        m = self.User_PolynomialParams.AdjustedDegree
        
        # NOTE For record of dos
        jac = self.User_PolynomialParams.AdjustedDampingKernels
        rho = self.User_PolynomialParams.theta_a - self.User_PolynomialParams.theta_b

        # NOTE in EVSL, w is assumed to be a pre-allocated array with 3n elements
        #      we are lazier we make 3 columns. 
        vk = cp.zeros(self.N, dtype = self.temp_dtype)
        vkp1 = cp.zeros(self.N, dtype = self.temp_dtype)
        vkm1 = cp.zeros(self.N, dtype = self.temp_dtype)

        # Copy v to vk NOTE numpy.copyto(dst, src, casting='same_kind', where=True)
        #vk = cp.copy(v)
        cp.copyto(vk, v)
        
        y = v * mu[0]

        # Degree loop. k IS the degree
        for k in range(1, m + 1):
            
            scal = 1.0 / dd if k == 1 else 2.0 / dd

            # Matrix-vector multiplication
            #vkp1 = A@vk
            self.KrylovAv(A,vk,vkp1)
            
            # Vector operations # TODO All these should be written in cupy kernel
            vkp1 = scal * (vkp1 - cc * vk) - (vkm1 if k > 1 else 0.0)
            y += mu[k] * vkp1

            #vkm1 = cp.copy(vk)
            #vk = cp.copy(vkp1)
            cp.copyto(vkm1, vk)
            cp.copyto(vk, vkp1)

            # NOTE for kpm rho
            #print(cp.dot(vk, v))
            rho -= 2 * jac[k] * cp.dot(vk, v) * (
                    cp.sin((k) * self.User_PolynomialParams.theta_b) - cp.sin((k) * self.User_PolynomialParams.theta_a)
                    ) / (k)
            
        rho /= np.pi

        vk = None
        vkp1 = None
        vkm1 = None
        del vk, vkm1, vkp1

        if User_ReturnRho:
            return y, rho
        else:
            return y


# NOTE Kpm as a demo of cheb
def A_Adiag_PolynomialParams_KpmEstimate(A, A_diag,
                                         User_HalfMemMode = True,
                                         User_PolynomialParams = None, User_NumberKpmTrials = 1000):

    assert User_PolynomialParams is not None, "ABORTED. You must initiate with a found Polynomial Param"
    N = A.shape[0]


    ChebyshevAvC = OOC2_A_Adiag_ChebyshevAv(A, A_diag, User_PolynomialParams = User_PolynomialParams, User_HalfMemMode = User_HalfMemMode)
    ChebyshevAv = ChebyshevAvC.ChebyshevAv

    # ===================================
    # KPM
    # ====================================
    # NOTE Record of density estimate.
    rho_sample_list = []
    rho = 0.0
    for _ in tqdm.tqdm(range(User_NumberKpmTrials)):

        v = cp.random.randn(N).astype(A.dtype)  
        v /= cp.sqrt((cp.multiply(v,v)).sum(axis=0)) 

        #vv, rho_sample = ChebyshevAv(A, A_diag, v, User_PolynomialParams, User_ReturnRho = True)
        vv, rho_sample = ChebyshevAv(A, v, User_ReturnRho = True)

        #print(v.dot(vv))
        
        vv = None
        del vv
        rho += rho_sample

        rho_sample_list.append(rho_sample.get()) # NOTE we can also calculate variance iteratively but I am too lazy to write here lol
        
    
    # NOTE Now we need to average the experiments and get the coutn
    rho_sample_list = np.array(rho_sample_list) * N
    rho_sample_list = rho_sample_list.astype(np.int64)
    eigval_count_estimate =  rho / User_NumberKpmTrials  * N  
    #print(rho_sample_list)
    print("Kpm Estiamte Mean %s Std %s" %(np.mean(rho_sample_list, dtype= np.float64),np.std(rho_sample_list, dtype= np.float64) ))


    return int(eigval_count_estimate) + 1  # NOTE To play safe let's add one...


# Polynomial Filter T2 
def A_Adiag_OptimizePolynomialParamsOnMemory(  A, A_diag,

                                                User_MaximumDegree = 1000,
                                                User_MinimumDegree = 10,
                                                User_DampingKernel = "Jackson",
                                                User_ExtremalIntervalDefinition = 1e-10,
                                                User_WantedInterval = (36.8, 37.2),
                                                User_SpectrumBound = (1.0, 759.0),
                                                User_DesignatedStart = None,
                                                User_WantedNumberEigenvalue = 12,
                                                User_AffordableMemoryMargin = 5, 
                                                User_HalfMemMode = True,
                                                User_NumberKpmTrials = 1000,
                                                User_ConvergenceRatio = 0.1):



    if User_DesignatedStart is not None:
        print("WARNING. Binary search begins with user supplied User_DesignatedStart")
        #print('gagagagag', User_WantedInterval, User_DesignatedStart, User_SpectrumBound)
        # Check if it is more than needed
        User_PolynomialParams = InchingLiteInteger.Burn.PolynomialFilters.T0.OOC0_OptimizePolynomialParamsOnDegree(
                                User_MaximumDegree = User_MaximumDegree,
                                User_MinimumDegree = User_MinimumDegree,
                                User_DampingKernel = User_DampingKernel,
                                User_ExtremalIntervalDefinition = User_ExtremalIntervalDefinition,

                                User_WantedInterval = (User_WantedInterval[0],User_DesignatedStart),

                                User_SpectrumBound = User_SpectrumBound, 
                                User_ConvergenceRatio = User_ConvergenceRatio
                                )
        
        #print("After T0", np.sum(User_PolynomialParams.AdjustedCoefficients))
        KpmEigvalCount = A_Adiag_PolynomialParams_KpmEstimate(A, A_diag, 
                                                                     User_PolynomialParams = User_PolynomialParams , 
                                                                     User_HalfMemMode = User_HalfMemMode, 
                                                                     User_NumberKpmTrials = User_NumberKpmTrials)
        if KpmEigvalCount > User_WantedNumberEigenvalue*2:
            User_WantedInterval = (User_WantedInterval[0],User_DesignatedStart)
            temp_User_WantedInterval = [User_WantedInterval[0], User_DesignatedStart]
        else:
            print("WARNING. User_DesignatedStart is producing less than User_WantedNumberEigenvalue in KPM. ")
            temp_User_WantedInterval = [User_WantedInterval[0], User_SpectrumBound[1]]
            pass


    else:
        temp_User_WantedInterval = [User_WantedInterval[0], User_SpectrumBound[1]]
    # NOTE This is Essentially a binary search 
    #temp_User_WantedInterval = User_WantedInterval
    #temp_User_WantedInterval = [temp_User_WantedInterval[0], User_SpectrumBound[1]]

    target = User_WantedNumberEigenvalue        # Your target integer value
    tolerance = User_AffordableMemoryMargin     # Tolerance for the closeness to the target

    # Define the initial search interval boundaries
    left = temp_User_WantedInterval[0]
    right = temp_User_WantedInterval[1]
    #right = temp_User_WantedInterval[0]
    # =======================
    # Binary search loop
    # =======================
    while (right - left >= 1e-8) : # NOTE Fail if your eigenvalue gap is smaller than this
        mid = (left + right) / 2  # Calculate the midpoint
        
        # Calculate eigval_count_estimate for the current mid
        temp_User_WantedInterval[1] = mid
        """
        # NOTE If the requested interval User_WantedInterval is much much less than the User_ConvergenceRatio,
        #      It is supposed to give less iterations. However, the cost of doing higher degree of 
        #      cheb might outweigh the advantage. Given that we are familiar with the systems 
        
        if abs(temp_User_WantedInterval[1] - temp_User_WantedInterval[0]) < 1 - User_ConvergenceRatio:
            print("WARNING. Adjusted Convergence Ratio to %s " %((temp_User_WantedInterval[1] - temp_User_WantedInterval[0])))
            User_ConvergenceRatio_ = 1.0 - abs(temp_User_WantedInterval[1] - temp_User_WantedInterval[0]) 
        else:
            User_ConvergenceRatio_ = User_ConvergenceRatio
        """
        User_ConvergenceRatio_ = User_ConvergenceRatio
        User_PolynomialParams = InchingLiteInteger.Burn.PolynomialFilters.T0.OOC0_OptimizePolynomialParamsOnDegree(
                                User_MaximumDegree = User_MaximumDegree,
                                User_MinimumDegree = User_MinimumDegree,
                                User_DampingKernel = User_DampingKernel,
                                User_ExtremalIntervalDefinition = User_ExtremalIntervalDefinition,

                                User_WantedInterval = temp_User_WantedInterval,

                                User_SpectrumBound = User_SpectrumBound, 
                                User_ConvergenceRatio = User_ConvergenceRatio_
                                )
        #print("After T0", np.sum(User_PolynomialParams.AdjustedCoefficients))
        KpmEigvalCount = A_Adiag_PolynomialParams_KpmEstimate(A, A_diag, 
                                                                     User_PolynomialParams = User_PolynomialParams , 
                                                                     User_HalfMemMode = User_HalfMemMode, 
                                                                     User_NumberKpmTrials = User_NumberKpmTrials)

        #print(temp_User_WantedInterval, User_SpectrumBound, User_PolynomialParams.AdjustedDegree, KpmEigvalCount)
        print("Adjusted degree to %s and Interval to %s,%s. Estimate number of eigval is %s" %(
        User_PolynomialParams.AdjustedDegree, temp_User_WantedInterval[0], temp_User_WantedInterval[1], KpmEigvalCount))
        print("Convergence ratio %s" %(User_PolynomialParams.User_ConvergenceRatio))
        # Check if eigval_count_estimate is within tolerance of the target
        if abs(KpmEigvalCount - target) <= tolerance:
            break  # Close enough to the target, exit the loop
        elif KpmEigvalCount < target: 
            
            left = mid   # Adjust the left boundary
        else:
            right = mid   # Adjust the right boundar

    #print("Adjusted degree to %s and Interval to %s,%s. Estimate number of eigval is %s" %(
    #    User_PolynomialParams.AdjustedDegree, temp_User_WantedInterval[0], temp_User_WantedInterval[1], KpmEigvalCount))
    #print("fainale", User_PolynomialParams.AdjustedCoefficients)
    return User_PolynomialParams, KpmEigvalCount, tuple(temp_User_WantedInterval)


