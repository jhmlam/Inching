import numpy


class PolParams:
    def __init__(self, User_MaximumDegree=10000, User_MinimumDegree=2, 
                 User_DampingKernel = "Jackson", 
                 mu=None, deg=0, 
                 User_ExtremalIntervalDefinition=1e-9, 
                 User_ConvergenceRatio = 0.1,
                 ):
        self.User_MaximumDegree = User_MaximumDegree
        self.User_MinimumDegree = User_MinimumDegree
        self.User_DampingKernel = User_DampingKernel
        self.User_ExtremalIntervalDefinition = User_ExtremalIntervalDefinition
        self.User_ConvergenceRatio = User_ConvergenceRatio 
        # 
        self.AdjustedCoefficients = mu
        self.AdjustedDampingKernels = None
        self.AdjustedDegree = deg




# NOTE On Cpu
def OOC1_DegreeM_DampingKernel(
            m, 
            User_MaximumDegree = 1000,
            User_DampingKernel = "Jackson" ,
            dtype_temp = numpy.float64):
    

    jac = numpy.zeros(User_MaximumDegree+1, 
                   dtype = dtype_temp) # TODO Make this preassigned.
    
    k_values = numpy.arange(1, m + 1,dtype = numpy.float64)

    if User_DampingKernel == "Jackson":
        thetJ = numpy.pi / (m + 2)
        a1 = 1 / (m + 2)
        a2 = numpy.sin(thetJ,dtype = dtype_temp)
        
        jac[1:m+1] = a1 * numpy.sin((k_values + 1) * thetJ, dtype = dtype_temp
                                    ) / a2 + (1 - (k_values + 1) * a1) * numpy.cos(k_values * thetJ,dtype = dtype_temp)
    
    elif User_DampingKernel == "LanczosSigma":
        thetL = numpy.pi / (m + 1)

        jac[1:m+1] = numpy.sin(k_values * thetL, dtype = dtype_temp) / (k_values * thetL)
    
    else: 
        jac[1:] = 1.0

    jac[0] = 0.5  # Adjust for the 1/2 factor in the zeroth term of Chebyshev expansion

    return jac



# =====================
# Extremal
# ========================

# NOTE On Cpu Determines polynomial for end interval cases.
def OOC3_PolynomialParams_aa_bb_ExtremalPolynomialParams(pol, a, b):
    # NOTE aIn,bIn is the wanted region
    User_MaximumDegree = pol.User_MaximumDegree
    User_MinimumDegree = pol.User_MinimumDegree
    #thresh = 0.1 # NOTE 1:9 in convergence ratio 

    PART00_TidyUpUnwantedInterval = True
    if PART00_TidyUpUnwantedInterval:

        # NOTE gam is the sign change in sigma!
        gam = numpy.sign(b - (1 - pol.User_ExtremalIntervalDefinition))
        x = a if gam > 0 else b
        # NOTE Adjust
        a , b = tuple(sorted([x - gam * 0.1 * numpy.sqrt((1 - gam * x) /2,dtype = numpy.float64) , -1 * gam])) 

        

    PART01_FindTheClosestDegree = True
    if PART01_FindTheClosestDegree:

        # NOTE y is the upper bound of unwanted interval if we want right extremal
        c = (b + a) * 0.5
        e = (b - a) * 0.5

        sigma = e / (gam - c)
        sigma1 = sigma

        g0 = 1.0
        g1 = (x - c) * (sigma1 / e)

        m_closest = 1
        # NOTE determine deg
        for k in range(2, User_MaximumDegree + 1):

            sigma_new = 1.0 / (2.0 / sigma1 - sigma)
            s1 = sigma_new / e

            gnew = 2 * (x - c) * s1 * g1 - (sigma * sigma_new) * g0

            g0 = numpy.copy(g1)
            g1 = numpy.copy(gnew)
            sigma = sigma_new
            m_closest += 1

            if (g1 < 0.1) & (k >= User_MinimumDegree): # NOTE Let's fix at 0.01. Otherwise it is easy for kpm to have a low degree and makes inaccurate (or negative estimate due to gibbs osc).
                #print('broken at ', g1)
                break

        fixed_bar = numpy.copy(g1)



    

    # NOTE Determine the Raw Coefficients
    #      The 
    PART02_CompileCoefficients = True
    if PART02_CompileCoefficients:
        
        sigma = e / (gam - c)
        sigma1 = sigma          # NOTE unchanged
        tau = 2/sigma1

        g0 = numpy.zeros(User_MaximumDegree + 1,dtype = numpy.float64)
        g1 = numpy.zeros(User_MaximumDegree + 1,dtype = numpy.float64)
        gnew = numpy.zeros(User_MaximumDegree + 1,dtype = numpy.float64)

        g0[0] = 1.0
        g1[0] = -c * (sigma1 / e)
        g1[1] =  1 * (sigma1 / e)

        for k in range(2, m_closest + 1):

            sigma_new = 1.0 / (tau - sigma)
            s1 = sigma_new / e

            # NOTE Fold left one place of g1 to the front
            gnew[:k] = g1[1:k+1]
            gnew[1] += g1[0] 

            # NOTE Fold one last time to make it 2* 
            gnew[1:k+1] += g1[:k]
            gnew *= s1

            # NOTE Remove the extra
            gnew[:k+1] -= (2 * s1 * c) * g1[:k+1] 

            # NOTE The rest same as before 
            gnew[:k+1] -= (sigma * sigma_new) * g0[:k+1]
            

            g0[:k+1] = numpy.copy(g1[:k+1])
            g1[:k+1] = numpy.copy(gnew[:k+1])

            sigma = sigma_new


    # ====================
    # Update the pol
    # ====================
    pol.AdjustedCoefficients = g1[:m_closest + 1] * pol.AdjustedDampingKernels[:m_closest + 1] # NOTE Apply Damping
    pol.AdjustedDegree = m_closest
    pol.bar = fixed_bar
    pol.gam = gam

    return pol




# =======================
# Interior
# ===========================
# NOTE Referenced on EVSL
def dif_eval(v, thc, jac):
    return numpy.sum(v * numpy.cos(numpy.arange(jac.shape[0]) * thc,dtype = numpy.float64) * jac)

# NOTE Given degree, find a suitable center to 'minimize' the effect of gibbs oscillation
def rootchb(m, v, jac, tha, thb):


    # NOTE tolerance for convergence.
    tolBal = abs(tha - thb) * 1e-6
    thc = (tha + thb) / 2
    thc_old = numpy.copy(thc)
    # NOTE In case dif_eval(v, thb, jac) < 0 or dif_eval(v, tha, jac) > 0, there is no 
    # NOTE While it is imbalanced it is still useful when we cannot afford increasing the degree.
    #fb = dif_eval(v, thb, jac)
    #fa = dif_eval(v, tha, jac)
    
    #if (fa > 0) or (fb < 0):
    #    
    #    return None, None  # No hope of a balance!

    for _ in range(759): # NOTE Restrict to 759 steps.

        fval = dif_eval(v,  thc, jac)

        # NOTE Record the difference as d
        d = numpy.sum(jac[1:m+1] * numpy.arange(1, m+1,dtype = numpy.float64
                                                ) * numpy.sin(numpy.arange(1, m+1) * thc,dtype = numpy.float64
                                                              ) * v[1:m+1])
        thN = thc + fval / d

        if (abs(fval) < tolBal
                ) or (
                    abs(thc - thN) < numpy.finfo(numpy.float64).eps * abs(thc)): # NOTE Stop if equal!
            break

        if fval > 0:  # NOTE The bisection step
            if (thN < thb) or (thN > tha):
                thN = 0.5 * (thc + tha)
            thb = thc
        else:
            if (thN < thb) or (thN > tha):
                thN = 0.5 * (thc + thb)
            tha = thc

        thc = thN

    #print("Center shift", numpy.abs(thc_old - thc)) # NOTE Very small... especially for higher degree
    mu = numpy.cos(numpy.arange(jac.shape[0]) * thc,dtype = numpy.float64) * jac 
    return mu, thc

# NOTE Do the polynomial for numbers
def ChebIv(m, mu, xi, dtype_temp = numpy.float64):
    n = len(xi)
    vkm1 = numpy.zeros(n, dtype = dtype_temp)
    vk = numpy.ones(n, dtype = dtype_temp)
    yi = vk * mu[0]  # NOTE Isn't mu[0] == 1?

    for k in range(1, m + 1):

        scal = 1.0 if (k == 1) else 2.0
        
        vkp1 = scal * xi * vk - vkm1
        yi += mu[k] * vkp1  

        numpy.copyto(vkm1, vk)
        numpy.copyto(vk, vkp1)

    return yi


def OOC3_PolynomialParams_aa_bb_InteriorPolynomialParams(pol, aa, bb):

    thb = numpy.arccos(bb)
    tha = numpy.arccos(aa)

    # Initialize variables
    mu = numpy.zeros(pol.User_MaximumDegree + 1,dtype = numpy.float64)
    v = numpy.zeros(pol.User_MaximumDegree + 1,dtype = numpy.float64)


    # NOTE Initialize v for the first few
    v[:pol.User_MinimumDegree] = numpy.cos(numpy.arange(pol.User_MinimumDegree) * thb,dtype = numpy.float64
                                        ) - numpy.cos(numpy.arange(pol.User_MinimumDegree) * tha,dtype = numpy.float64)


    #print(pol.User_MinimumDegree, pol.User_MaximumDegree)
    for m in range(pol.User_MinimumDegree, pol.User_MaximumDegree):

        # NOTE Update v there after
        v[m] = numpy.cos(m * thb,dtype = numpy.float64) - numpy.cos(m * tha,dtype = numpy.float64)


        # NOTE Damping factor
        jac = OOC1_DegreeM_DampingKernel(m, 
                        User_MaximumDegree = pol.User_MaximumDegree, 
                        User_DampingKernel = pol.User_DampingKernel)   

        new_mu, new_thc = rootchb(m, v, jac, tha, thb)

        if new_mu is None: 
            #continue
            print("This is now abolished. ", m)
            new_thc = (tha + thb) / 2
            gam = numpy.cos(new_thc,dtype = numpy.float64)
            new_mu = numpy.cos(numpy.arange(jac.shape[0]) * gam,dtype = numpy.float64) * jac 
        else:
            pass
            # NOTE if convergence is not found or impossible to be balanced we cannot take the subopotimal
        gam = numpy.cos(new_thc)
        t = ChebIv(m, new_mu, numpy.array([gam],dtype = numpy.float64))
        vals = ChebIv(m, new_mu, numpy.array([aa, bb],dtype = numpy.float64))
        mu = new_mu / t # NOTE Eqn 2.4 EVSL 
        if numpy.all(vals <= t * pol.User_ConvergenceRatio):
            m += 1
            break

    if m == pol.User_MaximumDegree:
        print("WARNING. Consider to increase the User_MaximumDegree.")
    #pol.bar = numpy.min(vals) / t  # NOTE Often too stringent...
    #print("I a mbar" , pol.bar)
    pol.gam = gam               # NOTE The center in [-1,1]
    pol.AdjustedDegree = m - 1             # NOTE Stored the best deg as determined in the newton iteration.
    pol.AdjustedCoefficients = mu                 # NOTE all the coefficients!

    #print(pol.AdjustedCoefficients)
    return pol



# ===========================
# Main
# ============================



def OOC0_OptimizePolynomialParamsOnDegree(User_MaximumDegree = 1000,
                             User_MinimumDegree = 10,
                             User_DampingKernel = "Jackson",
                             User_ExtremalIntervalDefinition = 1e-9,
                             User_WantedInterval = (36.8, 37.2),
                             User_SpectrumBound = (1.0, 759.0),
                             User_ConvergenceRatio = 0.1,
                             ):

    
    #User_WantedInterval[0], User_WantedInterval[1], User_SpectrumBound[0], User_SpectrumBound[1]


    PART00_Initialize = True
    if PART00_Initialize:


        pol = PolParams(User_MaximumDegree=User_MaximumDegree, 
                            User_MinimumDegree=User_MinimumDegree, 
                            User_DampingKernel=User_DampingKernel, 
                            User_ExtremalIntervalDefinition=User_ExtremalIntervalDefinition, 
                            User_ConvergenceRatio = User_ConvergenceRatio 
                            )



    PART01_AdjustIntervalAndThreshold = True
    if PART01_AdjustIntervalAndThreshold:
        # NOTE 
        aa = max(User_WantedInterval[0], User_SpectrumBound[0])
        bb = min(User_WantedInterval[1], User_SpectrumBound[1])

        cc = 0.5 * (User_SpectrumBound[1] + User_SpectrumBound[0])
        dd = 0.5 * (User_SpectrumBound[1] - User_SpectrumBound[0])

        aa = max((aa - cc) / dd,    -1.0) # NOTE Centered and bound within -1 1
        bb = min((bb - cc) / dd,     1.0)



        # NOTE Too close to boundary! We eliminate the case where aboth are near boundary
        if (aa <= -1.0 + User_ExtremalIntervalDefinition):
            aa = -1.0
        if (bb >= 1.0 - User_ExtremalIntervalDefinition):
            bb = 1.0


        assert not ((aa <= -1.0 + User_ExtremalIntervalDefinition) & (bb >= 1.0 - User_ExtremalIntervalDefinition)) , "ABORTED. Too greedy! Both ends of the wanted interval are close to the extremal. relax either side"
        thb = numpy.arccos(bb,dtype = numpy.float64)
        tha = numpy.arccos(aa,dtype = numpy.float64)


        pol.theta_a = tha
        pol.theta_b = thb
        pol.cc = cc
        pol.dd = dd


        # NOTE Precalculate Jackson User_DampingKernel kernel 
        pol.AdjustedDampingKernels = OOC1_DegreeM_DampingKernel(pol.User_MaximumDegree,  
                         User_MaximumDegree = pol.User_MaximumDegree + 1, 
                         User_DampingKernel = pol.User_DampingKernel) # NOTE Jackson kernel by default


    PART02_UpdatePolynomialParam = True
    if PART02_UpdatePolynomialParam:
        
        
        if ((aa <= -1.0 + User_ExtremalIntervalDefinition) or (bb  >= 1.0 - User_ExtremalIntervalDefinition)) : # NOTE Extremal case
            #print("i am aa bb ghagfdgdaf",aa, bb)
            # NOTE Then there is no way we will have a balance 
            pol = OOC3_PolynomialParams_aa_bb_ExtremalPolynomialParams(pol, aa, bb)
        else:
            pol = OOC3_PolynomialParams_aa_bb_InteriorPolynomialParams(pol, aa, bb)
    

    pol.User_WantedInterval = User_WantedInterval
    pol.User_SpetrumBound = User_SpectrumBound
    #print("Am i there?", pol.AdjustedCoefficients)
    return pol



