# =======
# Imports
# =======

# Python packages
import numpy
import scipy
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg
import sksparse
from sksparse.cholmod import cholesky
from scipy.optimize import minimize
import scipy.interpolate
from functools import partial
import multiprocessing
# import logging
# import psutil
# import ray

# Classes
from TraceEstimation import TraceEstimation
from PlotSettings import *
import RootFinding
import LinearAlgebra

# ====================
# Sparse Linear Solver
# ====================

def SparseLinerSolver(A,b):

    # Use direct method
    # x = scipy.sparse.linalg.spsolve(A,b)

    # Use iterative method
    Tol = 1e-5
    if b.ndim == 1:
        x = scipy.sparse.linalg.cg(A,b,tol=Tol)[0]
    else:
        x = numpy.zeros(b.shape)
        for i in range(x.shape[1]):
            x[:,i] = scipy.sparse.linalg.cg(A,b[:,i],tol=Tol)[0]

    return x

# ===========================
# Likelihood Estimation Class
# ===========================

class LikelihoodEstimation():

    # ====================
    # Monte Carlo Sampling
    # ====================

    # @ray.remote
    @staticmethod
    def MonteCarloSampling(A,Method='GolubKahnBidiagonalization',LanczosDegree=20):
        """
        Method should be 'LanczosTridiagonalization' or 'GolubKahnLanczosBidiagonalization'.
        LanczosDegree is an integer.
        """

        # Radamacher random vector, consists of 1 and -1.
        n = A.shape[0]
        w = numpy.sign(numpy.random.randn(n))

        if Method == 'LanczosTridiagonalization':
            # Lanczos recustive iteration to convert A to tridiagonal form T
            # T = LinearAlgebra.LanczosTridiagonalization(A,w,LanczosDegree)
            T = LinearAlgebra.LanczosTridiagonalization2(A,w,LanczosDegree)

            # Spectral decomposition of T
            Eigenvalues,Eigenvectors = numpy.linalg.eigh(T)

            Theta = numpy.abs(Eigenvalues)
            Tau2 = Eigenvectors[0,:]**2

            # Here, f(theta) = log(theta), since log det is trace of log
            LogDetEstimate = numpy.sum(Tau2 * (numpy.log(Theta))) * n

        elif Method == 'GolubKahnLanczosBidiagonalization':

            # Use Golub-Kahn-Lanczos bidigonalization instead of Lanczos tridiagonalization
            B = LinearAlgebra.GolubKahnLanczosBidiagonalization(A,w,LanczosDegree)
            LeftEigenvectors,SingularValues,RightEigenvectorsTransposed = numpy.linalg.svd(B)
            Theta = SingularValues   # Theta is just singular values, not singular values squared
            Tau2 = RightEigenvectorsTransposed[:,0]**2

            # Here, f(theta) = log(theta), since log det X = trace of log X.
            LogDetEstimate = numpy.sum(Tau2 * (numpy.log(Theta))) * n

        else:

            raise ValueError('Method should be "LanczosTridiagonalization" or "GolubKahnLanczosBidiagonalization"')

        return LogDetEstimate

    # =======================
    # Compute Log Determinant
    # =======================

    @staticmethod
    def ComputeLogDeterminant(A):
        """
        Based on https://www-users.cs.umn.edu/~saad/PDF/ys-2016-04.pdf
        
        In Lanczos tridiagonalization method, Theta is the eigenvalues of T. 
        However, in Golub-Kahn bidoagonalization method, Theta is the singular values of B.
        The relation between these two methords are are follows: B.T*B is the T for A.T*A.
        That is, if we have the input matrix A.T*T, its Lanczos tridiagonalization T is the same matrix
        as if we bidiagonalize A (not A.T*A) with Golub-Kahn to get B, then T = B.T*B.
        This has not been highlighted paper in the above paper.

        To correctly implement Golub-Kahn, here Theta should be the singular values of B, NOT
        the square of the singular values of B (as decribedin the paper incorrectly!).
        """

        # Use direct method (Cholesk factorization) or iterative method (Lanczos iteration)
        DirectMethod = False

        if DirectMethod:

            # Direct method
            Factor = sksparse.cholmod.cholesky(A)
            LogDe = Factor.logdet()

        else:

            # Iterative method. Number of iterations and number of Lanczos steps in each iteration
            NumIterations = 20
            LanczosDegree = 20
            # Method = 'LanczosTridiagonalization'         # This is fine for computing trace but not log det. If A has many zero singular values, use Golub-Kahn. 
            Method = 'GolubKahnLanczosBidiagonalization'   # for computing log det, this is prefered.

            # No Parallel processing
            LogDetEstimatesList = [LikelihoodEstimation.MonteCarloSampling(A,Method,LanczosDegree) for i in range(NumIterations)]

            # Parallel processing with Ray
            # Get number of cpus
            # NumProcessors = psutil.cpu_count()

            # Parallelization with ray
            # ray.init(num_cpus=NumProcessors,logging_level=logging.FATAL)

            # Put A into object store
            # A_id = ray.put(A)

            # Parallel section with ray. This just creates process Ids. It does not do computation
            # Process_Ids = [MonteCarloSampling.remote(A_id,Method,LanczosDegree) for i in range(NumIterations)]

            # Do the parallel computations
            # LogDetEstimatesList = ray.get(Process_Ids)

            # ray.shutdown()

            # LogDet = numpy.mean(numpy.array(LogDetEstimatesList))

            # Parallel processing with multiprocessing
            # NumProcessors = multiprocessing.cpu_count()
            # pool = multiprocessing.Pool(processes=NumProcessors)
            # ChunkSize = int(NumIterations / NumProcessors)
            # if ChunkSize < 1:
            #     ChunkSize = 1

            # MonteCarloSampling_PartialFunction = partial(LikelihoodEstimation.MonteCarloSampling,A,Method,LanczosDegree)

            # Iterations = range(NumIterations)
            # Processes = [pool.apply_async(LikelihoodEstimation.MonteCarloSampling,(A,Method,LanczosDegree)) for i in range(NumIterations)]
            # pool.close()
            # pool.join()
            # LogDetEstimatesList = [Process.get() for Process in Processes]

            LogDet = numpy.mean(numpy.array(LogDetEstimatesList))

        return LogDet

    # ================================
    # Log Likelihood With Sigma Sigma0
    # ================================

    @staticmethod
    def LogLikelihoodWithSigmaSigma0(z,X,K,TraceEstimationUtilities,SignSwitch,Parameters):
        """
        Here we use direct parameter, sigma and sigma0

        SignSwitch chnages the sign of the output from lp to -lp. When True, this is used to
        minimizing (instad of maximizing) the negative of log-likelihood function.
        """

        # Determine to use sparse
        UseSparse = False
        if scipy.sparse.isspmatrix(K):
            UseSparse = True

        # Parameters
        sigma = Parameters[0]
        sigma0 = Parameters[1]

        # Covariance
        if UseSparse:
            I = scipy.sparse.eye(K.shape[0],format='csc')
        else:
            I = numpy.eye(K.shape[0])
        S = (sigma**2)*K + (sigma0**2)*I

        # Compute log det (S)
        if TraceEstimationUtilities['UseEigenvaluesMethod'] == True:
            # Use eigenvalues of K to estimate trace
            K_eigenvalues = TraceEstimationUtilities['EigenvaluesMethodUtilities']['K_eigenvalues']
            LogDet_S = numpy.sum(numpy.log((sigma**2)*K_eigenvalues + (sigma0**2)))
        else:
            # Use Cholesky factorization
            if UseSparse:
                LogDet_S = LikelihoodEstimation.ComputeLogDeterminant(S)
            else:
                L = scipy.linalg.cholesky(S,lower=True,overwrite_a=True)
                Diag_L = numpy.diag(L).astype(numpy.complex128)
                LogDet_L = numpy.real(numpy.sum(numpy.log(Diag_L)))
                LogDet_S = 2.0*LogDet_L

        # Compute log det (X.T*Sinv*X)
        if UseSparse:
            # Y = scipy.sparse.linalg.spsolve(S,X)
            # w = scipy.sparse.linalg.spsolve(S,z)
            Y = SparseLinerSolver(S,X)
            w = SparseLinerSolver(S,z)
        else:
            Y = scipy.linalg.solve(S,X,sym_pos=True)
            w = scipy.linalg.solve(S,z,sym_pos=True)

        XtSinvX = numpy.matmul(X.T,Y)
        LogDet_XtSinvX = numpy.log(numpy.linalg.det(XtSinvX))

        Binv = numpy.linalg.inv(XtSinvX)
        YBinvYt = numpy.matmul(Y,numpy.matmul(Binv,Y.T))

        # Log likelihood
        lp = -0.5*LogDet_S -0.5*LogDet_XtSinvX -0.5*numpy.dot(z,w-numpy.dot(YBinvYt,z))

        # If lp is used in scipy.optimize.minimize, change the sign to optain the minimum of -lp
        if SignSwitch == True:
            lp = -lp

        return lp

    # =============================
    # Log Likelihood With Sigma Eta
    # =============================

    @staticmethod
    def LogLikelihoodWithSigmaEta(z,X,K,TraceEstimationUtilities,SignSwitch,Parameters):
        """
        Log likelihood function

            L = -(1/2) log det(S) - (1/2) log det(X.T*Sinv*X) - (1/2) sigma^(-2) * z.T * M1 * z

        where
            S = sigma^2 Kn is the covariance
            Sinv is the inverse of S
            M1 = Sinv = Sinv*X*(X.T*Sinv*X)^(-1)*X.T*Sinv

        Parameters = [sigma,eta]

        SignSwitch chnages the sign of the output from lp to -lp. When True, this is used to
        minimizing (instad of maximizing) the negative of log-likelihood function.
        """

        # Determine to use sparse
        UseSparse = False
        if scipy.sparse.isspmatrix(K):
            UseSparse = True

        # Parameters
        sigma = Parameters[0]
        eta = Parameters[1]

        # Correlation
        if UseSparse:
            I = scipy.sparse.eye(K.shape[0],format='csc')
        else:
            I = numpy.eye(K.shape[0])
        Kn = K + eta*I

        # Covariance
        S = sigma**2 * Kn

        # Compute log det(Kn)
        if TraceEstimationUtilities['UseEigenvaluesMethod'] == True:
            # Use eigenvalues to estimate the trace
            K_eigenvalues = TraceEstimationUtilities['EigenvaluesMethodUtilities']['K_eigenvalues']
            LogDet_Kn = numpy.sum(numpy.log(K_eigenvalues + eta))
        else:
            # Use Cholesky factorization
            if UseSparse:
                Factor = sksparse.cholmod.cholesky(Kn)
                LogDet_Kn = Factor.logdet()
            else:
                L = scipy.linalg.cholesky(Kn,lower=True,overwrite_a=True)
                Diag_L = numpy.diag(L).astype(numpy.complex128)
                LogDet_L = numpy.real(numpy.sum(numpy.log(Diag_L)))
                LogDet_Kn = 2.0*LogDet_L

        # Compute log det (X.T Kn_inv X)
        n,m = X.shape
        if UseSparse:
            # Y = scipy.sparse.linalg.spsolve(Kn,X)
            # w = scipy.sparse.linalg.spsolve(Kn,z)
            Y = SparseLinerSolver(Kn,X)
            w = SparseLinerSolver(Kn,z)
        else:
            Y = scipy.linalg.solve(Kn,X,sym_pos=True)
            w = scipy.linalg.solve(Kn,z,sym_pos=True)

        XtKninvX = numpy.matmul(X.T,Y)
        LogDet_XtKninvX = numpy.log(numpy.linalg.det(XtKninvX))

        # Suppose B is XtKninvX found above. We compute inverse of B
        Binv = numpy.linalg.inv(XtKninvX)
        YBinvYt = numpy.matmul(Y,numpy.matmul(Binv,Y.T))

        # Log likelihood
        lp = -0.5*(n-m)*numpy.log(sigma**2) - 0.5*LogDet_Kn - 0.5*LogDet_XtKninvX -(0.5/(sigma**2))*numpy.dot(z,w-numpy.dot(YBinvYt,z))

        # If lp is used in scipy.optimize.minimize, change the sign to optain the minimum of -lp
        if SignSwitch == True:
            lp = -lp

        return lp

    # ===============================
    # Log Likelihood First Derivative
    # ===============================

    @staticmethod
    def LogLikelihoodFirstDerivative(z,X,K,TraceEstimationUtilities,LogEta):
        """
        lp is the log likelihood probability
        dlp_deta is d(lp)/d(eta), is the derivative of lp with respect to eta when the optimal
        value of sigma is subtituted in the likelihood function per given eta.
        """

        # Determine to use sparse
        UseSparse = False
        if scipy.sparse.isspmatrix(K):
            UseSparse = True

        # Change LogEta to eta
        if numpy.isneginf(LogEta):
            eta = 0.0
        else:
            eta = 10.0**LogEta

        # Correlation
        if UseSparse:
            I = scipy.sparse.eye(K.shape[0],format='csc')
        else:
            I = numpy.eye(K.shape[0])
        Kn = K + eta*I
   
        # Compute Kn_inv*X and Kn_inv*z
        if UseSparse:
            # Y = scipy.sparse.linalg.spsolve(Kn,X)
            # w = scipy.sparse.linalg.spsolve(Kn,z)
            Y = SparseLinerSolver(Kn,X)
            w = SparseLinerSolver(Kn,z)
        else:
            Y = scipy.linalg.solve(Kn,X,sym_pos=True)
            w = scipy.linalg.solve(Kn,z,sym_pos=True)

        n,m = X.shape

        # Splitting M into M1 and M2. Here, we compute M2
        B = numpy.matmul(X.T,Y)
        Binv = numpy.linalg.inv(B)
        Ytz = numpy.matmul(Y.T,z)
        Binv_Ytz = numpy.matmul(Binv,Ytz)
        Y_Binv_Ytz = numpy.matmul(Y,Binv_Ytz)
        Mz = w - Y_Binv_Ytz

        # Traces
        if TraceEstimationUtilities['UseEigenvaluesMethod'] == True:
            K_eigenvalues = TraceEstimationUtilities['EigenvaluesMethodUtilities']['K_eigenvalues']
            TraceKninv = numpy.sum(1.0/(K_eigenvalues + eta))
        else:
            TraceKninv = TraceEstimation.EstimateTrace(TraceEstimationUtilities,eta)
            # TraceKninv = TraceEstimation.ComputeTraceOfInverse(Kn)    # Use direct method without interpolation, Test
        YtY = numpy.matmul(Y.T,Y)
        TraceBinvYtY = numpy.trace(numpy.matmul(Binv,YtY))
        # TraceBinvYtY = numpy.trace(numpy.matmul(Y,numpy.matmul(Binv,Y.T)))
        TraceM = TraceKninv - TraceBinvYtY

        # Derivative of log likelihood
        zMz = numpy.dot(z,Mz)
        zM2z = numpy.dot(Mz,Mz)
        sigma02 = zMz/(n-m)
        # dlp_deta = -0.5*((TraceM/(n-m))*zMz - zM2z)
        dlp_deta = -0.5*(TraceM - zM2z/sigma02)

        return dlp_deta

    # ================================
    # Log Likelihood Second Derivative
    # ================================

    @staticmethod
    def LogLikelihoodSecondDerivative(z,X,K,TraceEstimationUtilities,eta):
        """
        The second derivative of lp is computed as a function of only eta. Here, we
        substituted optimal value of sigma, which istself is a function of eta.
        """

        # Determin to use sparse
        UseSparse = False
        if scipy.sparse.isspmatrix(K):
            UseSparse = True

        # Correlation
        if UseSparse:
            I = scipy.sparse.eye(K.shape[0],format='csc')
        else:
            I = numpy.eye(K.shape[0])
        Kn = K + eta*I

        if UseSparse:
            # Y = scipy.sparse.linalg.spsolve(Kn,X)
            # V = scipy.sparse.linalg.spsolve(Kn,Y)
            # w = scipy.sparse.linalg.spsolve(Kn,z)
            Y = SparseLinerSolver(Kn,X)
            V = SparseLinerSolver(Kn,Y)
            w = SparseLinerSolver(Kn,z)
        else:
            Y = scipy.linalg.solve(Kn,X,sym_pos=True)
            V = scipy.linalg.solve(Kn,Y,sym_pos=True)
            w = scipy.linalg.solve(Kn,z,sym_pos=True)

        n,m = X.shape

        # Splitting M
        B = numpy.matmul(X.T,Y)
        Binv = numpy.linalg.inv(B)
        Ytz = numpy.matmul(Y.T,z)
        Binv_Ytz = numpy.matmul(Binv,Ytz)
        Y_Binv_Ytz = numpy.matmul(Y,Binv_Ytz)
        Mz = w - Y_Binv_Ytz

        # Trace of M
        if TraceEstimationUtilities['UseEigenvaluesMethod'] == True:
            # Use eigenvalues method
            K_eigenvalues = TraceEstimationUtilities['EigenvaluesMethodUtilities']['K_eigenvalues']
            TraceKninv = numpy.sum(1.0/(K_eigenvalues + eta))
        else:
            # Use interpolation method
            TraceKninv = TraceEstimation.EstimateTrace(TraceEstimationUtilities,eta)
        YtY = numpy.matmul(Y.T,Y)
        A = numpy.matmul(Binv,YtY)
        TraceA = numpy.trace(A)
        TraceM = TraceKninv - TraceA

        # Trace of M**2
        if TraceEstimationUtilities['UseEigenvaluesMethod'] == True:
            K_eigenvalues = TraceEstimationUtilities['EigenvaluesMethodUtilities']['K_eigenvalues']
            TraceKn2inv = numpy.sum(1.0/((K_eigenvalues + eta)**2))
        else:
            if UseSparse:
                Kn2 = Kn.dot(Kn)
            else:
                Kn2 = numpy.matmul(Kn,Kn)
            TraceKn2inv = TraceEstimation.ComputeTraceOfInverse(Kn2)
        YtV = numpy.matmul(Y.T,V)
        C = numpy.matmul(Binv,YtV)
        TraceC = numpy.trace(C)
        AA = numpy.matmul(A,A)
        TraceAA = numpy.trace(AA)
        TraceM2 = TraceKn2inv - 2.0*TraceC + TraceAA

        # Find z.T * M**3 * z
        YtMz = numpy.matmul(Y.T,Mz)
        Binv_YtMz = numpy.matmul(Binv,YtMz)
        Y_Binv_YtMz = numpy.matmul(Y,Binv_YtMz)

        if UseSparse:
            # v = scipy.sparse.linalg.spsolve(Kn,Mz)
            v = SparseLinerSolver(Kn,Mz)
        else:
            v = scipy.linalg.solve(Kn,Mz,sym_pos = True)
        MMz = v - Y_Binv_YtMz

        # Second derivative (only at the location ofzero first derivative)
        zMz = numpy.dot(z,Mz)
        zM2z = numpy.dot(Mz,Mz)
        zM3z = numpy.dot(Mz,MMz)
        sigma02 = zMz / (n-m)
        # d2lp_deta2 = 0.5*(TraceM2 * zM2z - 2.0*TraceM * zM3z)
        d2lp_deta2 = (0.5/sigma02)*((TraceM2/(n-m) + (TraceM/(n-m))**2) * zMz - 2.0*zM3z)

        return d2lp_deta2

    # ======================================
    # Maximize Log Likelihood With Sigma Eta
    # ======================================

    @staticmethod
    def MaximizeLogLikelihoodWithSigmaEta(z,X,K,TraceEstimationUtilities):
        """
        Maximizing the log-likelihood function over the space of parameters sigma and eta.
        """

        print('Maximize log likelihood with sigma sigma0 ...')

        # Initial points # SETTING
        Guess_sigma = 0.1
        Guess_eta = 0.1
        GuessParameters = [Guess_sigma,Guess_eta]

        # Partial function with minus to make maximization to a minimization
        LogLikelihood_PartialFunction = partial(
                LikelihoodEstimation.LogLikelihoodWithSigmaEta,
                z,X,K,TraceEstimationUtilities,True)

        # Minimize
        # Method = 'BFGS'
        # Method = 'CG'
        Method = 'Nelder-Mead'
        Tolerance = 1e-6 # SETTING
        Res = scipy.optimize.minimize(LogLikelihood_PartialFunction,GuessParameters,method=Method,tol=Tolerance)

        # Extract results
        sigma = Res.x[0]
        eta = Res.x[1]
        sigma0 = numpy.sqrt(eta) * sigma
        max_lp = -Res.fun

        # Output distionary
        Results =  \
        {
                'sigma': sigma,
                'sigma0' : sigma0,
                'eta': eta,
                'max_lp': max_lp
        }
        
        return Results

    # =========================================
    # Maximize Log Likelihood With Sigma Sigma0
    # =========================================

    @staticmethod
    def MaximizeLogLikelihoodWithSigmaSigma0(z,X,K,TraceEstimationUtilities):
        """
        Maximizing the log-likelihood function over the space of parameters sigma and eta.
        """

        print('Maximize log likelihood with sigma sigma0 ...')

        # Initial points # SETTING
        Guess_sigma = 0.001
        Guess_sigma0 = 0.001
        GuessParameters = [Guess_sigma,Guess_sigma0]

        # Partial function with minus to make maximization to a minimization
        LogLikelihood_PartialFunction = partial(
                LikelihoodEstimation.LogLikelihoodWithSigmaSigma0,
                z,X,K,TraceEstimationUtilities,True)

        # Minimize
        # Method = 'BFGS'
        # Method = 'CG'
        Method = 'Nelder-Mead'
        Tolerance = 1e-6 # SETTING
        Res = scipy.optimize.minimize(LogLikelihood_PartialFunction,GuessParameters,method=Method,tol=Tolerance)
        print('Iter: %d, Eval: %d, Success: %s'%(Res.nit,Res.nfev,Res.success))

        # Extract results
        sigma = Res.x[0]
        sigma0 = Res.x[1]
        eta = (sigma0/sigma)**2
        max_lp = -Res.fun

        # Output distionary
        Results =  \
        {
                'sigma': sigma,
                'sigma0' : sigma0,
                'eta': eta,
                'max_lp': max_lp
        }
        
        return Results

    # ======================================
    # Find Zero of Log Likelihood Derivative
    # ======================================

    @staticmethod
    def FindZeroOfLogLikelihoodFirstDerivative(z,X,K,TraceEstimationUtilities,Interval_eta):
        """
        root finding of the derivative of lp.
        The log likelihood function is implicitly a function of eta. We have substituted the
        value of optimal sigma, which itself is a function of eta.
        """

        # ------------------
        # Find Optimal Sigma
        # ------------------

        def FindOptimalSigma(z,X,K,eta):
            """
            Based on a given eta, finds optimal sigma
            """

            # Determine to use sparse
            UseSparse = False
            if scipy.sparse.isspmatrix(K):
                UseSparse = True

            if UseSparse:
                I = scipy.sparse.eye(K.shape[0],format='csc')
            else:
                I = numpy.eye(K.shape[0])
            Kn = K +  eta*I

            if UseSparse:
                # Y = scipy.sparse.linalg.spsolve(Kn,X)
                # w = scipy.sparse.linalg.spsolve(Kn,z)
                Y = SparseLinerSolver(Kn,X)
                w = SparseLinerSolver(Kn,z)
            else:
                Y = scipy.linalg.solve(Kn,X,sym_pos=True)
                w = scipy.linalg.solve(Kn,z,sym_pos=True)

            n,m = X.shape
            B = numpy.matmul(X.T,Y)
            Binv = numpy.linalg.inv(B) 
            Ytz = numpy.matmul(Y.T,z)
            v = numpy.matmul(Y,numpy.matmul(Binv,Ytz))
            sigma2 = numpy.dot(z,w-v) / (n-m)
            sigma = numpy.sqrt(sigma2)

            return sigma

        # -------------------
        # Find Optimal Sigma0
        # -------------------

        def FindOptimalSigma0(z,X):
            """
            When eta is very large, we assume sigma is zero. Thus, sigma0 is computed by 
            this function.
            """

            n,m = X.shape
            B = numpy.matmul(X.T,X)
            Binv = numpy.linalg.inv(B)
            Xtz = numpy.matmul(X.T,z)
            v = numpy.matmul(X,numpy.matmul(Binv,Xtz))
            sigma02 = numpy.dot(z,z-v) / (n-m)
            sigma0 = numpy.sqrt(sigma02)

            return sigma0

        # -----------------

        print('Find root of log likelihood derivative ...')

        # Find an interval that the function changes sign before finding its root (known as bracketing the function)
        LogEta_Start = numpy.log10(Interval_eta[0])
        LogEta_End = numpy.log10(Interval_eta[1])

        # Partial function with minus to make maximization to a minimization
        LogLikelihoodFirstDerivative_PartialFunction = partial(
                LikelihoodEstimation.LogLikelihoodFirstDerivative,
                z,X,K,TraceEstimationUtilities)

        # Initial points
        Bracket = [LogEta_Start,LogEta_End]
        NumTrials = 3    # SETTING
        BracketFound,Bracket,BracketValues = RootFinding.FindIntervalWithSignChange(LogLikelihoodFirstDerivative_PartialFunction,Bracket,NumTrials,args=(),)

        if BracketFound:
            # There is a sign change in the interval of eta. Find root of lp derivative
            Tolerance = 1e-6     # SETTING
            MaxIterations = 100  # SETTING

            # Find roots using Brent method
            # Method = 'brentq'
            # Res = scipy.optimize.root_scalar(LogLikelihoodFirstDerivative_PartialFunction,bracket=Bracket,method=Method,xtol=Tolerance)
            # print('Iter: %d, Eval: %d, Converged: %s'%(Res.iterations,Res.function_calls,Res.converged))
            
            # Find roots using Chandraputala method
            Res = RootFinding.ChandrupatlaMethod(LogLikelihoodFirstDerivative_PartialFunction,Bracket,BracketValues,verbose=False,eps_m=Tolerance,eps_a=Tolerance,maxiter=MaxIterations)
            print('Iter: %d'%(Res['iterations']))

            # Extract results
            # eta = 10**Res.root                       # Use with Brent
            eta = 10**Res['root']                      # Use with Chandrupatla
            sigma = FindOptimalSigma(z,X,K,eta)
            sigma0 = numpy.sqrt(eta) * sigma

            # Check second derivative
            Success = True
            # d2lp_deta2 = LikelihoodEstimation.LogLikelihoodSecondDerivative(z,X,K,TraceEstimationUtilities,eta)
            # if d2lp_deta2 < 0:
            #     Success = True
            # else:
            #     Success = False

        else:
            # Bracket with sign change was not found.

            # Evaluate the function in intervals
            eta_left = Bracket[0]
            eta_right = Bracket[1]
            dlp_deta_left = BracketValues[0]
            dlp_deta_right = BracketValues[1]

            # Second derivative of log likelihood at eta = zero, using either of the two methods below:
            eta_zero = 0.0
            d2lp_deta2_zero_eta = LikelihoodEstimation.LogLikelihoodSecondDerivative(z,X,K,TraceEstimationUtilities,eta_zero)   # Method 1: directly from analytical equation
            # dlp_deta_zero_eta = LikelihoodEstimation.LogLikelihoodFirstDerivative(z,X,K,TraceEstimationUtilities,numpy.log10(eta_zero))
            # d2lp_deta2_zero_eta = (dlp_deta_lowest_eta - dlp_deta_zero_eta) / eta_lowest        # Method 2: usng forward differencing from first derivative

            # print('dL/deta   at eta = 0.0:\t %0.2f'%dlp_deta_zero_eta)
            print('dL/deta   at eta = %0.2e:\t %0.2f'%(eta_left,dlp_deta_left))
            print('dL/deta   at eta = %0.2e:\t %0.16f'%(eta_right,dlp_deta_right))
            print('d2L/deta2 at eta = 0.0:\t %0.2f'%d2lp_deta2_zero_eta)

            # No sign change. Can not find a root
            if (dlp_deta_left > 0) and (dlp_deta_right > 0):
                if d2lp_deta2_zero_eta > 0:
                    eta = 0.0
                else:
                    eta = numpy.inf

            elif (dlp_deta_left < 0) and (dlp_deta_right < 0):
                if d2lp_deta2_zero_eta < 0:
                    eta = 0.0
                else:
                    eta = numpy.inf

            # Find sigma and sigma0
            if eta == 0:
                sigma0 = 0
                sigma = FindOptimalSigma(z,X,K,eta)
                Success = True
            elif eta == numpy.inf:
                sigma = 0
                sigma0 = FindOptimalSigma0(z,X)
                Success = True
            else:
                raise ValueError('eta must be zero or inf at this point.')

        # Output distionary
        Results =  \
        {
                'sigma': sigma,
                'sigma0' : sigma0,
                'eta': eta,
                'Success': Success
        }
        
        return Results

    # ===================
    # Plot Log Likelihood
    # ===================

    @staticmethod
    def PlotLogLikelihood(z,X,K,TraceEstimationUtilities):
        """
        Plots log likelihood versus sigma,eta parameters
        """
        
        eta = numpy.logspace(-3,3,20)
        sigma = numpy.logspace(-1,0,20)
        lp = numpy.zeros((eta.size,sigma.size))
        for i in range(eta.size):
            for j in range(sigma.size):
                lp[i,j] = LikelihoodEstimation.LogLikelihoodWithSigmaEta(z,X,K,TraceEstimationUtilities,False,[sigma[j],eta[i]])

        [sigma_mesh,eta_mesh] = numpy.meshgrid(sigma,eta)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # p = ax.plot_surface(sigma_mesh,eta_mesh,lp,linewidth=0,antialiased=False)
        p = ax.plot_surface(numpy.log10(sigma_mesh),numpy.log10(eta_mesh),lp,linewidth=0,antialiased=False)
        fig.colorbar(p,ax=ax)
        # ax.xaxis.set_scale('log')
        # ax.yaxis.set_scale('log')
        # plt.yscale('log')
        ax.set_xlabel(r'$\sigma$')
        ax.set_ylabel(r'$\eta$')
        ax.set_title('Log Likelihood function')
        plt.show

    # ==================================
    # Compute Bounds of First Derivative
    # ==================================

    @staticmethod
    def ComputeBoundsOfFirstDerivative(X,K,eta):
        """
        Upper and lower bound.
        """

        n,m = X.shape
        Eigenvalue_smallest = scipy.linalg.eigh(K,eigvals_only=True,check_finite=False,subset_by_index=[0,0])[0]
        Eigenvalue_largest = scipy.linalg.eigh(K,eigvals_only=True,check_finite=False,subset_by_index=[n-1,n-1])[0]
        # print('Eigenvalues of K:')
        # print(Eigenvalue_smallest)
        # print(Eigenvalue_largest)
        dlp_deta_UpperBound = 0.5*(n-m)*(1/(eta+Eigenvalue_smallest) - 1/(eta+Eigenvalue_largest))
        dlp_deta_LowerBound = -dlp_deta_UpperBound

        return dlp_deta_UpperBound,dlp_deta_LowerBound

    # =====================================
    # Compute Asymptote of First Derivative
    # =====================================

    @staticmethod
    def ComputeAsymptoteOfFirstDerivative(z,X,K,eta):
        """
        Computes first and second order asymptote to the first derivative of log marginal likelihood function.
        """

        # Initialize output
        Asymptote_1_order = numpy.empty(eta.size)
        Asymptote_2_order = numpy.empty(eta.size)

        n,m = X.shape
        I = numpy.eye(n)
        Im = numpy.eye(m)
        Q = X@numpy.linalg.inv(X.T@X)@X.T
        R = I - Q
        N  = K@R
        N2 = N@N
        N3 = N2@N
        N4 = N3@N

        mtrN = numpy.trace(N)/(n-m)
        mtrN2 = numpy.trace(N2)/(n-m)
        
        A0 = -R@(mtrN*I - N)
        A1 =  R@(mtrN*N + mtrN2*I - 2*N2)
        A2 = -R@(mtrN*N2 + mtrN2*N - 2*N3)
        A3 =  R@(mtrN2*N2 - N4)
        
        zRz = numpy.dot(z,numpy.dot(R,z))
        z_Rnorm = numpy.sqrt(zRz)
        zc = z / z_Rnorm
        
        a0 = numpy.dot(zc,numpy.dot(A0,zc))
        a1 = numpy.dot(zc,numpy.dot(A1,zc))
        a2 = numpy.dot(zc,numpy.dot(A2,zc))
        a3 = numpy.dot(zc,numpy.dot(A3,zc))
        
        for i in range(eta.size):

            Asymptote_1_order[i] = (-0.5*(n-m))*(a0 + a1/eta[i])/eta[i]**2 
            Asymptote_2_order[i] = (-0.5*(n-m))*(a0 + a1/eta[i] + a2/eta[i]**2 + a3/eta[i]**3)/eta[i]**2 

        # Roots
        Polynomial_1 = numpy.array([a0,a1])
        Polynomial_2 = numpy.array([a0,a1,a2,a3])

        Roots_1 = numpy.roots(Polynomial_1)
        Roots_2 = numpy.roots(Polynomial_2)

        # Remove complex roots
        Roots_2 = numpy.sort(numpy.real(Roots_2[numpy.abs(numpy.imag(Roots_2)) < 1e-10]))

        print('Asymptote roots:')
        print(Roots_1)
        print(Roots_2)

        return Asymptote_1_order,Asymptote_2_order,Roots_1,Roots_2

    # ====================================
    # Plot Log Likelihood First Derivative
    # ====================================

    @staticmethod
    def PlotLogLikelihoodFirstDerivative(z,X,K,TraceEstimationUtilities,Optimal_eta):
        """
        Plots the derivative of log likelihood as a function of eta.
        Also it shows where the optimal eta is, which is the location
        where the derivative is zero.
        """

        print('Plot first derivative ...')

        if (Optimal_eta != 0) and (not numpy.isinf(Optimal_eta)):
            PlotOptimal_eta = True
        else:
            PlotOptimal_eta = False

        # Specify which portion of eta array be high resolution for plotting in the inset axes
        LogEtaStart = -3
        LogEtaEnd = 3

        if PlotOptimal_eta:
            LogEtaStartHighRes = numpy.floor(numpy.log10(Optimal_eta))
            LogEtaEndHighRes = LogEtaStartHighRes + 2

            # Arrays of low and high resolutions of eta
            eta_HighRes = numpy.logspace(LogEtaStartHighRes,LogEtaEndHighRes,100)
            eta_LowRes_left = numpy.logspace(LogEtaStart,LogEtaStartHighRes,50)
            eta_LowRes_right = numpy.logspace(LogEtaEndHighRes,LogEtaEnd,20)

            # array of eta as a mix of low and high res
            if LogEtaEndHighRes >= LogEtaEnd:
                eta = numpy.r_[eta_LowRes_left,eta_HighRes]
            else:
                eta = numpy.r_[eta_LowRes_left,eta_HighRes,eta_LowRes_right]

        else:
            eta = numpy.logspace(LogEtaStart,LogEtaEnd,100)

        # Compute derivative of L
        dlp_deta = numpy.zeros(eta.size)
        for i in range(eta.size):
            dlp_deta[i] = LikelihoodEstimation.LogLikelihoodFirstDerivative(z,X,K,TraceEstimationUtilities,numpy.log10(eta[i]))

        # Compute upper and lower bound of derivative
        dlp_deta_UpperBound,dlp_deta_LowerBound = LikelihoodEstimation.ComputeBoundsOfFirstDerivative(X,K,eta)

        # Compute asymptote of first derivative, using both first and second order approximation
        try:
            # eta_HighRes migh not be defined, depending on PlotOptimal_eta
            x = eta_HighRes
        except:
            x = numpy.logspace(1,LogEtaEnd,100)
        dlp_deta_Asymptote_1,dlp_deta_Asymptote_2,Roots_1,Roots_2 = LikelihoodEstimation.ComputeAsymptoteOfFirstDerivative(z,X,K,x)

        # Main plot
        fig,ax1 = plt.subplots()
        ax1.semilogx(eta,dlp_deta_UpperBound,'--',color='black',label='Upper bound')
        ax1.semilogx(eta,dlp_deta_LowerBound,'-.',color='black',label='Lower bound')
        ax1.semilogx(eta,dlp_deta,color='black',label='Exact')
        if PlotOptimal_eta:
            ax1.semilogx(Optimal_eta,0,'.',marker='o',markersize=4,color='black')

        # Min of plot limit
        # ax1.set_yticks(numpy.r_[numpy.arange(-120,1,40),20])
        MaxPlot = numpy.max(dlp_deta)
        MaxPlotLim = numpy.ceil(numpy.abs(MaxPlot)/10.0)*10.0*numpy.sign(MaxPlot)

        MinPlotLim1 = -100
        ax1.set_yticks(numpy.array([MinPlotLim1,0,MaxPlotLim]))
        ax1.set_ylim([MinPlotLim1,MaxPlotLim])
        ax1.set_xlim([eta[0],eta[-1]])
        ax1.set_xlabel(r'$\eta$')
        ax1.set_ylabel(r'$\mathrm{d} \ell_{\hat{\sigma}^2(\eta)}(\eta)/\mathrm{d} \eta$')
        ax1.set_title('Derivative of Log Marginal Likelihood Function')
        ax1.grid(True)
        # ax1.legend(loc='upper left',frameon=False)
        ax1.patch.set_facecolor('none')

        # Inset plot
        if PlotOptimal_eta:
            ax2 = plt.axes([0,0,1,1])
            # Manually set the position and relative size of the inset axes within ax1
            ip = InsetPosition(ax1,[0.43,0.39,0.5,0.5])
            ax2.set_axes_locator(ip)
            # Mark the region corresponding to the inset axes on ax1 and draw lines
            # in grey linking the two axes.

            # Avoid inset mark lines interset the inset axes itself by setting its anchor
            if LogEtaEnd > LogEtaEndHighRes:
                mark_inset(ax1, ax2,loc1=3,loc2=4,facecolor='none',edgecolor='0.5')
            else:
                mark_inset(ax1, ax2,loc1=3,loc2=1,facecolor='none',edgecolor='0.5')

            ax2.semilogx(eta,numpy.abs(dlp_deta_UpperBound),'--',color='black')
            ax2.semilogx(eta,numpy.abs(dlp_deta_LowerBound),'-.',color='black')
            ax2.semilogx(x,dlp_deta_Asymptote_1,label=r'$1^{\text{st}}$ order asymptote',color='chocolate')
            ax2.semilogx(x,dlp_deta_Asymptote_2,label=r'$2^{\text{nd}}$ order asymptote',color='olivedrab')
            ax2.semilogx(eta_HighRes,dlp_deta[eta_LowRes_left.size:eta_LowRes_left.size+eta_HighRes.size],color='black')
            ax2.semilogx(Optimal_eta,0,marker='o',markersize=6,linewidth=0,color='white',markerfacecolor='black',label=r'Exact root at $\hat{\eta}_{\phantom{2}} = 10^{%0.2f}$'%numpy.log10(Optimal_eta))
            ax2.semilogx(Roots_1[-1],0,marker='o',markersize=6,linewidth=0,color='white',markerfacecolor='chocolate',label=r'Approximated root at $\hat{\eta}_1 = 10^{%0.2f}$'%numpy.log10(Roots_1[-1]))
            ax2.semilogx(Roots_2[-1],0,marker='o',markersize=6,linewidth=0,color='white',markerfacecolor='olivedrab',label=r'Approximated root at $\hat{\eta}_2 = 10^{%0.2f}$'%numpy.log10(Roots_2[-1]))
            ax2.set_xlim([eta_HighRes[0],eta_HighRes[-1]])
            # plt.setp(ax2.get_yticklabels(),backgroundcolor='white')

            # Find suitable range for plot limits
            MinPlot = numpy.abs(numpy.min(dlp_deta))
            MinPlotBase = 10**numpy.floor(numpy.log10(numpy.abs(MinPlot)))
            MinPlotLim = numpy.ceil(MinPlot/MinPlotBase)*MinPlotBase
            ax2.set_ylim([-MinPlotLim,MinPlotLim])
            ax2.set_yticks([-numpy.abs(MinPlotLim),0,numpy.abs(MinPlotLim)])

            ax2.text(Optimal_eta*10**0.05,MinPlotLim*0.05,r'$\hat{\eta}$'%numpy.log10(Optimal_eta),horizontalalignment='left',verticalalignment='bottom',fontsize=10)
            ax2.text(Roots_1[-1]*10**0.05,MinPlotLim*0.05,r'$\hat{\eta}_1$'%numpy.log10(Optimal_eta),horizontalalignment='left',verticalalignment='bottom',fontsize=10)
            ax2.text(Roots_2[-1]*10**0.05,MinPlotLim*0.05,r'$\hat{\eta}_2$'%numpy.log10(Optimal_eta),horizontalalignment='left',verticalalignment='bottom',fontsize=10)
            # ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            ax2.grid(True,axis='y')
            ax2.set_facecolor('oldlace')
            plt.setp(ax2.get_xticklabels(),backgroundcolor='white')
            ax2.tick_params(axis='x',labelsize=10)
            ax2.tick_params(axis='y',labelsize=10)

            # ax2.set_yticklabels(ax2.get_yticks(),backgroundcolor='w')
            # ax2.tick_params(axis='y',which='major',pad=0)

        handles,labels = [],[]
        for ax in [ax1,ax2]:
            for h,l in zip(*ax.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)
        plt.legend(handles,labels,frameon=False,fontsize='small',loc='upper left',bbox_to_anchor=(1.2,1.04))

        # Save plots
        # plt.tight_layout()
        SaveDir = './doc/images/'
        SaveFilename = 'LogLikelihoodFirstDerivative'
        SaveFilename_PDF = SaveDir + SaveFilename + '.pdf'
        SaveFilename_SVG = SaveDir + SaveFilename + '.svg'
        # plt.savefig(SaveFilename_PDF,transparent=True,bbox_inches='tight')
        plt.savefig(SaveFilename_PDF,bbox_inches='tight')
        plt.savefig(SaveFilename_SVG,bbox_inches='tight')
        print('Plot saved to %s.'%(SaveFilename_PDF))
        print('Plot saved to %s.'%(SaveFilename_SVG))
        # plt.show()
