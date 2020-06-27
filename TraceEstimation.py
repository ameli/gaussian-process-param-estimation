# =======
# Imports
# =======

import sys
import numpy
import scipy
from scipy import optimize
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg
import sksparse
from sksparse.cholmod import cholesky
from scipy.optimize import minimize
import scipy.interpolate
from functools import partial

# Classes, Files
from PlotSettings import *
import LinearAlgebra

# =============
# Linear Solver
# =============

def LinearSolver(A,b):

    if scipy.sparse.isspmatrix(A):

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
    else:
        # Dense matrix
        x = scipy.linalg.solve(A,b,sym_pos=True)

        return x

# ======================
# Trace Estimation Class
# ======================

class TraceEstimation():

    # ========================
    # Compute Trace Of Inverse
    # ========================

    @staticmethod
    def ComputeTraceOfInverse(A):
        """
        Computes trace of Ainv using Cholesky factorization.

        Ainv means inv(A). We want to find trace(Ainv).
        L is the Cholesky factorization of A.
        Linv is the inverse of L.

        Trace(Ainv) = Frobenius squared norm of Linv.
        """

        # ---------------
        # Cholesky Method
        # ---------------

        def CholeksyMethod(A):
            """
            Note: This function does not produce correct results when A is sparse.
            It seems sksparse.cholmod has a problem.
            When A = K, it produces correct result, But when A = Kn = K + eta I, its result is
            differen than Hurchinson, Lanczos method. Also its result becomes correct when A is converted
            to dense matrix, and when we do not use skspase.cholmod anymore.
            """

            # Matrix size limit to switch direct to indirect algorthm
            UseInverseMatrix = False  # SETTING
            # UseInverseMatrix = True  # SETTING

            # Determine to use Sparse
            UseSparse = False
            if scipy.sparse.isspmatrix(A):
                UseSparse = True

            # Cholesky factorization
            if UseSparse:
                # L = sksparse.cholmod.cholesky(A,mode='supernodal')
                L = sksparse.cholmod.cholesky(A)
            else:
                L = scipy.linalg.cholesky(A,lower=True)

            # Find Frobenius norm of the inverse of L. If matrix size is small, compute inverse directly
            if UseInverseMatrix == True:

                # Direct method. Take inverse of L, then compute its Frobenius norm.
                if UseSparse:

                    raise ValueError('Do not use sksparse.cholmod.inv, as it computes LDLt decomposition and the computed trace becomes incorrect. Either set UseInverseMatrix to False when using sparse matrices, or use Hutchinson or Lanczos method.')

                    # Note: the L.inv() uses LDLt decomposition, not LLt, which then the compueted Trace becomes incorrect.
                    Linv = L.inv()
                    Trace = scipy.sparse.linalg.norm(Linv,ord='fro')**2
                else:
                    Linv = scipy.linalg.inv(L)
                    Trace = numpy.linalg.norm(Linv,ord='fro')**2

            else:

                # Instead of finding L inverse, and then its norm, we directly find norm
                Norm2 = 0
                n = A.shape[0]

                # Solve a linear system that finds each of the columns of L inverse
                for i in range(n):

                    # Handle sparse matrices
                    if UseSparse:

                        # e is a zero vector with its i-th element is one
                        e = scipy.sparse.lil_matrix((n,1),dtype=float)
                        e[i] = 1.0

                        # x is the solution of L x = e. Thus, x is the i-th column of L inverse. Also, LDL SHOULD be disabled.
                        x = L.solve_L(e.tocsc(),use_LDLt_decomposition=False)

                        # Append to the Frobenius norm of L inverse
                        Norm2 += numpy.sum(x.toarray()**2)

                    else:

                        # e is a zero vector with its i-th element is one
                        e = numpy.zeros(n)
                        e[i] = 1.0

                        # x is the solution of L x = e. Thus, x is the i-th column of L inverse
                        x = scipy.linalg.solve_triangular(L,e,lower=True)

                        # Append to the Frobenius norm of L inverse
                        Norm2 += numpy.sum(x**2)

                Trace = Norm2

            return Trace

        # -----------------
        # Hutchinson Method
        # -----------------

        def HutchinsonMethod(A):
            """
            The random vectors have Radamacher distribution. Compared to the Gaissuan
            distribution, the former distribution yields estimation of trace with lower
            variance.
            """

            # Number of trials
            m = 20            # SETTING
            # m = 30            # SETTING
            n = A.shape[0]

            # Create a random matrix with m random vectors with Radamacher distribution.
            E = numpy.sign(numpy.random.randn(n,m))

            # Orthonormalize random vectors
            Q,R = scipy.linalg.qr(E,mode='economic',overwrite_a=True,pivoting=False,check_finite=False)
            AinvQ = LinearSolver(A,Q)
            QtAinvQ = numpy.matmul(Q.T,AinvQ)

            # Trace
            Trace = n*numpy.mean(numpy.diag(QtAinvQ))

            return Trace

        # ------------------------------------
        # Stochastic Lanczos Quadrature Method
        # ------------------------------------

        def StochasticLanczosQuadratureMethod(A):
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
           
            # Number of iterations and number of Lanczos steps in each iteration
            NumIterations = 20   # SETTING
            LanczosDegree = 20   # SETTING
            # NumIterations = 30   # SETTING
            # LanczosDegree = 30   # SETTING
            UseLanczosTridiagonalization = False  # SETTING

            n = A.shape[0]
            TraceEstimates = numpy.zeros((NumIterations,))

            for i in range(NumIterations):

                # Radamacher random vector, consists of 1 and -1.
                w = numpy.sign(numpy.random.randn(n))

                if UseLanczosTridiagonalization:
                    # Lanczos recustive iteration to convert A to tridiagonal form T
                    # T = LinearAlgebra.LanczosTridiagonalization(A,w,LanczosDegree)
                    T = LinearAlgebra.LanczosTridiagonalization2(A,w,LanczosDegree)

                    # Spectral decomposition of T
                    Eigenvalues,Eigenvectors = numpy.linalg.eigh(T)

                    Theta = numpy.abs(Eigenvalues)
                    Tau2 = Eigenvectors[0,:]**2

                else:

                    # Use Golub-Kahn-Lanczos bidigonalization instead of Lanczos tridiagonalization
                    B = LinearAlgebra.GolubKahnLanczosBidiagonalization(A,w,LanczosDegree)
                    LeftEigenvectors,SingularValues,RightEigenvectorsTransposed = numpy.linalg.svd(B)
                    Theta = SingularValues    # Theta is just singular values, not singular values squared
                    Tau2 = RightEigenvectorsTransposed[:,0]**2

                # Here, f(theta) = 1/theta, since we compute trace of matrix inverse
                TraceEstimates[i] = numpy.sum(Tau2 * (1.0/Theta)) * n

            Trace = numpy.mean(TraceEstimates)

            return Trace

        # --------------

        # Trace = CholeksyMethod(A)
        # Trace = HutchinsonMethod(A)
        Trace = StochasticLanczosQuadratureMethod(A)

        return Trace

    # ==================================
    # Compute Trace Estimation Utilities
    # ==================================

    @staticmethod
    def ComputeTraceEstimationUtilities( \
            K, \
            UseEigenvaluesMethod,
            EstimationMethod, \
            FunctionType, \
            InterpolantPoints,
            ComputeAuxilliaryMethod=False):
        """
        Computes T0 and w.
        """

        print('Initialize trace estimation utilities ...')

        # Determine to use sparse
        UseSparse = False
        if scipy.sparse.isspmatrix(K):
            UseSparse = True

        # Initialize output depending on the chosen method
        EigenvaluesMethodUtilities = {}
        NonOrthogonalFunctionsMethodUtilities = {}
        OrthogonalFunctionsMethodUtilities = {}
        OrthogonalFunctionsMethodUtilities2 = {}
        RBFMethodUtilities = {}
        AuxilliaryEstimationMethodUtilities = {}
        RationalPolynomialMethodUtilities = {}

        # For large matrices, do not use eigenvalues method
        if UseEigenvaluesMethod == True:

            # Use Eigenvalues Method
            if UseSparse:

                n = K.shape[0]
                K_eigenvalues = numpy.zeros(n)

                # find 90% of eigenvalues and assume the rest are very close to zero.
                NumNoneZeroEig = int(n*0.9)
                K_eigenvalues[:NumNoneZeroEig] = scipy.sparse.linalg.eigsh(K,NumNoneZeroEig,which='LM',tol=1e-3,return_eigenvectors=False)

            else:
                K_eigenvalues = scipy.linalg.eigh(K,eigvals_only=True,check_finite=False)
            EigenvaluesMethodUtilities = \
            {
                'K_eigenvalues': K_eigenvalues
            }

        # Interpolation Method
        n = K.shape[0]
        T0 = TraceEstimation.ComputeTraceOfInverse(K)

        # Compute trace at interpolant points
        eta_i = numpy.array(InterpolantPoints)
        p = eta_i.size
        trace_eta_i = numpy.zeros(p)
        for i in range(p):
            if UseSparse:
                I = scipy.sparse.eye(n,format='csc')
            else:
                I = numpy.eye(n)

            Kn = K + eta_i[i] * I
            trace_eta_i[i] = TraceEstimation.ComputeTraceOfInverse(Kn)

        # Scale eta, if some of eta_i are greater than 1
        Scale_eta = 1.0
        if eta_i.size > 0:
            if numpy.max(eta_i) > 1.0:
                Scale_eta = numpy.max(eta_i)

        # Method 1: Use non-orthogonal basis functions
        if EstimationMethod == 'NonOrthogonalFunctionsMethod':
            # Form a linear system for weights w
            b = (n/trace_eta_i) - (n/T0) - eta_i
            A = numpy.zeros((p,p))
            for i in range(p):
                for j in range(p):
                    A[i,j] = TraceEstimation.TraceEstimationBasisFunctions(j,eta_i[i],EstimationMethod)

            # Test
            # print('Condition number: %f'%(numpy.linalg.cond(A)))

            w = numpy.linalg.solve(A,b)

            # Save results
            NonOrthogonalFunctionsMethodUtilities = \
            {
                'T0': T0,
                'n': n,
                'w': w,
                'p': p,
                'eta_i': eta_i,
                'trace_eta_i': trace_eta_i,
                'Scale_eta': Scale_eta
            }

        elif EstimationMethod == 'OrthogonalFunctionsMethod':

            # Method 2: Use orthogonal basis functions
            alpha,a = TraceEstimation.TraceEstimationOrthogonalBasisFunctionCoefficients(EstimationMethod)

            if alpha.size < eta_i.size:
                raise ValueError('Cannot regress order higher than %d. Decrease the number of interpolation points.'%(alpha.size))

            # Form a linear system Aw = b for weights w
            b = numpy.zeros(p+1)
            b[:-1] = (n/trace_eta_i) - (n/T0)
            b[-1] = 1
            A = numpy.zeros((p+1,p+1))
            for i in range(p):
                for j in range(p+1):
                    A[i,j] = TraceEstimation.TraceEstimationBasisFunctions(j,eta_i[i]/Scale_eta,EstimationMethod)
            A[-1,:] = alpha[:p+1]*a[:p+1,0]

            # Solve weights
            w = numpy.linalg.solve(A,b)

            # Test
            # print('Condition number: %f'%(numpy.linalg.cond(A)))

            # Results as dictionary
            OrthogonalFunctionsMethodUtilities = \
            {
                'T0': T0,
                'w': w,
                'n': n,
                'eta_i': eta_i,
                'trace_eta_i': trace_eta_i,
                'p': p,
                'alpha': alpha,
                'a': a,
                'Scale_eta': Scale_eta
            }

        elif EstimationMethod == 'OrthogonalFunctionsMethod2':

            # Method 2: Use orthogonal basis functions
            alpha,a = TraceEstimation.TraceEstimationOrthogonalBasisFunctionCoefficients(EstimationMethod)

            if alpha.size < eta_i.size:
                raise ValueError('Cannot regress order higher than %d. Decrease the number of interpolation points.'%(alpha.size))

            # Form a linear system Aw = b for weights w
            b = (n/trace_eta_i) - (n/T0) - eta_i
            A = numpy.zeros((p,p))
            for i in range(p):
                for j in range(p):
                    A[i,j] = TraceEstimation.TraceEstimationBasisFunctions(j,eta_i[i]/Scale_eta,EstimationMethod)

            # Solve weights
            w = numpy.linalg.solve(A,b)
            # Lambda = 1e1   # Regularization parameter  # SETTING
            # A2 = A.T.dot(A) + Lambda * numpy.eye(A.shape[0])
            # b2 = A.T.dot(b)
            # w = numpy.linalg.solve(A2,b2)

            # Test
            print('Condition number: %f'%(numpy.linalg.cond(A)))

            # Results as dictionary
            OrthogonalFunctionsMethodUtilities2 = \
            {
                'T0': T0,
                'w': w,
                'n': n,
                'eta_i': eta_i,
                'trace_eta_i': trace_eta_i,
                'p': p,
                'alpha': alpha,
                'a': a,
                'Scale_eta': Scale_eta
            }

        elif EstimationMethod == 'RBFMethod':

            # Take logarithm of eta_i
            xi = numpy.log10(eta_i)
            tau_i = trace_eta_i/n
            tau_0 = T0/n

            if xi.size > 1:
                dxi = numpy.mean(numpy.diff(xi))
            else:
                dxi = 1

            # Function Type
            if FunctionType == 1:
                # Ascending function
                yi = 1/tau_i - (1/tau_0 + eta_i)
            elif FunctionType == 2:
                # Bell shape, going to zero at boundaries
                yi = (1/tau_i)/(1/tau_0 + eta_i) - 1
            elif FunctionType == 3:
                # Bell shape, going to zero at boundaries
                yi = 1 - (tau_i)*(1/tau_0 + eta_i)
            else:
                raise ValueError('Invalid function type.')

            # extend boundaries to zero
            LowLogThreshold = -4.5   # SETTING
            HighLogThreshold = 3.5   # SETTING
            NumExtend = 3            # SETTING
           
            # Avoid thresholds to cross interval of data
            if LowLogThreshold >= numpy.min(xi):
                LowLogThreshold = numpy.min(xi) - dxi
            if HighLogThreshold <= numpy.max(xi):
                HighLogThreshold = numpy.max(xi) + dxi

            # Extend interval of data by adding zeros to left and right
            if (FunctionType == 2) or (FunctionType == 3):
                ExtendLeft_x = numpy.linspace(LowLogThreshold-dxi,LowLogThreshold,NumExtend)
                ExtendRight_x = numpy.linspace(HighLogThreshold,HighLogThreshold+dxi,NumExtend)
                Extend_y = numpy.zeros(NumExtend)
                xi = numpy.r_[ExtendLeft_x,xi,ExtendRight_x]
                yi = numpy.r_[Extend_y,yi,Extend_y]

            # Radial Basis Function
            if FunctionType == 1:
                # These interpolation methods are good for the ascending shaped function
                RBF = scipy.interpolate.CubicSpline(xi,yi,bc_type=((1,0.0),(2,0)),extrapolate=True)       # best for ascneing function
                # RBF = scipy.interpolate.PchipInterpolator(xi,yi,extrapolate=True)                       # good
                # RBF = scipy.interpolate.UnivariateSpline(xi,yi,k=3,s=0.0)                               # bad
            elif (FunctionType == 2) or (FunctionType == 3):
                # These interpolation methods are good for the Bell shaped function
                RBF = scipy.interpolate.Rbf(xi,yi,function='gaussian',epsilon=dxi)                    # Best for function type 2,3,4
                # RBF = scipy.interpolate.Rbf(xi,yi,function='inverse',epsilon=dxi)
                # RBF = scipy.interpolate.CubicSpline(xi,yi,bc_type=((1,0.0),(1,0.0)),extrapolate=True)

            # Plot interpolation with RBF
            PlotFlag = False
            if PlotFlag:
                eta = numpy.logspace(LowLogThreshold-dxi,HighLogThreshold+dxi,100)
                x = numpy.log10(eta)
                y = RBF(x)
                fig,ax = plt.subplots()
                ax.plot(x,y)
                ax.plot(xi,yi,'o')
                ax.grid(True)
                ax.set_xlim([LowLogThreshold-dxi,HighLogThreshold+dxi])
                # ax.set_ylim(-0.01,0.18)
                plt.show()

            # Save results
            RBFMethodUtilities = \
            {
                'T0': T0,
                'n': n,
                'p': p,
                'RBF': RBF,
                'eta_i': eta_i,
                'trace_eta_i': trace_eta_i,
                'LowLogThreshold': LowLogThreshold,
                'HighLogThreshold': HighLogThreshold,
                'FunctionType': FunctionType
            }

        elif EstimationMethod == 'RationalPolynomialMethod':

            tau0 = T0/n
            tau_i = trace_eta_i / n

            # ----------------
            # Rational Poly 12
            # ----------------

            def RationalPoly12(eta_i,tau_i,tau0):
                """
                Rational polynomial of order 1 over 2
                """

                # Matrix of coefficients
                A = numpy.array([
                    [eta_i[0],1-tau0/tau_i[0]],
                    [eta_i[1],1-tau0/tau_i[1]]])

                # Vector of right hand side
                c = numpy.array([
                    eta_i[0]/tau_i[0]-eta_i[0]**2,
                    eta_i[1]/tau_i[1]-eta_i[1]**2])

                # Condition number
                print('Condition number: %0.2e'%(numpy.linalg.cond(A)))

                # Solve with least square. NOTE: do not solve with numpy.linalg.solve directly.
                b = numpy.linalg.solve(A,c)
                b0 = b[1]
                b1 = b[0]
                a0 = b0*tau0

                # Output
                Numerator = [1,a0]
                Denominator = [1,b1,b0]

                # Check poles
                Poles = numpy.roots(Denominator)
                if numpy.any(Poles > 0):
                    print('Denominator poles:')
                    print(Poles)
                    raise ValueError('RationalPolynomial has positive poles.')

                return Numerator,Denominator

            # ----------------
            # Rational Poly 23
            # ----------------

            def RationalPoly23(eta_i,tau_i,tau0):
                """
                Rational polynomial of order 2 over 3
                """

                # Matrix of coefficients
                A = numpy.array([
                    [eta_i[0]**2,eta_i[0],1-tau0/tau_i[0],-eta_i[0]/tau_i[0]],
                    [eta_i[1]**2,eta_i[1],1-tau0/tau_i[1],-eta_i[1]/tau_i[1]],
                    [eta_i[2]**2,eta_i[2],1-tau0/tau_i[2],-eta_i[2]/tau_i[2]],
                    [eta_i[3]**2,eta_i[3],1-tau0/tau_i[3],-eta_i[3]/tau_i[3]]])

                # Vector of right hand side
                c = numpy.array([
                    eta_i[0]**2/tau_i[0]-eta_i[0]**3,
                    eta_i[1]**2/tau_i[1]-eta_i[1]**3,
                    eta_i[2]**2/tau_i[2]-eta_i[2]**3,
                    eta_i[3]**2/tau_i[3]-eta_i[3]**3])

                # Condition number
                print('Condition number: %0.2e'%(numpy.linalg.cond(A)))

                # Solve with least square. NOTE: do not solve with numpy.linalg.solve directly.
                b = numpy.linalg.solve(A,c)
                b2 = b[0]
                b1 = b[1]
                b0 = b[2]
                a1 = b[3]
                a0 = b0*tau0

                # Output
                Numerator = [1,a1,a0]
                Denominator = [1,b2,b1,b0]

                # Check poles
                Poles = numpy.roots(Denominator)
                if numpy.any(Poles > 0):
                    print('Denominator poles:')
                    print(Poles)
                    raise ValueError('RationalPolynomial has positive poles.')

                return Numerator,Denominator

            # ------------------

            # Coefficients of a linear system
            if p == 2:
                Numerator,Denominator = RationalPoly12(eta_i,tau_i,tau0)

            elif p == 4:
                Numerator,Denominator = RationalPoly23(eta_i,tau_i,tau0)

            else:
                raise ValueError('In RationalPolynomial method, the number of interpolant points, p, should be 2 or 4.')

            print('Numerator:')
            print(Numerator)
            print('Denominator:')
            print(Denominator)

            RationalPolynomialMethodUtilities = \
            {
                'n': n,
                'p': p,
                'T0': T0,
                'Numerator': Numerator,
                'Denominator': Denominator
            }

        else:
            raise ValueError('Method is invalid.')

        # Interpolant points for the auxilliary estimation method
        eta1 = 0.1    # SETTING
        if UseSparse:
            I = scipy.sparse.eye(n,format='csc')
        else:
            I = numpy.eye(n)
        Kn = K + eta1*I
        if ComputeAuxilliaryMethod:
            T1 = TraceEstimation.ComputeTraceOfInverse(Kn)
        else:
            T1 = None
        AuxilliaryEstimationMethodUtilities = \
        {
            'eta1': eta1,
            'T1': T1,
            'T0': T0,
            'n': n,
            'p': p
        }

        # Output
        TraceEstimationUtilities = \
        {
            'UseEigenvaluesMethod': UseEigenvaluesMethod,
            'EstimationMethod': EstimationMethod,
            'EigenvaluesMethodUtilities': EigenvaluesMethodUtilities,
            'NonOrthogonalFunctionsMethodUtilities': NonOrthogonalFunctionsMethodUtilities,
            'OrthogonalFunctionsMethodUtilities': OrthogonalFunctionsMethodUtilities,
            'OrthogonalFunctionsMethodUtilities2': OrthogonalFunctionsMethodUtilities2,
            'RBFMethodUtilities': RBFMethodUtilities,
            'AuxilliaryEstimationMethodUtilities': AuxilliaryEstimationMethodUtilities,
            'RationalPolynomialMethodUtilities': RationalPolynomialMethodUtilities
        }

        return TraceEstimationUtilities

    # ================================
    # Trace Estimation Basis Functions
    # ================================

    @staticmethod
    def TraceEstimationBasisFunctions(j,eta,EstimationMethod):
        """
        Functions phi_j(eta).
        The index j of the basis functions starts from 1.
        """

        # non-orthogonal basis function
        def phi(i,eta):
            return eta**(1.0/i)

        if EstimationMethod == 'NonOrthogonalFunctionsMethod':
            return phi(j+2,eta)

        elif EstimationMethod == 'OrthogonalFunctionsMethod':

            # Use Orthogonal basis functions
            alpha,a = TraceEstimation.TraceEstimationOrthogonalBasisFunctionCoefficients(EstimationMethod)

            phi_perp = 0
            for i in range(a.shape[1]):
                phi_perp += alpha[j]*a[j,i]*phi(i+1,eta)

            return phi_perp

        elif EstimationMethod == 'OrthogonalFunctionsMethod2':

            # Use Orthogonal basis functions
            alpha,a = TraceEstimation.TraceEstimationOrthogonalBasisFunctionCoefficients(EstimationMethod)

            phi_perp = 0
            for i in range(a.shape[1]):
                phi_perp += alpha[j]*a[j,i]*phi(i+2,eta)

            return phi_perp

        else:
            raise ValueError('Method is invalid.')

    # =======================================================
    # Trace Estimation Orthogonal Basis Function Coefficients
    # =======================================================

    @staticmethod
    def TraceEstimationOrthogonalBasisFunctionCoefficients(EstimationMethod):
        """
        Coefficients alpha and a.
        To genrate these coefficients, see GenerateOrthogonalFunctions.py
        """

        p = 9
        a = numpy.zeros((p,p),dtype=float)

        if EstimationMethod == 'OrthogonalFunctionsMethod':
            alpha = numpy.array([
                +numpy.sqrt(2/1),
                -numpy.sqrt(2/2),
                +numpy.sqrt(2/3),
                -numpy.sqrt(2/4),
                +numpy.sqrt(2/5),
                -numpy.sqrt(2/6),
                +numpy.sqrt(2/7),
                -numpy.sqrt(2/8),
                +numpy.sqrt(2/9)])

            a[0,:1] = numpy.array([1])
            a[1,:2] = numpy.array([4, -3])
            a[2,:3] = numpy.array([9, -18, 10])
            a[3,:4] = numpy.array([16, -60, 80, -35])
            a[4,:5] = numpy.array([25, -150, 350, -350, 126])
            a[5,:6] = numpy.array([36, -315, 1120, -1890, 1512, -462])
            a[6,:7] = numpy.array([49, -588, 2940, -7350, 9702, -6468, 1716])
            a[7,:8] = numpy.array([64, -1008, 6720, -23100, 44352, -48048, 27456, -6435])
            a[8,:9] = numpy.array([81, -1620, 13860, -62370, 162162, -252252, 231660, -115830, 24310])

        elif EstimationMethod == 'OrthogonalFunctionsMethod2':
            alpha = numpy.array([
                +numpy.sqrt(2/2),
                -numpy.sqrt(2/3),
                +numpy.sqrt(2/4),
                -numpy.sqrt(2/5),
                +numpy.sqrt(2/6),
                -numpy.sqrt(2/7),
                +numpy.sqrt(2/8),
                -numpy.sqrt(2/9),
                +numpy.sqrt(2/10)])

            a[0,:1] = numpy.array([1])
            a[1,:2] = numpy.array([6, -5])
            a[2,:3] = numpy.array([20, -40, 21])
            a[3,:4] = numpy.array([50, -175, 210, -84])
            a[4,:5] = numpy.array([105, -560, 1134, -1008, 330])
            a[5,:6] = numpy.array([196, -1470, 4410, -6468, 4620, -1287])
            a[6,:7] = numpy.array([336, -3360, 13860, -29568, 34320, -20592, 5005])
            a[7,:8] = numpy.array([540, -6930, 37422, -108108, 180180, -173745, 90090, -19448])
            a[8,:9] = numpy.array([825, -13200, 90090, -336336, 750750, -1029600, 850850, -388960, 75582])

        return alpha,a

    # ==============
    # Estimate Trace
    # ==============

    @staticmethod
    def EstimateTrace(TraceEstimationUtilities,eta):
        """
        Estimate trace using interpolation function with given parameters.
        """

        # Unpack parameters
        EstimationMethod = TraceEstimationUtilities['EstimationMethod']

        if EstimationMethod == 'NonOrthogonalFunctionsMethod':

            T0 = TraceEstimationUtilities['NonOrthogonalFunctionsMethodUtilities']['T0']
            n  = TraceEstimationUtilities['NonOrthogonalFunctionsMethodUtilities']['n']
            w  = TraceEstimationUtilities['NonOrthogonalFunctionsMethodUtilities']['w']
            p  = TraceEstimationUtilities['NonOrthogonalFunctionsMethodUtilities']['p']
            Scale_eta  = TraceEstimationUtilities['NonOrthogonalFunctionsMethodUtilities']['Scale_eta']

            S = 0.0
            for j in range(p):
                S += w[j] * TraceEstimation.TraceEstimationBasisFunctions(j,eta,EstimationMethod)

            T = n / (n/T0+S+eta)

            return T
                
        elif EstimationMethod == 'OrthogonalFunctionsMethod':
            T0 = TraceEstimationUtilities['OrthogonalFunctionsMethodUtilities']['T0']
            n  = TraceEstimationUtilities['OrthogonalFunctionsMethodUtilities']['n']
            w  = TraceEstimationUtilities['OrthogonalFunctionsMethodUtilities']['w']
            p  = TraceEstimationUtilities['OrthogonalFunctionsMethodUtilities']['p']
            Scale_eta  = TraceEstimationUtilities['OrthogonalFunctionsMethodUtilities']['Scale_eta']

            S = 0.0
            for j in range(w.size):
                S += w[j] * TraceEstimation.TraceEstimationBasisFunctions(j,eta/Scale_eta,EstimationMethod)

            T = n / (n/T0+S)

            return T

        elif EstimationMethod == 'OrthogonalFunctionsMethod2':
            T0 = TraceEstimationUtilities['OrthogonalFunctionsMethodUtilities2']['T0']
            n  = TraceEstimationUtilities['OrthogonalFunctionsMethodUtilities2']['n']
            w  = TraceEstimationUtilities['OrthogonalFunctionsMethodUtilities2']['w']
            p  = TraceEstimationUtilities['OrthogonalFunctionsMethodUtilities2']['p']
            Scale_eta  = TraceEstimationUtilities['OrthogonalFunctionsMethodUtilities2']['Scale_eta']

            S = 0.0
            for j in range(p):
                S += w[j] * TraceEstimation.TraceEstimationBasisFunctions(j,eta/Scale_eta,EstimationMethod)

            T = n / (n/T0+S+eta)

            return T

        elif EstimationMethod == 'RBFMethod':
            T0 = TraceEstimationUtilities['RBFMethodUtilities']['T0']
            n  = TraceEstimationUtilities['RBFMethodUtilities']['n']
            RBF = TraceEstimationUtilities['RBFMethodUtilities']['RBF']
            LowLogThreshold= TraceEstimationUtilities['RBFMethodUtilities']['LowLogThreshold']
            HighLogThreshold = TraceEstimationUtilities['RBFMethodUtilities']['HighLogThreshold']
            FunctionType = TraceEstimationUtilities['RBFMethodUtilities']['FunctionType']

            x = numpy.log10(eta)
            if x < LowLogThreshold or x > HighLogThreshold:
                y = 0
            else:
                y = RBF(x)

            tau_0 = T0 / n

            if FunctionType == 1:
                tau = 1/(y + 1/tau_0 + eta)
            elif FunctionType == 2:
                tau = 1/((y+1)*(1/tau_0+eta))
            elif FunctionType == 3:
                tau = (1-y)/(1/tau_0+eta)
            else:
                raise ValueError('Invalid function type.')
            T = n*tau
            
            return T

        elif EstimationMethod == 'RationalPolynomialMethod':
            
            n = TraceEstimationUtilities['RationalPolynomialMethodUtilities']['n']
            T0 = TraceEstimationUtilities['RationalPolynomialMethodUtilities']['T0']
            Numerator = TraceEstimationUtilities['RationalPolynomialMethodUtilities']['Numerator']
            Denominator = TraceEstimationUtilities['RationalPolynomialMethodUtilities']['Denominator']

            def RationalPoly(x,Numerator,Denominator):
                return numpy.polyval(Numerator,x) / numpy.polyval(Denominator,x)

            tau = RationalPoly(eta,Numerator,Denominator)
            T = tau*n

            return T

        else:
            raise ValueError('Method is invalid.')

    # ===================
    # Plot Trace Estimate
    # ===================

    @staticmethod
    def PlotTraceEstimate(TraceEstimationUtilitiesList,K):
        """
        Plots the curve of trace of Kn inverse versus eta.
        """

        # If not a list, embed the object into a list
        if not isinstance(TraceEstimationUtilitiesList,list):
            TraceEstimationUtilitiesList = [TraceEstimationUtilitiesList]

        # Determine to use sparse
        UseSparse = False
        if scipy.sparse.isspmatrix(K):
            UseSparse = True

        # Extract parameters
        T0   = TraceEstimationUtilitiesList[0]['AuxilliaryEstimationMethodUtilities']['T0']
        n    = TraceEstimationUtilitiesList[0]['AuxilliaryEstimationMethodUtilities']['n']
        T1   = TraceEstimationUtilitiesList[0]['AuxilliaryEstimationMethodUtilities']['T1']
        eta1 = TraceEstimationUtilitiesList[0]['AuxilliaryEstimationMethodUtilities']['eta1']

        NumberOfEstimates = len(TraceEstimationUtilitiesList)

        eta = numpy.logspace(-4,3,100)
        trace_upperbound = numpy.zeros(eta.size)
        trace_lowerbound = numpy.zeros(eta.size)
        trace_exact = numpy.zeros(eta.size)
        trace_estimate = numpy.zeros((NumberOfEstimates,eta.size))
        trace_estimate_alt = numpy.zeros(eta.size)

        for i in range(eta.size):
            trace_upperbound[i] = 1.0/(1.0/T0 + eta[i]/n)
            trace_lowerbound[i] = n/(1.0+eta[i])

            # Kn
            if UseSparse:
                I = scipy.sparse.eye(K.shape[0],format='csc')
            else:
                I = numpy.eye(K.shape[0])
            Kn = K + eta[i]*I
            trace_exact[i] = TraceEstimation.ComputeTraceOfInverse(Kn)
            trace_estimate_alt[i] = 1.0 / (numpy.sqrt((eta[i]/n)**2 + ((1.0/T1)**2 - (1.0/T0)**2 - (eta1/n)**2)*(eta[i]/eta1) + (1/T0)**2));

            for j in range(NumberOfEstimates):
                trace_estimate[j,i] = TraceEstimation.EstimateTrace(TraceEstimationUtilitiesList[j],eta[i])

        # Tau
        tau_upperbound = trace_upperbound / n
        tau_lowerbound = trace_lowerbound / n
        tau_exact = trace_exact / n
        tau_estimate = trace_estimate / n
        tau_estimate_alt = trace_estimate_alt / n

        # Plots trace
        fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(12,6))
        ax[0].loglog(eta,tau_exact,color='black',label='Exact')
        ax[0].loglog(eta,tau_upperbound,'--',color='black',label='Upper bound')
        ax[0].loglog(eta,tau_lowerbound,'-.',color='black',label='Lower bound')

        ColorsList =["#d62728",
                "#2ca02c",
                "#bcbd22",
                "#ff7f0e",
                "#1f77b4",
                "#9467bd",
                "#8c564b",
                "#17becf",
                "#7f7f7f",
                "#e377c2"]

        for j in reversed(range(NumberOfEstimates)):
            p = TraceEstimationUtilitiesList[j]['AuxilliaryEstimationMethodUtilities']['p']
            q = ax[0].loglog(eta,tau_estimate[j,:],label=r'Interpolation, $p=%d$'%(p),color=ColorsList[j])
            if j == 0:
                q[0].set_zorder(20)

        # ax[0].loglog(eta,tau_estimate_alt,label=r'Alt. estimation',zorder=-20)

        ax[0].set_xlim([eta[0],eta[-1]])
        ax[0].set_ylim([1e-3,1e1])
        ax[0].set_xlabel(r'$\eta$')
        ax[0].set_ylabel(r'$\tau(\eta)$')
        ax[0].set_title(r'(a) Exact, interpolation, and bounds of $\tau(\eta)$')
        ax[0].grid(True)
        ax[0].legend(fontsize='x-small',loc='upper right')

        # Inset plot
        ax2 = plt.axes([0,0,1,1])
        # Manually set the position and relative size of the inset axes within ax1
        ip = InsetPosition(ax[0],[0.14,0.1,0.5,0.4])
        ax2.set_axes_locator(ip)
        # Mark the region corresponding to the inset axes on ax1 and draw lines
        # in grey linking the two axes.

        # Avoid inset mark lines interset the inset axes itself by setting its anchor
        InsetColor = 'oldlace'
        mark_inset(ax[0],ax2,loc1=1,loc2=2,facecolor=InsetColor,edgecolor='0.5')
        ax2.semilogx(eta,tau_exact,color='black',label='Exact')
        ax2.semilogx(eta,tau_upperbound,'--',color='black',label='Upper bound')
        for j in reversed(range(NumberOfEstimates)):
            ax2.semilogx(eta,tau_estimate[j,:],color=ColorsList[j])
        # ax2.semilogx(eta,tau_estimate_alt,label=r'Alt. estimation',zorder=-1)
        ax2.set_xlim([1e-2,1e-1])
        # ax2.set_xlim([0.35,0.4])
        # ax2.set_ylim(2.5,4)
        ax2.set_ylim(4,6)
        # ax2.set_ylim(1.4,1.6)
        # ax2.set_yticks([2.5,3,3.5,4])
        ax2.set_yticks([4,5,6])
        ax2.xaxis.set_minor_formatter(NullFormatter())
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.set_facecolor(InsetColor)
        plt.setp(ax2.get_yticklabels(),backgroundcolor='white')

        # ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        # ax2.grid(True,axis='y')

        # Plot errors
        # ax[1].semilogx(eta,tau_upperbound-tau_exact,'--',color='black',label='Upper bound')  # Absolute error
        ax[1].semilogx(eta,100*(tau_upperbound/tau_exact-1),'--',color='black',label='Upper bound',zorder=15)  # Relative error
        for j in reversed(range(NumberOfEstimates)):
            p = TraceEstimationUtilitiesList[j]['AuxilliaryEstimationMethodUtilities']['p']
            # q = ax[1].semilogx(eta,tau_estimate[j,:]-tau_exact,label=r'Estimation, $p=%d$'%(p),color=ColorsList[j])  # Absolute error
            q = ax[1].semilogx(eta,100*(tau_estimate[j,:]/tau_exact-1),label=r'Interpolation, $p=%d$'%(p),color=ColorsList[j])       # Relative error
            if j == 0:
                q[0].set_zorder(20)
        # ax[1].semilogx(eta,tau_estimate_alt-tau_exact,label=r'Alt. estimation',zorder=-20)   # Absolute error
        # ax[1].semilogx(eta,tau_estimate_alt/tau_exact-1,label=r'Alt. estimation',zorder=-20)   # Relative error
        ax[1].set_xlim([eta[0],eta[-1]])
        ax[1].set_yticks(numpy.arange(-0.03,0.13,0.03)*100)
        ax[1].set_ylim([-3,12])
        ax[1].set_xlabel(r'$\eta$')
        ax[1].set_ylabel(r'$\tau_{\mathrm{approx}}(\eta)/\tau_{\mathrm{exact}}(\eta) - 1$')
        ax[1].set_title(r'(b) Relative error of interpolation of $\tau(\eta)$')
        ax[1].grid(True)
        ax[1].legend(fontsize='x-small')

        ax[1].yaxis.set_major_formatter(PercentFormatter(decimals=0))

        # Save plots
        plt.tight_layout()
        SaveDir = './doc/images/'
        SaveFilename = 'EstimateTrace'
        SaveFilename_PDF = SaveDir + SaveFilename + '.pdf'
        SaveFilename_SVG = SaveDir + SaveFilename + '.svg'
        # plt.savefig(SaveFilename_PDF,transparent=True,bbox_inches='tight')
        plt.savefig(SaveFilename_PDF,bbox_inches='tight')
        plt.savefig(SaveFilename_SVG,bbox_inches='tight')
        print('Plot saved to %s.'%(SaveFilename_PDF))
        print('Plot saved to %s.'%(SaveFilename_SVG))

        # plt.show()
