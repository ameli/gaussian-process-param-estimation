# =======
# Imports
# =======

import numpy
import scipy
from scipy import special
from scipy import sparse
import matplotlib.pyplot as plt
# import multiprocessing
import psutil
import logging

try:
    import ray
    RayInstalled = True
except:
    RayInstalled = False

# =============
# Generate Data
# =============

def GenerateData(NumPoints,NoiseMagnitude,GridOfPoints):
    """
    Generates three (n*1) column vectors x,y, and z, where
    x and y are coordinates of data z.
    """

    print('Generate data ...')

    # Grid of points
    if GridOfPoints == True:
        x_axis = numpy.linspace(0,1,NumPoints)
        y_axis = numpy.linspace(0,1,NumPoints)
        x_mesh,y_mesh = numpy.meshgrid(x_axis,y_axis)

        # Column vectors of x and y of data
        x = x_mesh.ravel()
        y = y_mesh.ravel()
    else:
        # Randomized points in a square area
        x = numpy.random.rand(NumPoints)
        y = numpy.random.rand(NumPoints)

    # Data
    z = numpy.sin(x*numpy.pi) + numpy.sin(y*numpy.pi)
    n = z.size

    # Add noise
    numpy.random.seed(31)
    z += NoiseMagnitude*numpy.random.randn(n)

    # Plot data
    PlotFlag = False
    if PlotFlag:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        p = ax.plot_surface(x_mesh,y_mesh,z.reshape(NumPoints,NumPointsAlongAxis),linewidth=0,antialiased=False)
        fig.colorbar(p,ax=ax)
        plt.show()

    return x,y,z

# ==================
# Correlation Kernel
# ==================

def CorrelationKernel(Distance,DecorrelationScale,nu):
    """
    Matern class correlation function.
    If nu is half integer, the Matern function has exponential form. Otherwise it
    is represented by Bessel function.
    """

    # scaled distance
    ScaledDistance = Distance / DecorrelationScale

    if nu == 0.5:
        Correlation = numpy.exp(-ScaledDistance)

    elif nu == 1.5:
        Correlation = (1.0 + numpy.sqrt(3.0)*ScaledDistance) * numpy.exp(-numpy.sqrt(3.0)*ScaledDistance)

    elif nu == 2.5:
        Correlation = (1.0 + numpy.sqrt(5.0)*ScaledDistance + (5.0/3.0)*(ScaledDistance**2)) * numpy.exp(-numpy.sqrt(5.0)*ScaledDistance)

    elif nu < 100:
        
        # Change zero elements of ScaledDistance to a dummy number, to avoid multiplication of zero by Inf in Bessel function below
        ScaledDistance[0] = 1
        Correlation = ((2.0**(1.0-nu))/scipy.special.gamma(nu)) * ((numpy.sqrt(2.0*nu) * ScaledDistance)**nu) * scipy.special.kv(nu,numpy.sqrt(2.0*nu)*ScaledDistance)

        # Set diagonals of correlation to one, since we altered the diagonals of ScaledDistance
        Correlation[0] = 1

        if numpy.any(numpy.isnan(Correlation)):
            raise ValueError('Correlation has nan element. nu: %f, DecorelationScale: %f'%(nu,DecorrelationScale))
        if numpy.any(numpy.isinf(Correlation)):
            raise ValueError('Correlation has inf element. nu: %f, DecorelationScale: %f'%(nu,DecorrelationScale))

    else:
        # For nu > 100, assume nu is Inf. In this case, Matern function approaches Gaussian kernel
        Correlation = numpy.exp(-0.5*ScaledDistance**2)

    return Correlation

# =================================
# Compute Correlation For A Process
# =================================

# if RayInstalled:
# @ray.remote
def ComputeCorrelationForAProcess(DecorrelationScale,nu,KernelThreshold,x,y,UseSparse,NumCPUs,StartIndex):
    """
    Computes correlation at the ColumnIndex-th column and row of K.
    K is updated inplace.
    This function is used as a partial function for parallel processing.

    If StartIndex is none, it fills all columns of correlation matrix K.
    If StartIndex is not none, it fills only a sub-rang of columns of K.
    """

    n = x.size

    if UseSparse:
        K = scipy.sparse.lil_matrix((n,n))
    else:
        K = numpy.zeros((n,n),dtype=float)

    # Range of filling columns of correlation
    if StartIndex is None:
        Range = range(n)
    else:
        Range = range(StartIndex,n,NumCPUs)

    # Fill K only at each NumCPU columns starting from StartIndex
    for i in Range:

        # Euclidean distance of points
        Distance = numpy.sqrt((x[i:]-x[i])**2 + (y[i:] - y[i])**2)
        Correlation = CorrelationKernel(Distance,DecorrelationScale,nu)

        # Sparsify
        if UseSparse:
            Correlation[Correlation < KernelThreshold] = 0

        # Diagonal element
        K[i,i] = Correlation[0] * 0.5

        # Upper-right elements
        if i < n-1:
            K[i,i+1:] = Correlation[1:]

    if UseSparse:
        return K.tocsc()
    else:
        return K

# ===========================
# Generate Correlation Matrix
# ===========================

def GenerateCorrelationMatrix(x,y,z,DecorrelationScale,nu,UseSparse):
    """
    Generates correlation matrix K and the eigenvalues of correlation matrix
    """

    print('Generate correlation matrix ...')

    # Distance matrix
    n = z.size

    # If this threshold is large, it causes (1) K to not be positive definite, (2) trace(K_inv+eta*I) to oscillate
    KernelThreshold = 0.03    # SETTING

    # Check if the thresholding is not too much to avoid the correlation matrix becomes identity. Each point should have at least one neighbor point in correlation matrix.
    if UseSparse:
        # Compute Adjacency
        NumPointsAlongAxis = numpy.rint(numpy.sqrt(n))
        GridSize = 1.0 / (NumPointsAlongAxis - 1.0)
        KernelLength = -DecorrelationScale*numpy.log(KernelThreshold)
        Adjacency = KernelLength / GridSize

        # If Adjacency is less that one, the correlation matrix becomes identity since no point will be adjacet to other in the correlation matrix.
        if Adjacency < 1.0:
            raise ValueError('Adjacency: %0.2f. Correlation matrix will become identity since Kernel length is less that grid size. To increase adjacency, consider decreasing KernelThreshold or increase DecorrelationScale.'%(Adjacency))

    # RunInParallel = True  # SETTING
    RunInParallel = False  # SETTING

    # Disable parallel processing if ray is not installed
    if not RayInstalled:
        RunInParallel = False

    # If matrice are sparse, it is better to generate columns of correlation in parallel
    if (RunInParallel == False) and (UseSparse == True):
        raise ValueError('If matrices are sparse, it is better to generate columns of correlation matrix in parallel. Set "RunInParallel" to True.')

    if RunInParallel:

        try:
            # Get number of cpus
            NumCPUs = psutil.cpu_count()

            # Parallelization with ray
            ray.init(num_cpus=NumCPUs,logging_level=logging.FATAL)

            # Parallel section with ray. This just creates process Ids. It does not do computation
            Process_Ids = [ComputeCorrelationForAProcess.remote(DecorrelationScale,nu,KernelThreshold,x,y,UseSparse,NumCPUs,StartIndex) for StartIndex in range(NumCPUs)]

            # Do the parallel computations
            K_List = ray.get(Process_Ids)

            # Initialize an empty correlation
            if UseSparse:
                K = scipy.sparse.csc_matrix((n,n))
            else:
                K = numpy.zeros((n,n),dtype=float)

            # Sum K in each process to complete the correlation
            for K_InList in K_List:
                K = K + K_InList

            ray.shutdown()

        except:

            print('Ray parallel processing to generate correlation failed. Try with a single process ...')

            # Sometimes Ray's communications fail. Compute correlation withput parallel section
            K = ComputeCorrelationForAProcess(DecorrelationScale,nu,KernelThreshold,x,y,UseSparse,None,None)

    else:

        # Compute correlation withput parallel section
        K = ComputeCorrelationForAProcess(DecorrelationScale,nu,KernelThreshold,x,y,UseSparse,None,None)

    # Fill lower left elements using symmetry of matrix
    K = K + K.T

    # Density
    if UseSparse == True:

        Density = K.nnz / numpy.prod(K.shape)
        print('Using sparse correlation matrix with kernel threshold: %0.4f and sparsity: %0.4f'%(KernelThreshold,Density))

    # Plot Correlation matrix
    PlotFlag = False
    if PlotFlag:
        fig2,ax2 = plt.subplots()
        p2 = ax2.matshow(K)
        fig2.colorbar(p2,ax=ax2)
        plt.title('Correlation Matrix')
        plt.show()

    return K

# =====================================
# Generate Linear Model Basis Functions
# =====================================

def GenerateLinearModelBasisFunctions(x,y,BasisFunctionsType):
    """
    Generates the basis functions for the mean function of the general linear model.
    """

    print('Computing basis functions of linear model ...')

    n = x.size

    # Basis functions
    if BasisFunctionsType == 'Polynomial-2-Trigonometric-1':
        # Polynomials of x and y of order 2, and trigonometric functions of x and y of order 1. X is matrix of size n*10
        # X = numpy.array([numpy.ones(n),x,y,x**2,x*y,y**2,numpy.sin(x*numpy.pi),numpy.cos(x*numpy.pi),numpy.sin(y*numpy.pi),numpy.cos(y*numpy.pi)]).T
        # X = numpy.array([numpy.ones(n),numpy.sin(x*numpy.pi),numpy.cos(x*numpy.pi),numpy.sin(y*numpy.pi),numpy.cos(y*numpy.pi)]).T
        X = numpy.array([numpy.sin(x*numpy.pi),numpy.cos(x*numpy.pi),numpy.sin(y*numpy.pi),numpy.cos(y*numpy.pi)]).T

    elif BasisFunctionsType == 'Polynomial-5':
        # Polynomial of x and y or order 3. X is matrix of size n*10
        X = numpy.array([numpy.ones(n),x,y,x**2,x*y,y**2,x**3,x**2*y,x*y**2,y**3,x**4,x**3*y,x**2*y**2,x*y**3,y**4,x**5,x**4*y,x**3*y**2,x**2*y**3,x*y**4,y**5]).T  # matrix of size n*21

    elif BasisFunctionsType == 'Polynomial-4':
        # Polynomial of x and y or order 3. X is matrix of size n*10
        X = numpy.array([numpy.ones(n),x,y,x**2,x*y,y**2,x**3,x**2*y,x*y**2,y**3,x**4,x**3*y,x**2*y**2,x*y**3,y**4]).T  # matrix of size n*15

    elif BasisFunctionsType == 'Polynomial-3':
        # Polynomial of x and y or order 3. X is matrix of size n*10
        X = numpy.array([numpy.ones(n),x,y,x**2,x*y,y**2,x**3,x**2*y,x*y**2,y**3]).T  # matrix of size n*10

    elif BasisFunctionsType == 'Polynomial-2':
        # Polynomial of x and y or order 2. X is matrix of size n*6
        X = numpy.array([numpy.ones(n),x,y,x**2,x*y,y**2]).T

    elif BasisFunctionsType == 'Polynomial-1':
        # Polynomial of x and y or order 1. X is matrix of size n*3
        X = numpy.array([numpy.ones(n),x,y]).T

    elif BasisFunctionsType == 'Polynomial-0':
        # Polynomial of x and y or order 0. X is matrix of size n*1
        X = numpy.array([numpy.ones(n)]).T

    else:
        raise ValueError('BasisFunctionsType is invalid')

    return X
