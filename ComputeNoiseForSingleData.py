# =======
# Imports
# =======

# Classes
import Data
from LikelihoodEstimation import LikelihoodEstimation
from TraceEstimation import TraceEstimation
from PlotSettings import *

# =============================
# Compute Noise For Single Data
# =============================

def ComputeNoiseForSingleData():
    """
    This function uses three methods
        1. Maximizing log likelihood with parameters sigma and sigma0
        2. Maximizing log likelihood with parameters sigma and eta
        3. Finding zeros of derivative of log likelihood

    This script uses a single data, for which the random noise with a given standard deviation is added to the data once.
    It plots
        1. liklelihood in 3D as function of parameters sigma and eta
        2. Trace estimation using interpolation
        3. Derivative of log likelihood.
    """

    # Generate noisy data
    NumPointsAlongAxis = 50
    NoiseMagnitude = 0.2
    GridOfPoints = True
    x,y,z = Data.GenerateData(NumPointsAlongAxis,NoiseMagnitude,GridOfPoints)

    # Generate Linear Model
    DecorrelationScale = 0.1
    UseSparse = False
    nu = 0.5
    K = Data.GenerateCorrelationMatrix(x,y,z,DecorrelationScale,nu,UseSparse)

    # BasisFunctionsType = 'Polynomial-2-Trigonometric-1'
    # BasisFunctionsType = 'Polynomial-5'
    # BasisFunctionsType = 'Polynomial-4'
    # BasisFunctionsType = 'Polynomial-3'
    BasisFunctionsType = 'Polynomial-2'
    # BasisFunctionsType = 'Polynomial-1'
    # BasisFunctionsType = 'Polynomial-0'
    X = Data.GenerateLinearModelBasisFunctions(x,y,BasisFunctionsType)

    # Trace estimation weights
    UseEigenvaluesMethod = False    # If set to True, it overrides the interpolation estimation methods
    # EstimationMethod = 'NonOrthogonalFunctionsMethod'   # highest condtion number
    # EstimationMethod = 'OrthogonalFunctionsMethod'      # still high condition number
    EstimationMethod = 'OrthogonalFunctionsMethod2'     # best (lowest) condition number
    # EstimationMethod = 'RBFMethod'

    # Precompute trace interpolation function
    TraceEstimationUtilities_1 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,'OrthogonalFunctionsMethod2',None,[1e-4,4e-4,1e-3,1e-2,1e-1,1,1e+1,1e+2,1e+3])
    TraceEstimationUtilities_2 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,'OrthogonalFunctionsMethod2',None,[1e-4,1e-3,1e-2,1e-1,1,1e+1,1e+3])
    TraceEstimationUtilities_3 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,'OrthogonalFunctionsMethod2',None,[1e-3,1e-2,1e-1,1e+1,1e+3])
    TraceEstimationUtilities_4 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,'OrthogonalFunctionsMethod2',None,[1e-3,1e-1,1e+1])
    TraceEstimationUtilities_5 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,'OrthogonalFunctionsMethod2',None,[1e-1])
    # TraceEstimationUtilities_6 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,'RBFMethod',1,[1e-4,1e-3,1e-2,1e-1,1,1e+1,1e+2,1e+3])
    # TraceEstimationUtilities_7 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,'RBFMethod',2,[1e-2,1e-1,1,1e+1,1e+2])
    # TraceEstimationUtilities_8 = TraceEstimation.ComputeTraceEstimationUtilities(K,UseEigenvaluesMethod,'RBFMethod',3,[1e-2,1e-1,1,1e+1,1e+2])

    TraceEstimationUtilitiesList = [ \
            TraceEstimationUtilities_1,
            TraceEstimationUtilities_2,
            TraceEstimationUtilities_3,
            TraceEstimationUtilities_4,
            TraceEstimationUtilities_5]

    # Finding optimal parameters with maximum likelihood using parameters (sigma,sigma0)
    # Results = LikelihoodEstimation.MaximizeLogLikelihoodWithSigmaSigma0(z,X,K,TraceEstimationUtilities_1)
    # print(Results)

    # Finding optimal parameters with maximum likelihood using parameters (sigma,eta)
    # Results = LikelihoodEstimation.MaximizeLogLikelihoodWithSigmaEta(z,X,K,TraceEstimationUtilities_1)
    # print(Results)

    # Finding optimal parameters with derivative of likelihood
    Interval_eta = [1e-4,1e+3]   # Note: make sure the interval is exactly the end points of eta_i, not less or more.
    Results = LikelihoodEstimation.FindZeroOfLogLikelihoodFirstDerivative(z,X,K,TraceEstimationUtilities_1,Interval_eta)
    print(Results)

    # Plot likelihood and its derivative
    # LikelihoodEstimation.PlotLogLikelihood(z,X,K,TraceEstimationUtilities_1)
    LikelihoodEstimation.PlotLogLikelihoodFirstDerivative(z,X,K,TraceEstimationUtilities_1,Results['eta'])

    # Plot Trace Estimate
    TraceEstimation.PlotTraceEstimate(TraceEstimationUtilitiesList,K)

# ====
# Main
# ====

if __name__ == "__main__":

    ComputeNoiseForSingleData()
