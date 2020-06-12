#! /usr/bin/env python

# =======
# Imports
# =======

import numpy
import scipy.special
from PlotSettings import *

# =============
# Matern Kernel
# =============

def MaternKernel(nu,x):
    """
    Matern correlation Kernel on variable x and parameter nu.
    If nu is infinity, the Matern correlation kernel becomes the Gaussian kernel.

    When x = 0, the kernel should return 1. In this case, to avoid multiplication of zero by
    infinity, we exclude indices of x that x == 0.
    """

    if numpy.isinf(nu):

        # Gassian Kernel
        y = numpy.exp(-0.5*x**2)

    else:

        # If x is zero, avoid multiplication of zero nu infinity
        ZeroIndex = numpy.where(x == 0)[0][:]

        # Alter zeros in x to a dummy value. We will return it back.
        z = numpy.copy(x)
        z[ZeroIndex] = 1

        y = ((2**(1-nu))/scipy.special.gamma(nu)) * ((numpy.sqrt(2.0*nu)*z)**nu) * scipy.special.kv(nu,numpy.sqrt(2.0*nu)*z)

        # Correlation at x = 0 is 1.
        y[ZeroIndex] = 1

    return y

# ====
# Main
# ====

if __name__ == "__main__":
    """
    Set 'PlotErrors' to True to plot errors besides the Matern correlation kernel.
    The errors are the difference between Matern kernel and Gaussian kernel.
    The Gaussian kernel is essentially the Matern kernel for the parameter nu = infinity.
    The purpose of the plot are to show at nu > 25, the Matern kernel is almost the same
    as the Gaussian kernel with less than 1 percent difference error.
    """

    # Set this to True for error plots
    PlotErrors = False

    if PlotErrors:
        nCols = 2
        FigSize = (9.7,3.8)
    else:
        nCols = 1
        FigSize = (5,3.5)

    fig,ax = plt.subplots(ncols=nCols,figsize=FigSize)
    if PlotErrors == False:
        ax = [ax] # to use ax[0]

    Nu = [0.1,0.5,1.0,3.2,25]
    NuLabels = ['0.1','0.5','1.0','3.2','25']
    Colors = sns.color_palette("OrRd_d",len(Nu))[::-1]
    x = numpy.linspace(0,4,1000)

    for i in range(len(Nu)):
        ax[0].plot(x,MaternKernel(Nu[i],x),label=r'$\nu = %s$'%(NuLabels[i]),color=Colors[i])

        if PlotErrors:
            ax[1].plot(x,MaternKernel(numpy.inf,x) - MaternKernel(Nu[i],x),label=r'$\nu = %s$'%(NuLabels[i]),color=Colors[i])

    # Gaussian kernel at nu = infinity
    ax[0].plot(x,MaternKernel(numpy.inf,x),label=r'$\nu = \infty$',color='black')

    ax[0].legend(frameon=False)
    ax[0].set_xlim([x[0],x[-1]])
    ax[0].set_ylim([0,1])
    ax[0].set_xticks(numpy.arange(0,4.01,1))
    ax[0].set_yticks(numpy.arange(0,1.01,0.5))
    ax[0].set_xlabel(r'$r$')
    ax[0].set_ylabel(r'$K(r|\nu)$')
    ax[0].set_title(r'Mat\'{e}rn Correlation Kernel')

    if PlotErrors:
        ax[1].legend(frameon=False)
        ax[1].set_xlim([x[0],x[-1]])
        ax[1].set_ylim([-0.1,0.7])
        ax[1].set_xticks(numpy.arange(0,4.01,1))
        ax[1].set_yticks([-0.1,0,0.1,0.7])
        ax[1].set_xlabel(r'$r$')
        ax[1].set_ylabel(r'$K(r|\infty) - K(r|\nu)$')
        ax[1].set_title(r'Difference of Gaussian and Mat\'{e}rn kernels')

    # Save plot
    plt.tight_layout()
    SaveDir = './doc/images/'
    SaveFilename = 'MaternKernel'
    SaveFilename_PDF = SaveDir + SaveFilename + '.pdf'
    SaveFilename_SVG = SaveDir + SaveFilename + '.svg'
    plt.savefig(SaveFilename_PDF,transparent=True,bbox_inches='tight',pad_inches=0)
    plt.savefig(SaveFilename_SVG,transparent=True,bbox_inches='tight',pad_inches=0)
    print('Plot saved to %s.'%(SaveFilename_PDF))
    print('Plot saved to %s.'%(SaveFilename_SVG))
    # plt.show()
