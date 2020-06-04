#! /usr/bin/env python

# =======
# Imports
# =======

# Python packages
import sys

# Classes and files
import ComputeNoiseForSingleData
import CompareVariousNoiseLevel
import CompareVariousNumberOfPoints

# ====
# Main
# ====

def main(argv):
    """
    Manager of the script.
    """

    # ComputeNoiseForSingleData.ComputeNoiseForSingleData()
    CompareVariousNoiseLevel.CompareComputationWithVariousNoiselevel()
    # CompareVariousNumberOfPoints.CompareComputationWithVariousNumberOfPoints()

# ===========
# System Main
# ===========

if __name__ == "__main__":
    main(sys.argv)
