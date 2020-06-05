### Description

Todo

### Package Prerequisits

The Python packages that are required are `numpy`, `sympy`, `scikit-sparse`, `psutil`, and `matplotlib`. If you are using [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) python distirbutions in Linux environment, these Python packages can be installed by

    $ sudo conda install -c conda-forge numpy sympy psutil scikit-sparse matplotlib -y

If you run the examples, the `ray` package is also needed. Install it using `pip` as follows,

   $ sudo conda install -c conda-forge pip

   # Use pip installation from anaconda/miniconda.
   # If neccessary, use the full path for pip, such as /opt/miniconda/bin/pip
   $ sudo pip install ray
    
### Usage

    $ python NoiseEstimation.py [options]
    
### Credits

__Author:__

   Siavash Ameli (University of California, Berkeley)

__Citation:__

   Ameli, S. and Shadden. S. C. (2020). Maximum Likelihood Estimation of Variance and Nugget in General Linear Model.

__License:__ [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)
