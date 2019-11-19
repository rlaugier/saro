SARO: a python package for Statistical Analysis of Robust Observables
=====================================================================

SARO is a companion to XARA by Frantz Martinache (https://github.com/fmartinache/xara). It adds post-processing functionnality to
the package:

In particular:

- Colinearity maps: Laugier et al. (2019), 
- Energy detector sensitivity maps and GLR detection: Ceau et al. (2019)
- Angular Differential Kernel (ADK): Laugier et al. (submitted)

Additional dependences:
-----------------------

tqdm
lmfit

*optional:* cupy (cupy is especially helpful for faster computations of
in the case of long series with variable detector position angle like 
for ADK.

Usage
-----

The package is designed to be imported along with XARA.
