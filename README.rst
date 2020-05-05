SARO: a python package for Statistical Analysis of Robust Observables
=====================================================================

SARO is a companion to XARA by Frantz Martinache (https://github.com/fmartinache/xara).
It adds post-processing functionnality to the package:

In particular:

- Colinearity maps: Laugier et al. (2019), 
- Angular Differential Kernel (ADK): Laugier et al. (2020),
- Energy detector sensitivity maps and GLR detection: Ceau et al. (2019)

Additional dependences:
-----------------------

tqdm
lmfit

*optional:* cupy (cupy is especially helpful for faster computations
in the case of long series with variable detector position angle like 
for ADK.

Usage
-----

The package is designed to be imported along with XARA. Create a saro.KPO instead of the usual xara.KPO. (The package also supersedes xara.KPI.plot_pupil_and_uv() for added functionnalities)

This is not a stable release. Interfaces of methods will evolve. Refer to the saro_test.ipynb notebooks for the intended usage.


Acknowledgement
----------------

SARO is a development carried out in the context of the KERNEL project. KERNEL has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement CoG - 683029). For more information about the KERNEL project, visit: http://frantzmartinache.eu/index.php/category/kernel/
