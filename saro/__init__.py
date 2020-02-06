gpu = False
import numpy as np
if not gpu:
    import numpy as cp
else:
    import cupy as cp
import xara

from . import kernel_cm
from .kernel_cm import br, bo, vdg, bbr
from . import detection_maps as dmap
#from .detection_maps import detection_maps as dmap


import xaosim
from tqdm import  tqdm
import matplotlib.pyplot as plt

import scipy.sparse as sparse
from scipy.linalg import sqrtm

from xara import fft, ifft, shift

from . import saro
from .saro import *



version_info = (0,1,0)
__version__ = '.'.join(str(c) for c in version_info)

# -------------------------------------------------
# set some defaults to display images that will
# look more like the DS9 display....
# -------------------------------------------------
#plt.set_cmap(cm.gray)
(plt.rcParams)['image.origin']        = 'lower'
(plt.rcParams)['image.interpolation'] = 'nearest'
# -------------------------------------------------

plt.ion()
plt.show()
