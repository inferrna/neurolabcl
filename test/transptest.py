import numpy as np
import mynp as mnp
import pyopencl as cl
from pyopencl import array
import time

ms = mnp.random.randint(255, size=(512, 256, 128,))
s = ms.get()
ms.transpose(2,1,0)
s.transpose(2,1,0)

