import numpy as np
import mynp as mnp
import pyopencl as cl
from pyopencl import array
import time

ms = mnp.random.randint(255, size=(8192,3072,4,))
s = ms.get()
ms.transpose(1,0,2)
t1 = time.time()
s.transpose(1,0,2)
ts = time.time() - t1

t1 = time.time()
ms.transpose(1,0,2)
tms = time.time() - t1

print(ts, "vs", tms)
