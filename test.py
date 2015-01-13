import pyopencl as cl
from pyopencl import array
from pyopencl import clrandom
import numpy as np
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

s = clrandom.rand(queue, 4, np.int32, luxury=None, a=0, b=2)
