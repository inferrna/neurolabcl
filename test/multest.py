import mynp as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel

ctx = np.ctx

multiply = ElementwiseKernel(ctx,
           "float *a, float *b, float *c",
           "c[i] = z[i] * b[i]",
           "multiplication")

x = np.linspace(0, 1000000, dtype=np.float_)
y = np.linspace(0, 1000000, dtype=np.float_)
z = np.empty(x.shape, x.dtype)

for i in range(2):
    multiply(x, y, z)
