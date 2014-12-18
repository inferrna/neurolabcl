import numpy as np
import pyopencl as cl
from pyopencl import clrandom, array, clmath
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
#random = np.random
Inf = np.Inf

#randomeer.uniform(queue, (10,2,), np.float32, a=-0.5, b=0.5)
#np.random.uniform(-0.5, 0.5, (10, 2))

class myrandom():
    def __init__(self):
        print("Init random")
        #np.random.__init__(self)
        self.randomeer = clrandom.RanluxGenerator(queue)
    def random(self, size):
        print("call random")
        res = clrandom.rand(queue, size, np.float32, a=0.0, b=1.0)
        return res
    def uniform(self, low=0.0, high=1.0, size=1):
        print("call uniform")
        res = self.randomeer.uniform(queue, size, np.float32, a=low, b=high)
        return res
    def randint(self, low, high=1.0, size=1):
        print("call randint")
        res = clrandom.rand(queue, size, np.int32, a=low, b=high)
        return res
    def rand(self, *args):
        print("args==",args)
        res = clrandom.rand(queue, args, np.float32, a=0.0, b=1.0)
        return res
        
def arr_from_np(nparr):
    buf = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=nparr)
    return array.Array(queue, nparr.shape, nparr.dtype, data=buf)

random = myrandom()

def argmin(*args, **kwargs):
    return np.argmin(*args, **kwargs)


def concatenate(arrays, axis=0):
    return array.concatenate(arrays, axis, queue)#np.concatenate(*args, **kwargs)


def dot(a, b, out=None):
    print("dot args==", [type(a) for a in args])
    #TODO: work with out
    return array.dot(a, b)#np.dot(*args, **kwargs)


def floor(a, out=None):
    #TODO: work with out
    return clmath.floor(a, queue=queue) #np.floor(*args, **kwargs)


def isneginf(*args, **kwargs):
    return np.isneginf(*args, **kwargs)


def ones_like(a, dtype=None, order='K', subok=True):
    res = empty(a.shape, dtype=a.dtype)
    res.fill(1, queue=queue)
    return res


def row_stack(*args, **kwargs):
    return np.row_stack(*args, **kwargs)


def tanh(a, out=None):
    #TODO: work with out
    return clmath.tanh(a, queue=queue) #np.tanh(*args, **kwargs)


def all(a, axis=None, out=None, keepdims=False):
    #TODO: work with axis, out, keepdims
    return a.all(queue=queue) #np.all(*args, **kwargs)


def asfarray(a, dtype=np.float32):
    return a.astype(dtype, queue=queue)

def exp(a, out=None):
    #TODO: work with out
    return clmath.exp(a, queue=queue) #np.tanh(*args, **kwargs)


def linspace(*args, **kwargs):
    return np.linspace(*args, **kwargs)


def min(*args, **kwargs):
    return np.min(*args, **kwargs)


def sqrt(*args, **kwargs):
    return np.sqrt(*args, **kwargs)


def values(*args, **kwargs):
    return np.values(*args, **kwargs)


def isinf(*args, **kwargs):
    return np.isinf(*args, **kwargs)


def items(*args, **kwargs):
    return np.items(*args, **kwargs)


def max(*args, **kwargs):
    return np.max(*args, **kwargs)


def py(*args, **kwargs):
    return np.py(*args, **kwargs)


def abs(*args, **kwargs):
    arr = args[0]
    if type(arr).__name__ == 'Array':
        return arr.__abs__()
    else:
        return np.abs(*args, **kwargs)


def empty(shape, dtype=np.float32):
    #return arr_from_np( np.empty(*args, **kwargs) )
    return array.Array(queue, shape, dtype)



def argmax(*args, **kwargs):
    return np.argmax(*args, **kwargs)


def square(*args, **kwargs):
    return np.square(*args, **kwargs)


def sign(*args, **kwargs):
    return np.sign(*args, **kwargs)


def zeros_like(a, dtype=None, order='K', subok=True):
    return array.zeros_like(a)


def sum(*args, **kwargs):
    return np.sum(*args, **kwargs)


def zeros(*args, **kwargs):
    return arr_from_np( np.zeros(*args, **kwargs) )

def asfarray(*args, **kwargs):
    return np.asfarray(*args, **kwargs)


def array(*args, **kwargs):
    return arr_from_np( np.array(*args, **kwargs) )

