import numpy as np
import pyopencl as cl
from pyopencl import clrandom, clmath
from pyopencl import array as clarray
from pyopencl import algorithm
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
#random = np.random
Inf = np.Inf
mf = cl.mem_flags

signsrc = """
__kernel void asign(__global float *inpt, __global float *outpt){
    uint gid = get_global_id(0);
    float res = copysign(1, inpt[gid]);
    outpt[gid] = res; 
}\n
"""
isinfsrc = """
__kernel void isposinf(__global float *inpt, __global uint *outpt){
    uint gid = get_global_id(0);
    float val = inpt[gid];
    float res = isinf(val);
    outpt[gid] = res;
}\n
__kernel void isneginf(__global float *inpt, __global uint *outpt){
    uint gid = get_global_id(0);
    float val = inpt[gid];
    float res =  signbit(val) * isinf(val);
    outpt[gid] = res;
}\n
"""

class myclArray(clarray.Array):
    def __init__(self, *args, **kwargs):
        clarray.Array.__init__(self, *args, **kwargs)
        self.ndim = len(self.shape)
#TODO rewrite inner operators to support boolean array as parameter
#https://docs.python.org/3/library/operator.html
        self.is_boolean = False

    def __lt__(self, other):
        result = clarray.Array.__lt__(self, other)
        result.is_boolean = True
        return result
    def __le__(self, other):
        result = clarray.Array.__le__(self, other)
        result.is_boolean = True
        return result
    def __eq__(self, other):
        result = clarray.Array.__eq__(self, other)
        result.is_boolean = True
        return result
    def __ne__(self, other):
        result = clarray.Array.__ne__(self, other)
        result.is_boolean = True
        return result
    def __ge__(self, other):
        result = clarray.Array.__ge__(self, other)
        result.is_boolean = True
        return result
    def __gt__(self, other):
        result = clarray.Array.__gt__(self, other)
        result.is_boolean = True
        return result

    def reshape(self, *shape, **kwargs):
        _res = clarray.Array.reshape(self, *shape, **kwargs)
        res = myclArray(queue, _res.shape, _res.dtype, data=_res.data)
        return res


    def __getitem__(self, index):
        if isinstance(index, myclArray) and index.is_boolean == True:
            #print("index is myclArray")
            x, y, z = algorithm.copy_if(self.reshape((self.size,)), "index[i]!=0", [("index", index.reshape((index.size,)))])
            _res = x[:y.get()]
            res = myclArray(queue, _res.shape, _res.dtype, data=_res.data)
        else:
            #print("index is not myclArray, but", type(index))
            res = clarray.Array.__getitem__(self, index)
        return res

    def __setitem__(self, subscript, _value):
        if type(_value) in (type(0.4), type(-1)):
            value = arr_from_np(np.array([_value], dtype=self.dtype))
        else:
            value = _value
        if isinstance(subscript, myclArray) and subscript.is_boolean == True:
            idxcl = clarray.arange(queue, 0, self.size, 1, dtype=np.int32)
            #print("subscript is myclArray")
            x, y, z = algorithm.copy_if(idxcl, "index[i]!=0", [("index", subscript.reshape((subscript.size,)))])
            _res = x[:y.get()]
            #print(type(_res), _res.dtype.kind, _res)
            clarray.Array.setitem(self.reshape((self.size,)), _res, value, queue=queue)
            #reself = self.reshape((self.size,))
            #reself.setitem(_res, value)
        else:
            #print("subscript is not myclArray, but", type(subscript))
            self.setitem(subscript, value, queue=queue)
        #return res


run = cl.Program(ctx, signsrc+isinfsrc).build()

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
    return myclArray(queue, nparr.shape, nparr.dtype, data=buf)

random = myrandom()

def argmin(*args, **kwargs):
    return np.argmin(*args, **kwargs)


def concatenate(arrays, axis=0):
    return clarray.concatenate(arrays, axis, queue)#np.concatenate(*args, **kwargs)


def dot(a, b, out=None):
    print("dot args==", [type(a) for a in args])
    #TODO: work with out
    return clarray.dot(a, b)#np.dot(*args, **kwargs)


def floor(a, out=None):
    #TODO: work with out
    return clmath.floor(a, queue=queue) #np.floor(*args, **kwargs)


def isneginf(a, out=None):
    if out:
        run.isneginf(queue, (a.size,), None, a.data, out.data)
        out.is_boolean = True
        return out
    else:
        res = empty(a.shape, dtype=np.uint32)
        #res = clarray.empty_like(a)
        run.isneginf(queue, (a.size,), None, a.data, res.data)
        res.is_boolean = True
        return res
    #return np.isneginf(*args, **kwargs)


def ones_like(a, dtype=np.float32, order='K', subok=True):
    res = empty(a.shape, dtype=(dtype or a.dtype))
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
    if type(a).__name__ == 'Array':
        return a.astype(dtype, queue=queue)
    else:
        return array(a, dtype=dtype)


def exp(a, out=None):
    #TODO: work with out
    return clmath.exp(a, queue=queue) #np.exp(*args, **kwargs)


def linspace(*args, **kwargs):
    #TODO: create native function
    return arr_from_np( np.linspace(*args, **kwargs) )


def min(a):
    return a.min()#np.min(*args, **kwargs)


def sqrt(a, out=None):
    #TODO: work with out
    return clmath.sqrt(a, queue=queue) #np.sqrt(*args, **kwargs)


def values(*args, **kwargs):
    return np.values(*args, **kwargs)


def isinf(a, out=None):
    if out:
        run.isposinf(queue, (a.size,), None, a.data, out.data)
        out.is_boolean = True
        return out
    else:
        res = empty(a.shape, dtype=np.uint32)
        run.isposinf(queue, (a.size,), None, a.data, res.data)
        res.is_boolean = True
        return res
    #return np.isinf(*args, **kwargs)


def items(*args, **kwargs):
    return np.items(*args, **kwargs)

def max(a):
    return a.max()#np.max(*args, **kwargs)

def abs(*args, **kwargs):
    arr = args[0]
    if type(arr).__name__ == 'Array':
        return arr.__abs__()
    else:
        return np.abs(*args, **kwargs)

def empty(shape, dtype=np.float32):
    #return arr_from_np( np.empty(*args, **kwargs) )
    return myclArray(queue, shape, dtype)


def argmax(*args, **kwargs):
    return np.argmax(*args, **kwargs)


def square(a, out=None):
    #TODO: work with out
    return a*a #np.square(*args, **kwargs)


def sign(a, out=None):
    if out:
        run.asign(queue, (a.size,), None, a.data, out.data)
        return out
    else:
        res = clarray.empty_like(a)
        run.asign(queue, (a.size,), None, a.data, res.data)
        return res


def zeros_like(a, dtype=None, order='K', subok=True):
    return clarray.zeros_like(a)


def sum(a, axis=None, dtype=None, out=None):
    #TODO: work with axis, out, keepdims
    return clarray.sum(a, queue=queue) #np.sum(*args, **kwargs)


def zeros(*args, **kwargs):
    if not 'dtype' in kwargs.keys():
        kwargs['dtype'] = np.float32
    return arr_from_np( np.zeros(*args, **kwargs) )

def array(*args, **kwargs):
    if not 'dtype' in kwargs.keys():
        kwargs['dtype'] = np.float32
    return arr_from_np( np.array(*args, **kwargs) )
