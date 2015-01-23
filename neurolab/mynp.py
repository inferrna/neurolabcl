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

programcache = {}

typemaps = {
np.int8.__name__:    "char",
np.int16.__name__:   "short",
np.int32.__name__:   "int",
np.int64.__name__:   "long",
np.uint8.__name__:   "uchar",
np.uint16.__name__:  "ushort",
np.uint32.__name__:  "uint",
np.uint64.__name__:  "ulong",
np.float16.__name__: "half",
np.float32.__name__: "float",
np.float64.__name__: "double"}

slicedefs = """
#define dtype {0}
#define PC {1} //Dimensions count
"""
slicesrc = """
uint slice(uint id, __global uint4 *params, uint c){
    uint N = params[c].s0;
    uint x = params[c].s1;
    uint y = params[c].s2;
    uint d = params[c].s3;
    uint ipg = 1+(min(N, y)-(x%N)-1)/d;
    uint s = x/N;
    uint group = s+id/ipg;
    if(c>0) group = slice(group, params, c-1);
    uint groupstart = group*N;
    uint cmd = id%ipg;
    uint groupid = x%N+cmd*d;
    return  groupid+groupstart;
}
"""
slicegetsrc = """
__kernel void mislice(__global uint4 *params, __global dtype *data, __global dtype *result){
    uint gid = get_global_id(0);
    result[gid] = data[slice(gid, params, PC-1)];
}
"""
slicesetsrc = """
__kernel void mislice(__global uint4 *params, __global dtype *data, __global dtype *source){
    uint gid = get_global_id(0);
    data[slice(gid, params, PC-1)] = source[gid];
}
"""


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
        self.ismine = 1

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

    def createshapes(self, index):
            if isinstance(index, slice):
                index = (index,)
            def getslice(x, a):
                if isinstance(x, slice):
                    return x.indices(a)
                elif isinstance(x, int):
                    return slice(x, x+1).indices(a)
            dl = len(self.shape) - len(index)
            for i in range(1, dl+1):
                index = index + (slice(0, self.shape[-i], 1),)
            npindices = np.array([(a,)+getslice(b, a) for a, b in zip(self.shape, index)], dtype=np.uint32)
            newshape = [1+(a[2]-a[1]-1)//a[3] for a in npindices]
            newshape = tuple([a for a, b in zip(newshape, self.shape) if not a==1])
            if newshape == (): newshape = (1,)
            indices = arr_from_np(npindices)
            print("indices == ", npindices)
            print("newshape == ", newshape)
            return indices, newshape

    def __getitem__(self, index):
        if isinstance(index, myclArray) and index.is_boolean == True:
            #print("index is myclArray")
            x, y, z = algorithm.copy_if(self.reshape((self.size,)), "index[i]!=0", [("index", index.reshape((index.size,)))])
            _res = x[:y.get()]
            res = myclArray(queue, _res.shape, _res.dtype, data=_res.data)
        elif isinstance(index, tuple):
            indices, newshape = self.createshapes(index)
            key = (self.dtype, len(self.shape), 'get')
            if not key in programcache.keys():
                ksource = slicedefs.format(typemaps[self.dtype.name], len(self.shape)) + slicesrc + slicegetsrc
                programcache[key] = cl.Program(ctx, ksource).build()
            result = empty(newshape, self.dtype)
            programcache[key].mislice(queue, (result.size,), None, indices.data, self.data, result.data)
            return result
        else:
            #print("index is not myclArray, but", type(index))
            res = clarray.Array.__getitem__(self, index)
        return res

    def __setitem__(self, subscript, _value):
        if isinstance(_value, myclArray) or 'myclArray' in str(type(_value)):
            value = _value
        elif type(_value) in (type(0.4), type(-1)):
            value = arr_from_np(np.array([_value], dtype=self.dtype))
        elif isinstance(_value, np.ndarray):
            value = arr_from_np(_value).astype(self.dtype)

        if isinstance(subscript, myclArray) and subscript.is_boolean == True:
            idxcl = clarray.arange(queue, 0, self.size, 1, dtype=np.int32)
            #print("subscript is myclArray")
            x, y, z = algorithm.copy_if(idxcl, "index[i]!=0", [("index", subscript.reshape((subscript.size,)))])
            _res = x[:y.get()]
            #print(type(_res), _res.dtype.kind, _res)
            clarray.Array.setitem(self.reshape((self.size,)), _res, value, queue=queue)
            #reself = self.reshape((self.size,))
        elif isinstance(subscript, tuple) or isinstance(subscript, slice):
            indices, newshape = self.createshapes(subscript)
            key = (self.dtype, len(self.shape), 'set')
            if not key in programcache.keys():
                ksource = slicedefs.format(typemaps[self.dtype.name], len(self.shape)) + slicesrc + slicesetsrc
                programcache[key] = cl.Program(ctx, ksource).build()
                print(ksource)
            result = empty(newshape, self.dtype)
            print("type(value) == ", type(value))
            programcache[key].mislice(queue, (result.size,), None, indices.data, self.data, value.data)
            #return result
            #reself.setitem(_res, value)
        else:
            #print("subscript is not myclArray, but", type(subscript))
            self.setitem(subscript, _value, queue=queue)
        #return res
    def __add__(self, other):
        if isinstance(other, myclArray) and other.size<2:
            if other.size == 1:
                _res = clarray.Array.__add__(self, other.get())
            elif other.size == 0:
                _res = clarray.Array.__add__(self, other.get())
            else:
                _res = clarray.Array.__add__(self.reshape((self.size,)), other.reshape((other.size,)))
            #    assert False==True, "Unimlimented mul. shapes is {0} and {1}".format(self.shape, other.shape)
        else:
            _res = clarray.Array.__add__(self, other)
        res = myclArray(queue, self.shape, _res.dtype, data=_res.data)
        print("type res == ", type(res))
        return res

    def __mul__(self, other):
        if isinstance(other, myclArray):
            if other.size<2:
                if other.size == 1:
                    _res = clarray.Array.__mul__(self, other.get()[0])
                elif other.size == 0:
                    _res = clarray.Array.__mul__(self, other.get())
            else:
                _res = clarray.Array.__mul__(self.reshape((self.size,)), other.reshape((other.size,)))
            #assert False==True, "Unimlimented mul"
        else:
            _res = clarray.Array.__mul__(self, other)
        res = myclArray(queue, self.shape, _res.dtype, data=_res.data)
        print("type res == ", type(res))
        return res
    def sum(*args, **kwargs):
        return sum(*args, **kwargs)

    def flatten(self):
        return self.ravel()

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
        _res = clrandom.rand(queue, size, np.float32, a=0.0, b=1.0)
        return myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def uniform(self, low=0.0, high=1.0, size=1):
        print("call uniform")
        _res = self.randomeer.uniform(queue, size, np.float32, a=low, b=high)
        return myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def randint(self, low, high=1.0, size=1):
        print("call randint")
        _res = clrandom.rand(queue, size, np.int32, a=low, b=high)
        return myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def rand(self, *args):
        print("args==",args)
        _res = clrandom.rand(queue, args, np.float32, a=0.0, b=1.0)
        return myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def randn(self, *args):
        print("args==",args)
        _res = clrandom.rand(queue, args, np.float32, a=-1.0, b=1.0)
        return myclArray(queue, _res.shape, _res.dtype, data=_res.data)

        
def arr_from_np(nparr):
    if nparr.dtype == np.object:
        nparr = np.concatenate(nparr)
    buf = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=nparr)
    return myclArray(queue, nparr.shape, nparr.dtype, data=buf)

random = myrandom()

def argmin(*args, **kwargs):
    return np.argmin(*args, **kwargs)


def concatenate(arrays, axis=0):
    return clarray.concatenate(arrays, axis, queue)#np.concatenate(*args, **kwargs)


def dot(a, b, out=None):
    #print("dot args==", [type(a) for a in args])
    #TODO: work with out
    _res = clarray.dot(a, b)#np.dot(*args, **kwargs)
    res = myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    return res
    

def floor(a, out=None):
    #TODO: work with out
    _res = clmath.floor(a, queue=queue) #np.floor(*args, **kwargs)
    res = myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    return res


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
    return arr_from_np(np.row_stack(*args, **kwargs))


def tanh(a, out=None):
    #TODO: work with out
    _res = clmath.tanh(a, queue=queue) #np.tanh(*args, **kwargs)
    res = myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    return res


def all(a, axis=None, out=None, keepdims=False):
    #TODO: work with axis, out, keepdims
    return a.all(queue=queue) #np.all(*args, **kwargs)


def asfarray(a, dtype=np.float32):
    if isinstance(a, myclArray):
        return a.astype(dtype, queue=queue)
    else:
        return array(a, dtype=dtype)


def exp(a, out=None):
    #TODO: work with out
    _res = clmath.exp(a, queue=queue) #np.exp(*args, **kwargs)
    res = myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    return res


def linspace(*args, **kwargs):
    #TODO: create native function
    return arr_from_np( np.linspace(*args, **kwargs) )


def min(a):
    return a.min()#np.min(*args, **kwargs)


def sqrt(a, out=None):
    #TODO: work with out
    _res = clmath.sqrt(a, queue=queue) #np.sqrt(*args, **kwargs)
    res = myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    return res


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
    if isinstance(arr, myclArray):
        return arr.__abs__()
    else:
        return arr_from_np(np.abs(*args, **kwargs))

def empty(shape, dtype=np.float32):
    #return arr_from_np( np.empty(*args, **kwargs) )
    return myclArray(queue, shape, dtype)


def argmax(*args, **kwargs):
    return arr_from_np(np.argmax(*args, **kwargs))


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
    _res = clarray.zeros_like(a)
    res = myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    return res


def sum(a, axis=None, dtype=None, out=None):
    #TODO: work with axis, out, keepdims
    _res = clarray.sum(a, queue=queue) #np.sum(*args, **kwargs)
    res = myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    return res

def sin(arr):
    #TODO: work with axis, out, keepdims
    _res = clmath.sin(arr, queue=queue) #np.sum(*args, **kwargs)
    res = myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    return res


def zeros(*args, **kwargs):
    if not 'dtype' in kwargs.keys():
        kwargs['dtype'] = np.float32
    return arr_from_np( np.zeros(*args, **kwargs) )

def array(*args, **kwargs):
    if not 'dtype' in kwargs.keys():
        kwargs['dtype'] = np.float32
    return arr_from_np( np.array(*args, **kwargs) )
