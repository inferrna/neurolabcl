import numpy as np
import pyopencl as cl
from pyopencl import clrandom, clmath
from pyopencl import array as clarray
from pyopencl import algorithm
import clsrc

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
        _res.__class__ = myclArray
        res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
        return res

    #def _new_like_me(self):
    #    return empty(self.shape, self.dtype)

    def createshapes(self, index):
            if isinstance(index, slice):
                index = (index,)
            def getslice(x, a):
                if isinstance(x, slice):
                    return x.indices(a)
                elif isinstance(x, int):
                    return slice(x, x+1).indices(a)
            dl = len(self.shape) - len(index)
            for i in range(0, dl):
                index = index + (slice(0, self.shape[i-dl], 1),)
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
        elif isinstance(index, tuple) or isinstance(index, slice):
            indices, newshape = self.createshapes(index)
            key = (self.dtype, len(self.shape), 'get')
            if not key in programcache.keys():
                ksource = clsrc.slicedefs.format(typemaps[self.dtype.name], len(self.shape)) + clsrc.slicesrc + clsrc.slicegetsrc
                programcache[key] = cl.Program(ctx, ksource).build()
            _res = empty(newshape, self.dtype)
            programcache[key].mislice(queue, (_res.size,), None, indices.data, self.data, _res.data)
            return _res
        else: 
            _res = clarray.Array.__getitem__(self, index)
        _res.__class__ = myclArray
        return _res

    def transpose(self, *args):
        replaces = np.array(args, dtype=np.uint32)
        print(replaces)
        olddims = np.array(self.shape, dtype=np.uint32)
        result = empty(tuple(olddims[replaces]), self.dtype)
        clolddims = cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=olddims)
        clreplaces = cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=replaces)
        key = (self.dtype, self.ndim, 'transpose')
        if not key in programcache.keys():
            ksource = clsrc.slicedefs.format(typemaps[self.dtype.name], self.ndim) + clsrc.transpsrc
            programcache[key] = cl.Program(ctx, ksource).build()
            #print(ksource)
        
        #print("type(value) == ", type(value))
        programcache[key].mitransp(queue, (self.size,), None, clolddims, clreplaces, self.data, result.data)
        return result


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
                ksource = clsrc.slicedefs.format(typemaps[self.dtype.name], len(self.shape)) + clsrc.slicesrc + clsrc.slicesetsrc
                programcache[key] = cl.Program(ctx, ksource).build()
                print(ksource)
            result = empty(newshape, self.dtype)
            assert value.size == result.size or value.size == 1, "Size of value array {0} does not match size of result indices {1}"\
                                                                 .format(value.size, result.size)
            print("type(value) == ", type(value))
            if value.size == result.size: 
                programcache[key].mislice(queue, (result.size,), None, indices.data, self.data, value.data)
            elif value.size == 1:
                programcache[key].mislicesingle(queue, (result.size,), None, indices.data, self.data, value.data)
            #return result
            #reself.setitem(_res, value)
        else:
            #print("subscript is not myclArray, but", type(subscript))
            self.setitem(subscript, _value, queue=queue)
        #return res
    def __add__(self, other):
        if isinstance(other, myclArray) and other.size<2:
            if(self.size<2 and other.size>2):
                self, other = other, self
            if other.size == 1:
                _res = clarray.Array.__add__(self, other.get())
            elif other.size == 0:
                _res = clarray.Array.__add__(self, other.get())
            else:
                _res = clarray.Array.__add__(self.reshape((self.size,)), other.reshape((other.size,)))
            #    assert False==True, "Unimlimented mul. shapes is {0} and {1}".format(self.shape, other.shape)
        else:
            _res = clarray.Array.__add__(self, other)
        _res.__class__ = myclArray
        res = _res#myclArray(queue, self.shape, _res.dtype, data=_res.data)
        print("__add__. type res == ", type(res))
        return res

    def __iadd__(self, other):
        print("Shapes is", self.shape, other.shape)
        if isinstance(other, myclArray) and other.size<2:
            if(self.size<2 and other.size>2):
                self, other = other, self
            if other.size == 1:
                _res = clarray.Array.__iadd__(self, other.get())
            elif other.size == 0:
                _res = clarray.Array.__iadd__(self, other.get())
            elif self.size == other.size:
                _res = clarray.Array.__iadd__(self.reshape((self.size,)), other.reshape((other.size,)))
            #    assert False==True, "Unimlimented mul. shapes is {0} and {1}".format(self.shape, other.shape)
        else:
            _res = clarray.Array.__iadd__(self, other)
        _res.__class__ = myclArray
        res = _res#myclArray(queue, self.shape, _res.dtype, data=_res.data)
        print("__iadd__. type res == ", type(res))
        return res

    def __mul__(self, other):
        if isinstance(other, myclArray):
            if(self.size<2 and other.size>2):
                self, other = other, self
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
        res = _res#myclArray(queue, self.shape, _res.dtype, data=_res.data)
        res.__class__ = myclArray
        print("__mul__. type res == ", type(res))
        return res

    def sum(*args, **kwargs):

        return sum(*args, **kwargs)

    def flatten(self):
        return self.ravel()

run = cl.Program(ctx, clsrc.signsrc+clsrc.isinfsrc).build()

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
        _res.__class__ = myclArray
        return _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def uniform(self, low=0.0, high=1.0, size=1):
        print("call uniform")
        _res = self.randomeer.uniform(queue, size, np.float32, a=low, b=high)
        _res.__class__ = myclArray
        return _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def randint(self, low, high=1.0, size=1):
        print("call randint")
        _res = clrandom.rand(queue, size, np.int32, a=low, b=high)
        _res.__class__ = myclArray
        return _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def rand(self, *args):
        print("args==",args)
        _res = clrandom.rand(queue, args, np.float32, a=0.0, b=1.0)
        _res.__class__ = myclArray
        return _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def randn(self, *args):
        print("args==",args)
        _res = clrandom.rand(queue, args, np.float32, a=-1.0, b=1.0)
        _res.__class__ = myclArray
        return _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)

        
def arr_from_np(nparr):
    if nparr.dtype == np.object:
        print(type(nparr[0]))
        print(nparr)
        nparr = np.array(nparr, dtype=nparr[0].dtype)
    buf = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=nparr)
    return myclArray(queue, nparr.shape, nparr.dtype, data=buf)

random = myrandom()

def argmin(*args, **kwargs):
    return arr_from_np(np.argmin(*args, **kwargs))


def concatenate(arrays, axis=0):
    _res = clarray.concatenate(arrays, axis, queue)#np.concatenate(*args, **kwargs)
    _res.__class__ = myclArray
    res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    return res

def dot(a, b, out=None):
    #print("dot args==", [type(a) for a in args])
    #TODO: work with out
    _res = clarray.dot(a, b)#np.dot(*args, **kwargs)
    _res.__class__ = myclArray
    res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    return res
    

def floor(a, out=None):
    #TODO: work with out
    _res = clmath.floor(a, queue=queue) #np.floor(*args, **kwargs)
    _res.__class__ = myclArray
    res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
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
    _res.__class__ = myclArray
    res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
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
    _res.__class__ = myclArray
    res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    return res


def linspace(*args, **kwargs):
    #TODO: create native function
    return arr_from_np( np.linspace(*args, **kwargs) )


def min(a):
    return a.min()#np.min(*args, **kwargs)


def sqrt(a, out=None):
    #TODO: work with out
    print(type(a))
    _res = clmath.sqrt(a, queue=queue) #np.sqrt(*args, **kwargs)
    _res.__class__ = myclArray
    res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    return res


def values(*args, **kwargs):
    return arr_from_np(np.values(*args, **kwargs))


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
    _res.__class__ = myclArray
    res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    return res


def sum(a, axis=None, dtype=None, out=None):
    #TODO: work with axis, out, keepdims
    if not axis==None and a.ndim>1:
        #Transpose first to shift target axis to the end
        #do not transpose if axis already is the end
        if axis == a.ndim-1:
            replaces = np.append(np.delete(np.arange(a.ndim), axis, 0), [axis], 0).astype(np.uint32)
            olddims = np.array(a.shape, dtype=np.uint32)
            clolddims = cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=olddims)
            clreplaces = cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=replaces)
            cltrresult = cl.Buffer(ctx, mf.READ_WRITE, a.nbytes)
            key = (a.dtype, a.ndim, 'transpose')
            if not key in programcache.keys():
                ksource = clsrc.slicedefs.format(typemaps[a.dtype.name], a.ndim) + clsrc.transpsrc
                programcache[key] = cl.Program(ctx, ksource).build()
            programcache[key].mitransp(queue, (a.size,), None, clolddims, clreplaces, a.data, cltrresult)
        else:
            cltrresult = a.data

        key = (a.dtype, a.shape[axis], 'sum')
        #Sum for last axis
        if not key in programcache.keys():
            ksource  = clsrc.slicedefs.format(typemaps[a.dtype.name], a.shape[axis]) + clsrc.sumsrc
            programcache[key] = cl.Program(ctx, ksource).build()
        result = empty(tuple(olddims[replaces[:-1]]), a.dtype)
        programcache[key].misum(queue, (a.size//a.shape[axis],), None, cltrresult, result.data)
        result.__class__ = myclArray
        #print("type(value) == ", type(value))
        return result
    else:
        _res = clarray.sum(a, queue=queue) #np.sum(*args, **kwargs)
        _res.__class__ = myclArray
        res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
        #if res.size==1: return res.get()
        #else: 
        return res

def sin(arr):
    #TODO: work with axis, out, keepdims
    _res = clmath.sin(arr, queue=queue) #np.sum(*args, **kwargs)
    _res.__class__ = myclArray
    res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    return res


def zeros(*args, **kwargs):
    if not 'dtype' in kwargs.keys():
        kwargs['dtype'] = np.float32
    return arr_from_np( np.zeros(*args, **kwargs) )

def array(*args, **kwargs):
    if not 'dtype' in kwargs.keys():
        kwargs['dtype'] = np.float32
    return arr_from_np( np.array(*args, **kwargs) )
