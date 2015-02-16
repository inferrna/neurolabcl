import numpy as np
import pyopencl as cl
from pyopencl import clrandom, clmath
from pyopencl import array as clarray
from pyopencl import algorithm
import clsrc
import clprograms
from checker import chkmethod

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
programs = clprograms.programs(ctx)
#random = np.random
Inf = np.Inf
mf = cl.mem_flags

class myBuffer(cl._cl.Buffer):
    def __init__(self, *args, **kwargs):
        cl._cl.Buffer.__init__(self, *args, **kwargs)
        self.nowners = 0
    def __del__(self):
        self.nowners -= 1
        if self.nowners == 0:
            #print("released", self.size, "bytes directly")
            self.release()

class myclArray(clarray.Array):
    def __init__(self, *args, **kwargs):
        clarray.Array.__init__(self, *args, **kwargs)
        self.reinit()

    def reinit(self):
        self.ndim = len(self.shape)
#TODO rewrite inner operators to support boolean array as parameter
#https://docs.python.org/3/library/operator.html
        self.is_boolean = False
        self.ismine = 1
        if not isinstance(self.base_data, myBuffer):
            self.base_data.__class__ = myBuffer
            self.base_data.nowners = 1
        else:
            self.base_data.nowners += 1

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
    def __del__(self):
        self.base_data.nowners -=1
        if self.base_data.nowners == 0:
            #print("released", self.base_data.size, "bytes")
            self.base_data.release()
        

    def reshape(self, *shape, **kwargs):
        _res = clarray.Array.reshape(self, *shape, **kwargs)
        if not isinstance(_res, myclArray):
            _res.__class__ = myclArray
            _res.reinit()
        _res.__class__ = myclArray
        _res.reinit()
        res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
        return res

    #def _new_like_me(self):
    #    return empty(self.shape, self.dtype)

    def createshapes(self, index):
            if isinstance(index, slice):
                index = (index,)
            def getslice(x, a):
                if isinstance(x, slice):
                    if x.step and x.step<0:
                        return slice(x.start, x.stop, -x.step).indices(a)[:2]+(x.step,)
                    else:
                        return x.indices(a)
                elif isinstance(x, int):
                    return slice(x, x+1).indices(a)
            dl = len(self.shape) - len(index)
            #Extend index to shape size if less.
            for i in range(0, dl):
                index = index + (slice(0, self.shape[i-dl], 1),)
            npindices = np.array([(a,)+getslice(b, a) for a, b in zip(self.shape, index)], dtype=np.int32)
            newshape = [1+(a[2]-a[1]-1)//a[3].__abs__() for a in npindices]
            newshape = tuple([a for a, b in zip(newshape, self.shape) if not a==1])
            if newshape == (): newshape = (1,)
            indices = arr_from_np(npindices)
            return indices, newshape

    def __getitem__(self, index):
        if isinstance(index, myclArray) and index.is_boolean == True:
            x, y, z = algorithm.copy_if(self.reshape((self.size,)), "index[i]!=0", [("index", index.reshape((index.size,)))])
            _res = x[:y.get()]
        elif isinstance(index, tuple) or isinstance(index, slice):
            indices, newshape = self.createshapes(index)
            program = programs.sliceget(self.dtype, len(self.shape))
            _res = empty(newshape, self.dtype)
            program.mislice(queue, (_res.size,), None, indices.data, self.data, _res.data)
            return _res
        else: 
            _res = clarray.Array.__getitem__(self, index)
        if not isinstance(_res, myclArray):
            _res.__class__ = myclArray
            _res.reinit()
        return _res

    @chkmethod
    def transpose(self, *args):
        replaces = np.array(args, dtype=np.uint32)
        olddims = np.array(self.shape, dtype=np.uint32)
        result = empty(tuple(olddims[replaces]), self.dtype)
        clolddims = myBuffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=olddims)
        clreplaces = myBuffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=replaces)
        program = programs.transpose(self.dtype, self.ndim)
        program.mitransp(queue, (self.size,), None, clolddims, clreplaces, self.data, result.data)
        clreplaces.release()
        clolddims.release()
        return result


    def __setitem__(self, subscript, _value):
        if isinstance(_value, myclArray) or 'myclArray' in str(type(_value)):
            value = _value
        elif type(_value) in (type(0.4), type(-1)):
            value = arr_from_np(np.array([_value], dtype=self.dtype))
        elif isinstance(_value, np.ndarray):
            value = arr_from_np(_value).astype(self.dtype)
        else:
            exit()

        if isinstance(subscript, myclArray) and subscript.is_boolean == True:
            idxcl = clarray.arange(queue, 0, self.size, 1, dtype=np.int32)
            x, y, z = algorithm.copy_if(idxcl, "index[i]!=0", [("index", subscript.reshape((subscript.size,)))])
            _res = x[:y.get()]
            clarray.Array.setitem(self.reshape((self.size,)), _res, value, queue=queue)
        elif isinstance(subscript, tuple) or isinstance(subscript, slice):
            indices, newshape = self.createshapes(subscript)
            program = programs.sliceset(self.dtype, len(self.shape))
            result = empty(newshape, self.dtype)
            assert value.size == result.size or value.size == 1, "Size of value array {0} does not match size of result indices {1}"\
                                                                 .format(value.size, result.size)
            if value.size == result.size: 
                program.mislice(queue, (result.size,), None, indices.data, self.data, value.data)
            elif value.size == 1:
                program.mislicesingle(queue, (result.size,), None, indices.data, self.data, value.data)
        else:
            self.setitem(subscript, _value, queue=queue)

    @chkmethod
    def __sub__(self, other):
        if isinstance(other, myclArray) and not self.shape == other.shape:
            if self.size == 1 and other.size>2:
                result = empty(other.shape, self.dtype)
                program = programs.singlesms(self.dtype)
                program.misinglenegsub(queue, (self.size,), None, other.data, result.data, self.data)
                _res = result
            if other.size==1:
                result = empty(self.shape, self.dtype)
                program = programs.singlesms(self.dtype)
                program.misinglesub(queue, (self.size,), None, self.data, result.data, other.data)
                _res = result
            elif self.size == other.size:
                _res = clarray.Array.__sub__(self, other)
        else:
            _res = clarray.Array.__sub__(self, other)
        if not isinstance(_res, myclArray):
            _res.__class__ = myclArray
            _res.reinit()
        res = _res#myclArray(queue, self.shape, _res.dtype, data=_res.data)
        return res

    @chkmethod
    def __add__(self, other):
        if isinstance(other, myclArray) and not self.shape == other.shape:
            if self.size<2 and other.size>2:
                self, other = other, self
            if other.size==1:
                result = empty(self.shape, self.dtype)
                program = programs.singlesms(self.dtype)
                program.misinglesum(queue, (self.size,), None, self.data, result.data, other.data)
                _res = result
            elif self.size == other.size:
                _res = clarray.Array.__add__(self, other)
        else:
            _res = clarray.Array.__add__(self, other)
        if not isinstance(_res, myclArray):
            _res.__class__ = myclArray
            _res.reinit()
        res = _res#myclArray(queue, self.shape, _res.dtype, data=_res.data)
        return res

    def __iadd__(self, other):
        if isinstance(other, myclArray) and not self.shape == other.shape:
            if self.size<2 and other.size>2:
                self, other = other, self
            if other.size == 1:
                program = programs.singlesms(self.dtype)
                program.misinglesum(queue, (self.size,), None, self.data, self.data, other.data)
                _res = self
            elif self.size == other.size:
                _res = clarray.Array.__iadd__(self.reshape(self.size), other.reshape(self.size)).reshape(self.shape)
            #    assert False==True, "Unimlimented mul. shapes is {0} and {1}".format(self.shape, other.shape)
        else:
            _res = clarray.Array.__iadd__(self, other)
        if not isinstance(_res, myclArray):
            _res.__class__ = myclArray
            _res.reinit()
        res = _res
        return res

    def __mul__(self, other):
        if isinstance(other, myclArray):
            if self.size==1 and other.size>2:
                self, other = other, self
            if other.size == 1:
                program = programs.singlesms(self.dtype)
                _res = empty(self.shape, self.dtype)
                #try:
                program.misinglemul(queue, (_res.size,), None, self.base_data, _res.data, other.base_data)
                #except:
                #    print("types is", type(_res), type(other))
                #    exit()
            else:
                _res = clarray.Array.__mul__(self, other.reshape(self.shape))
            #assert False==True, "Unimlimented mul"
        else:
            _res = clarray.Array.__mul__(self, other)
        if not isinstance(_res, myclArray):
            _res.__class__ = myclArray
            _res.reinit()
        res = _res#myclArray(queue, self.shape, _res.dtype, data=_res.data)
        return res

    def max(*args, **kwargs):
        a = args[0]
        if a.ndim==0 or not 'axis' in kwargs.keys():
            _res = clarray.max(a, queue=queue) #np.sum(*args, **kwargs)
            if not isinstance(_res, myclArray):
                _res.__class__ = myclArray
                _res.reinit()
            return _res
        else:
            kwargs['prg2load'] = programs.max
            return sum(*args, **kwargs)

    def min(*args, **kwargs):
        a = args[0]
        if a.ndim==0 or not 'axis' in kwargs.keys():
            _res = clarray.min(a, queue=queue) #np.sum(*args, **kwargs)
            if not isinstance(_res, myclArray):
                _res.__class__ = myclArray
                _res.reinit()
            return _res
        else:
            kwargs['prg2load'] = programs.min
            return sum(*args, **kwargs)

    def sum(*args, **kwargs):
        a = args[0]
        if a.ndim==0 or not 'axis' in kwargs.keys():
            _res = clarray.sum(a, queue=queue) #np.sum(*args, **kwargs)
            if not isinstance(_res, myclArray):
                _res.__class__ = myclArray
                _res.reinit()
            return _res
        else:
            kwargs['prg2load'] = programs.sum
            return sum(*args, **kwargs)

    def flatten(self):
        return self.ravel()

run = cl.Program(ctx, clsrc.signsrc+clsrc.isinfsrc).build()

#randomeer.uniform(queue, (10,2,), np.float32, a=-0.5, b=0.5)
#np.random.uniform(-0.5, 0.5, (10, 2))

class myrandom():
    def __init__(self):
        #np.random.__init__(self)
        self.randomeer = clrandom.RanluxGenerator(queue)
    def random(self, size):
        _res = clrandom.rand(queue, size, np.float32, a=0.0, b=1.0)
        _res.__class__ = myclArray
        res.reinit()
        return _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def uniform(self, low=0.0, high=1.0, size=1):
        _res = self.randomeer.uniform(queue, size, np.float32, a=low, b=high)
        _res.__class__ = myclArray
        _res.reinit()
        return _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def randint(self, low, high=1.0, size=1):
        _res = clrandom.rand(queue, size, np.int32, a=low, b=high)
        _res.__class__ = myclArray
        _res.reinit()
        return _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def rand(self, *args):
        _res = clrandom.rand(queue, args, np.float32, a=0.0, b=1.0)
        _res.__class__ = myclArray
        _res.reinit()
        return _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def randn(self, *args):
        _res = clrandom.rand(queue, args, np.float32, a=-1.0, b=1.0)
        _res.__class__ = myclArray
        _res.reinit()
        return _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)

        
def arr_from_np(nparr):
    if nparr.dtype == np.object:
        nparr = np.concatenate(nparr)
    buf = myBuffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=nparr)
    return myclArray(queue, nparr.shape, nparr.dtype, data=buf)

random = myrandom()

def argmin(*args, **kwargs):
    return arr_from_np(np.argmin(*args, **kwargs))


def concatenate(arrays, axis=0):
    _res = clarray.concatenate(arrays, axis, queue)#np.concatenate(*args, **kwargs)
    _res.__class__ = myclArray
    res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    res.reinit()
    return res

def dot(a, b, out=None):
    #TODO: work with out
    _res = clarray.dot(a, b)#np.dot(*args, **kwargs)
    _res.__class__ = myclArray
    res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    res.reinit()
    return res
    

def floor(a, out=None):
    #TODO: work with out
    _res = clmath.floor(a, queue=queue) #np.floor(*args, **kwargs)
    _res.__class__ = myclArray
    res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    res.reinit()
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
    res.reinit()
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
    res.reinit()
    return res


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=np.float32):
    #TODO: create native function
    if num<2: return array([start])
    if endpoint:
        mnum = num-1
    else:
        mnum = num
    diff = (stop - start) / mnum
    if endpoint:
        stop = stop + diff
    res = clarray.arange(queue, start, stop, diff, dtype=np.float32)[:num]
    res.__class__ = myclArray
    res.reinit()
    return res


def min(a):
    return a.min()#np.min(*args, **kwargs)


def sqrt(a, out=None):
    #TODO: work with out
    _res = clmath.sqrt(a, queue=queue) #np.sqrt(*args, **kwargs)
    _res.__class__ = myclArray
    res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    res.reinit()
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
        return out1111111111111
    else:
        res = empty(a.shape, dtype=a.dtype)
        if not isinstance(res, myclArray):
            res.__class__ = myclArray
            res.reinit()
        run.asign(queue, (a.size,), None, a.data, res.data)
        return res


def zeros_like(a, dtype=None, order='K', subok=True):
    _res = clarray.zeros_like(a)
    _res.__class__ = myclArray
    res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    res.reinit()
    return res


def sum(a, axis=None, dtype=None, out=None, prg2load=programs.sum):
    #Transpose first to shift target axis to the end
    #do not transpose if axis already is the end
    if axis==None:
        res = clarray.sum(a, queue=queue)
        if not isinstance(res, myclArray):
            res.__class__ = myclArray
            res.reinit()
        return res
    olddims = np.array(a.shape, dtype=np.uint32)
    replaces = np.append(np.delete(np.arange(a.ndim), axis, 0), [axis], 0).astype(np.uint32)
    if axis != a.ndim-1:
        clolddims = myBuffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=olddims)
        clreplaces = myBuffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=replaces)
        cltrresult = myBuffer(ctx, mf.READ_WRITE, a.nbytes)
        program = programs.transpose(a.dtype, a.ndim)
        program.mitransp(queue, (a.size,), None, clolddims, clreplaces, a.data, cltrresult)
    else:
        cltrresult = a.data
    program = prg2load(a.dtype, a.shape[axis])
    #Sum for last axis
    result = empty(tuple(olddims[replaces[:-1]]), a.dtype)
    program.misum(queue, (a.size//a.shape[axis],), None, cltrresult, result.data)
    return result

def sin(arr):
    #TODO: work with axis, out, keepdims
    _res = clmath.sin(arr, queue=queue) #np.sum(*args, **kwargs)
    _res.__class__ = myclArray
    res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    res.reinit()
    return res


def zeros(shape, dtype=np.float32, order='C'):
    res = clarray.zeros(queue, shape, dtype, order)
    res.__class__ = myclArray
    res.reinit()
    return res

def array(*args, **kwargs):
    if not 'dtype' in kwargs.keys():
        kwargs['dtype'] = np.float32
    return arr_from_np( np.array(*args, **kwargs) )
