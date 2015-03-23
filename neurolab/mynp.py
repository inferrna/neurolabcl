import numpy as np
import pyopencl as cl
from pyopencl import clrandom, clmath
from pyopencl import array as clarray
from pyopencl import algorithm
import clsrc
import clprograms
from checker import chkvoidmethod, chkmethod, justtime, chkfunc
from builtins import sum as bsum

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
programs = clprograms.programs(ctx)
#random = np.random
Inf = np.Inf
mf = cl.mem_flags
arngd = np.array([0])

@justtime
def get_arng(size, dtype=np.int32):
    return clarray.arange(queue, 0, size, 1, dtype=dtype)

class myBuffer(cl._cl.Buffer):
    def __init__(self, *args, **kwargs):
        cl._cl.Buffer.__init__(self, *args, **kwargs)
        self.nowners = 0
    def __del__(self):
        self.nowners -= 1
        if self.nowners == 0:
            #print("released", self.size, "bytes directly")
            self.release()

fallbacks = {
    'isub': clarray.Array.__isub__,
    'sub':  clarray.Array.__sub__,
    'iadd': clarray.Array.__iadd__,
    'add':  clarray.Array.__add__,
    'imul': clarray.Array.__imul__,
    'mul':  clarray.Array.__mul__
}

        
def meta_add(self, other, actnames):
    actname = actnames[-1]
    # Original method, eg clarray.Array.__add__
    fallbackM = fallbacks[''.join(actnames)]
    result = None
    if actnames[0] == 'i': result = self

    if isinstance(other, myclArray) and not self.shape == other.shape:
        neg = ''
        if self.size == 1 and other.size>2:
            self, other = other, self
            if actname == 'sub':
                neg = '-'
                actname = 'add'
        if other.size==1 and not other.offset:
            if not result: result = empty(self.shape, self.dtype)
            singleprogram = programs.singlesms(self.dtype, actname, neg).prg
            singleprogram(queue, (self.size,), None, self.data, result.data, other.data)
            res = result
        elif other.size==1 and other.offset:
            res = fallbackM(self, other.get()[0])
        elif self.size == other.size:
            res = fallbackM(self.reshape(self.size), other.reshape(self.size)).reshape(self.shape)
        elif self.shape[-other.ndim:] == other.shape:
            if not result: result = empty(self.shape, self.dtype)
            s1 = np.prod(self.shape[:-other.ndim])
            s2 = np.prod(other.shape)
            ndprogram = programs.ndsms(self.dtype, fname).prg
            ndprogram(queue,\
                      tuple([int(s1), int(s2)]),\
                      None,\
                      self.data,\
                      result.data,\
                      other.data)
            res = result
        elif self.shape[:other.ndim] == other.shape:
            if not result: result = empty(self.shape, self.dtype)
            N = np.prod(self.shape[other.ndim:])
            ndrprogram = programs.ndrsms(self.dtype, N, fname).prg
            ndrprogram(queue,\
                       (self.size,),\
                       None,\
                       self.data,\
                       result.data,\
                       other.data)
            res = result
    else:
        res = fallbackM(self, other)
    if not isinstance(res, myclArray):
        res.__class__ = myclArray
        res.reinit()
    return res

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

    @chkmethod
    def reshape(self, *shape, **kwargs):
        res = clarray.Array.reshape(self, *shape, **kwargs)
        if not isinstance(res, myclArray):
            res.__class__ = myclArray
            res.reinit()
        return res

    def createshapes(self, index):
        if isinstance(index, slice):
            index = (index,)
        def getslice(_x, a):
            if isinstance(_x, slice):
                x = slice(int(_x.start) if isinstance(_x.start, float) else _x.start,\
                          int(_x.stop)  if isinstance(_x.stop,  float) else _x.stop,\
                          int(_x.step)  if isinstance(_x.step,  float) else _x.step)
                if x.step and x.step<0:
                    return slice(x.start, x.stop, -x.step).indices(a)[:2]+(x.step,)
                else:
                    return x.indices(a)
            elif isinstance(_x, int):
                return slice(_x, _x+1).indices(a)
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


    @chkmethod
    def __getitem__(self, index):
        if isinstance(index, myclArray) and index.is_boolean == True:
            x, y, z = algorithm.copy_if(self.reshape((self.size,)), "index[i]!=0", [("index", index.reshape((index.size,)))])
            res = x[:y.get()]
        elif isinstance(index, tuple) or isinstance(index, slice):
            indices, newshape = self.createshapes(index)
            program = programs.sliceget(self.dtype, len(self.shape))
            res = empty(newshape, self.dtype)
            program.mislice(queue, (res.size,), None, indices.data, self.data, res.data)
            return res
        elif isinstance(index, myclArray) and self.ndim>0:
            program = programs.getndbyids(self.dtype, index.dtype)
            dims = (int(np.prod(self.shape[1:])), int(index.size),)
            res = empty((index.size,) + self.shape[1:], self.dtype)
            program.getbyids(queue, dims, None, index.data, self.data, res.data)
            return res
        else:
            res = clarray.Array.__getitem__(self, index)
        if not isinstance(res, myclArray):
            res.__class__ = myclArray
            res.reinit()
        return res

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


    @chkvoidmethod
    def __setitem__(self, subscript, _value):
        def fix_val(_vl):
            if isinstance(_vl, myclArray):
                val = _vl
            elif type(_vl) in (int, float):
                val = arr_from_np(np.array([_vl], dtype=self.dtype))
            elif isinstance(_vl, np.ndarray):
                val = arr_from_np(_vl).astype(self.dtype)
            else:
                assert True==False, "Can not determine value type in setitem of {0}".format(_value)
            return val
        if isinstance(subscript, myclArray) and subscript.is_boolean == True:
            value = fix_val(_value)
            idxcl = get_arng(self.size)#clarray.arange(queue, 0, self.size, 1, dtype=np.int32)
            x, y, z = algorithm.copy_if(idxcl, "index[i]!=0", [("index", subscript.reshape((subscript.size,)))])
            if y:
                res = x[:y.get()]
                self.reshape((self.size,))[res] = value
        #elif isinstance(subscript, myclArray) and subscript.ndim > 1:
        #    clarray.Array.setitem(self.reshape(self.size), subscript.reshape(subscript.size), value, queue=queue)
        elif isinstance(subscript, tuple) or isinstance(subscript, slice):
            value = fix_val(_value)
            indices, newshape = self.createshapes(subscript)
            program = programs.sliceset(self.dtype, len(self.shape))
            result = empty(newshape, self.dtype)
            assert value.size == result.size or value.size == 1, "Size of value array {0} does not match size of result indices {1}"\
                                                                 .format(value.size, result.size)
            if value.size == result.size: 
                program.mislice(queue, (result.size,), None, indices.data, self.data, value.data)
            elif value.size == 1:
                program.mislicesingle(queue, (result.size,), None, indices.data, self.data, value.data)
        elif isinstance(_value, myclArray) and type(subscript) == int and self.shape[-_value.ndim:] == _value.shape:
            count = np.prod(self.shape[-_value.ndim:])
            s1 = count*subscript
            s2 = count*(subscript+1)
            #print(subscript, count, s1, s2)
            self.reshape(self.size)[s1:s2] = _value.reshape(_value.size)
        else:
            try:
                clarray.Array.setitem(self, subscript, _value, queue=queue)
            except:
                assert False==True, "Can not set array {0} by value {1} on [psition {2}".format(self, _value, subscript)
        #return self

    @chkmethod
    def __sub__(self, other):
        return meta_add(self, other, ('sub',))

    @chkvoidmethod
    def __isub__(self, other):
        return meta_add(self, other, ('i', 'sub',))

    @chkmethod
    def __add__(self, other):
        return meta_add(self, other, ('add',))

    @chkvoidmethod
    def __iadd__(self, other):
        return meta_add(self, other, ('i', 'add',))

    @chkmethod
    def __mul__(self, other):
        return meta_add(self, other, ('mul', ))

    @chkvoidmethod
    def __imul__(self, other):
        return meta_add(self, other, ('i', 'mul', ))

    @chkmethod
    def max(*args, **kwargs):
        a = args[0]
        if a.ndim==0 or not 'axis' in kwargs.keys():
            res = clarray.max(a, queue=queue) #np.sum(*args, **kwargs)
            if not isinstance(res, myclArray):
                res.__class__ = myclArray
                res.reinit()
            return res
        else:
            kwargs['prg2load'] = programs.max
            return _sum(*args, **kwargs)

    @chkmethod
    def min(*args, **kwargs):
        a = args[0]
        if a.ndim==0 or not 'axis' in kwargs.keys():
            res = clarray.min(a, queue=queue) #np.sum(*args, **kwargs)
            if not isinstance(res, myclArray):
                res.__class__ = myclArray
                res.reinit()
            return res
        else:
            kwargs['prg2load'] = programs.min
            return _sum(*args, **kwargs)

    @chkmethod
    def sum(*args, **kwargs):
        a = args[0]
        if a.ndim==0 or not 'axis' in kwargs.keys():
            res = clarray.sum(a, queue=queue) #np.sum(*args, **kwargs)
            if not isinstance(res, myclArray):
                res.__class__ = myclArray
                res.reinit()
            return res
        else:
            kwargs['prg2load'] = programs.sum
            return _sum(*args, **kwargs)

    @justtime        
    def flatten(self):
        return self.ravel()

#randomeer.uniform(queue, (10,2,), np.float32, a=-0.5, b=0.5)
#np.random.uniform(-0.5, 0.5, (10, 2))

class myrandom():
    def __init__(self):
        #np.random.__init__(self)
        self.randomeer = clrandom.RanluxGenerator(queue)
    @justtime        
    def random(self, size):
        res = clrandom.rand(queue, size, np.float32, a=0.0, b=1.0)
        res.__class__ = myclArray
        res.reinit()
        return res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    @justtime        
    def uniform(self, low=0.0, high=1.0, size=1):
        res = self.randomeer.uniform(queue, size, np.float32, a=low, b=high)
        res.__class__ = myclArray
        res.reinit()
        return res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    @justtime        
    def randint(self, low, high=1.0, size=1):
        res = clrandom.rand(queue, size, np.int32, a=low, b=high)
        res.__class__ = myclArray
        res.reinit()
        return res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    @justtime        
    def rand(self, *args):
        dtype=np.float32
        res = clrandom.rand(queue, args, dtype, a=0.0, b=1.0)
        res.__class__ = myclArray
        res.reinit()
        return res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    @justtime        
    def randn(self, shape, dtype=np.float32):
        res = clrandom.rand(queue, shape, dtype, a=-1.0, b=1.0)
        res.__class__ = myclArray
        res.reinit()
        return res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)

@justtime        
def arr_from_np(nparr):
    if nparr.dtype == np.object:
        nparr = np.concatenate(nparr)
    buf = myBuffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=nparr)
    return myclArray(queue, nparr.shape, nparr.dtype, data=buf)

random = myrandom()

#def argmin(*args, **kwargs):
#    return arr_from_np(np.argmin(*args, **kwargs))


@chkfunc
def concatenate(arrays, axis=0):
    res = clarray.concatenate(arrays, axis, queue)#np.concatenate(*args, **kwargs)
    res.__class__ = myclArray
    res.reinit()
    return res

@chkfunc
def dot(a, b, out=None):
    assert a.shape[-1] == b.size, "Sizes does not match, {0} vs {1}".format(a.shape[-1], b.size)
    prg = programs.dot(a.dtype, a.shape[-1])
    res = empty(a.shape[:-1], dtype=a.dtype)
    prg.midot(queue, (res.size,), None, a.data, b.data, res.data)
    #TODO: work with out
    #_res = clarray.dot(a, b)#np.dot(*args, **kwargs)
    #print("Dot result")
    #print(_res)
    #_res.__class__ = myclArray
    #res = _res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    #res.reinit()
    return res
    

@chkfunc
def floor(a, out=None):
    #TODO: work with out
    res = clmath.floor(a, queue=queue) #np.floor(*args, **kwargs)
    res.__class__ = myclArray
    res.reinit()
    return res


@chkfunc
def isneginf(a, out=None):
    program = programs.isinf(a.dtype)
    if out:
        program.isneginf(queue, (a.size,), None, a.data, out.data)
        out.is_boolean = True
        return out
    else:
        res = empty(a.shape, dtype=np.uint32)
        #res = clarray.empty_like(a)
        program.isneginf(queue, (a.size,), None, a.data, res.data)
        res.is_boolean = True
        return res
    #return np.isneginf(*args, **kwargs)


@chkfunc
def ones_like(a, dtype=np.float32, order='K', subok=True):
    res = empty(a.shape, dtype=(dtype or a.dtype))
    res.fill(1, queue=queue)
    return res


@chkfunc
def row_stack(*args, **kwargs):
    return arr_from_np(np.row_stack(*args, **kwargs))


@chkfunc
def tanh(a, out=None):
    #TODO: work with out
    res = clmath.tanh(a, queue=queue) #np.tanh(*args, **kwargs)
    res.__class__ = myclArray
    res.reinit()
    return res


@chkfunc
def all(a, axis=None, out=None, keepdims=False):
    #TODO: work with axis, out, keepdims
    return a.all(queue=queue) #np.all(*args, **kwargs)


@chkfunc
def asfarray(a, dtype=np.float32):
    if isinstance(a, myclArray):
        return a.astype(dtype, queue=queue)
    else:
        return array(a, dtype=dtype)


@chkfunc
def exp(a, out=None):
    #TODO: work with out
    res = clmath.exp(a, queue=queue) #np.exp(*args, **kwargs)
    res.__class__ = myclArray
    res.reinit()
    return res


@chkfunc
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


@chkfunc
def min(a):
    return a.min()#np.min(*args, **kwargs)


@chkfunc
def sqrt(a, out=None):
    #TODO: work with out
    res = clmath.sqrt(a, queue=queue) #np.sqrt(*args, **kwargs)
    res.__class__ = myclArray
    res.reinit()
    return res


@justtime
def values(*args, **kwargs):
    return arr_from_np(np.values(*args, **kwargs))


@chkfunc
def isinf(a, out=None):
    program = programs.isinf(a.dtype)
    if out:
        program.isposinf(queue, (a.size,), None, a.data, out.data)
        out.is_boolean = True
        return out
    else:
        res = empty(a.shape, dtype=np.uint32)
        program.isposinf(queue, (a.size,), None, a.data, res.data)
        res.is_boolean = True
        return res
    #return np.isinf(*args, **kwargs)


@justtime
def items(*args, **kwargs):
    return np.items(*args, **kwargs)

@chkfunc
def max(a):
    return a.max()#np.max(*args, **kwargs)

@chkfunc
def abs(*args, **kwargs):
    arr = args[0]
    if isinstance(arr, myclArray):
        return arr.__abs__()
    else:
        return arr_from_np(np.abs(*args, **kwargs))

@justtime
def empty(shape, dtype=np.float32):
    #return arr_from_np( np.empty(*args, **kwargs) )
    return myclArray(queue, shape, dtype)

@chkfunc
def square(a, out=None):
    #TODO: work with out
    return a*a #np.square(*args, **kwargs)


@chkfunc
def sign(a, out=None):
    program = programs.sign(a.dtype)
    if out:
        program.asign(queue, (a.size,), None, a.data, out.data)
        return out
    else:
        res = empty(a.shape, dtype=a.dtype)
        if not isinstance(res, myclArray):
            res.__class__ = myclArray
            res.reinit()
        program.asign(queue, (a.size,), None, a.data, res.data)
        return res


@chkfunc
def zeros_like(a, dtype=None, order='K', subok=True):
    res = clarray.zeros_like(a)
    res.__class__ = myclArray
    res.reinit()
    return res

@chkfunc
def argmin(a):
    return argsort(a)[0]
@chkfunc
def argmax(a):
    return argsort(a)[-1]
@chkfunc
def argsort(a):
    arng = get_arng(a.size, np.uint32)#clarray.arange(queue, 0, a.size, 1, dtype=np.int32)
    prg = programs.argsort(a.dtype)
    res = prg(a, arng, key_bits=a.dtype.itemsize*8)[0][1]
    if not isinstance(res, myclArray):
        res.__class__ = myclArray
        res.reinit()
    #print(ret)
    return res


@chkfunc
def sum(*args, **kwargs):
    kwargs['prg2load'] = programs.sum
    return _sum(*args, **kwargs)

@justtime
def _sum(a, axis=None, dtype=None, out=None, prg2load=programs.sum):
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
    program.misum(queue, (int(a.size//a.shape[axis]),), None, cltrresult, result.data)
    return result

@chkfunc
def sin(arr):
    #TODO: work with axis, out, keepdims
    res = clmath.sin(arr, queue=queue) #np.sum(*args, **kwargs)
    res.__class__ = myclArray
    res.reinit()
    return res


@chkfunc
def zeros(shape, dtype=np.float32, order='C'):
    res = clarray.zeros(queue, shape, dtype, order)
    res.__class__ = myclArray
    res.reinit()
    return res

@chkfunc
def array(*args, **kwargs):
    if not 'dtype' in kwargs.keys():
        kwargs['dtype'] = np.float32
    return arr_from_np( np.array(*args, **kwargs) )
