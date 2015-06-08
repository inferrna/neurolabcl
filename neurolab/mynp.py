import numpy as np
import pyopencl as cl
from pyopencl import clrandom, clmath
from pyopencl import array as clarray
from pyopencl import algorithm
import clsrc
import clprograms
from checker import justtime,  chkvoidmethod, chkmethod, chkfunc, backtonp_voidmethod, backtonp_method, backtonp_func
from builtins import sum as bsum

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
programs = clprograms.programs(ctx)
align = queue.device.get_info(cl.device_info.MEM_BASE_ADDR_ALIGN)
print("align is", align)
#random = np.random
Inf = np.Inf
mf = cl.mem_flags
arngd = np.array([0])
dtbool = np.dtype('bool')
float_ = np.float32
cl.tools.get_or_register_dtype(['bool'], dtype=dtbool)

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
    'lt': clarray.Array.__lt__,
    'gt': clarray.Array.__gt__,
    'le': clarray.Array.__le__,
    'ge': clarray.Array.__ge__,
    'eq': clarray.Array.__eq__,
    'ne': clarray.Array.__ne__,
    'isub': clarray.Array.__isub__,
    'sub':  clarray.Array.__sub__,
    'iadd': clarray.Array.__iadd__,
    'add':  clarray.Array.__add__,
    'imul': clarray.Array.__imul__,
    'mul':  clarray.Array.__mul__,
    #'itruediv': clarray.Array.__itruediv__,
    'truediv':  clarray.Array.__truediv__
}

        
def meta_add(arr, other, actnames, resdtype=None):
    actname = actnames[-1]
    # Original method, eg clarray.Array.__add__
    fallbackM = fallbacks[''.join(actnames)]
    if actnames[0] == 'i': 
        result = arr
        nores = False
    else: 
        nores = True
    if not resdtype: resdtype = arr.dtype

    if isinstance(other, myclArray) and not arr.shape == other.shape:
        neg = ''
        if arr.size == 1 and other.size>2:
            arr, other = other, arr
            if actname == 'sub':
                neg = '-'
                actname = 'add'
            if actname == 'truediv':
                neg = '1/'
                actname = 'mul'

        if other.offset:
            bsz  = other.size*other.dtype.itemsize
            if other.offset % align == 0:
                odata = other.base_data.get_sub_region(other.offset, bsz)
            else:
                odata = cl.Buffer(ctx, flags=cl.mem_flags.READ_WRITE, size=bsz)
                ev = cl.enqueue_copy(queue, odata, other.base_data, src_offset=other.offset, byte_count=bsz)
                ev.wait() #May not be needed
        else:
            odata = other.data

        if arr.offset:
            adata = arr.base_data.get_sub_region(arr.offset, arr.size*arr.dtype.itemsize)
        else:
            adata = arr.data
        if other.size==1:
            if nores: result = empty(arr.shape, resdtype)
            singleprogram = programs.singlesms(arr.dtype, actname, neg).prg
            singleprogram(queue, (arr.size,), None, adata, result.data, odata)
            res = result
        elif other.size==1 and other.offset:
            res = fallbackM(arr, other.get()[0])
        elif arr.size == other.size:
            res = fallbackM(arr.reshape(arr.size), other.reshape(arr.size)).reshape(arr.shape)
        elif arr.shape[-other.ndim:] == other.shape:
            #print("arr.shape[-other.ndim:] == other.shape case")
            if nores: result = empty(arr.shape, resdtype)
            s1 = np.prod(arr.shape[:-other.ndim])
            s2 = np.prod(other.shape)
            ndprogram = programs.ndsms(arr.dtype, actname).prg
            ndprogram(queue,\
                      tuple([int(s1), int(s2)]),\
                      None,\
                      adata,\
                      result.data,\
                      odata)
            res = result
        elif arr.shape[:other.ndim] == other.shape:
            if nores: result = empty(arr.shape, resdtype)
            N = np.prod(arr.shape[other.ndim:])
            ndrprogram = programs.ndrsms(arr.dtype, N, actname).prg
            ndrprogram(queue,\
                       (arr.size,),\
                       None,\
                       adata,\
                       result.data,\
                       odata)
            res = result
    else:
        #print("fallbackM with ", arr.dtype, type(other))
        if type(other).__name__ in ['float', 'int']:
            other = np.__dict__[type(other).__name__+'32'](other)
        res = fallbackM(arr, other)
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

    @chkmethod
    def __lt__(self, other):
        result = meta_add(self, other, ('lt',), resdtype=dtbool)
        result.dtype = dtbool
        return result
    @chkmethod
    def __le__(self, other):
        result = meta_add(self, other, ('le',), resdtype=dtbool)
        result.dtype = dtbool
        return result
    @chkmethod
    def __eq__(self, other):
        result = meta_add(self, other, ('eq',), resdtype=dtbool)
        result.dtype = dtbool
        return result
    @chkmethod
    def __ne__(self, other):
        result = meta_add(self, other, ('ne',), resdtype=dtbool)
        result.dtype = dtbool
        return result
    @chkmethod
    def __ge__(self, other):
        result = meta_add(self, other, ('ge',), resdtype=dtbool)
        result.dtype = dtbool
        return result
    @chkmethod
    def __gt__(self, other):
        result = meta_add(self, other, ('gt',), resdtype=dtbool)
        result.dtype = dtbool
        return result
    def __del__(self):
        self.base_data.nowners -=1
        if self.base_data.nowners == 0:
            #print("released", self.base_data.size, "bytes")
            self.base_data.release()

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
    def __truediv__(self, other):
        return meta_add(self, other, ('truediv', ), resdtype=float_)

    @chkvoidmethod
    def __itruediv__(self, other):
        return meta_add(self, other, ('i', 'truediv', ), resdtype=float_)

    @chkmethod
    def reshape(self, *shape, **kwargs):
        res = clarray.Array.reshape(self, *shape, **kwargs)
        if not isinstance(res, myclArray):
            res.__class__ = myclArray
            res.reinit()
        return res

    def tolist(self, *args, **kwargs):
        return self.get().tolist(*args, **kwargs)

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
        #print("npindices is", npindices)
        newshape = [1+(a[2]-a[1]-1)//a[3].__abs__() for a in npindices]
        newshape = tuple([a for a, b, i in zip(newshape, self.shape, index) if not isinstance(i, int)])
        if newshape == (): newshape = (1,)
        indices = arr_from_np(npindices)
        return indices, newshape


    @chkmethod
    def __getitem__(self, index):
        if isinstance(index, myclArray) and index.dtype == dtbool:
            x, y, z = algorithm.copy_if(self.reshape((self.size,)),\
                                        "index[i]!=0",\
                                        # TODO: avoid type convert
                                        [("index", bool2int(index).reshape((index.size,)))])
            res = x[:y.get()]
        elif isinstance(index, tuple) or isinstance(index, slice):
            indices, newshape = self.createshapes(index)
            program = programs.sliceget(self.dtype, len(self.shape))
            res = empty(newshape, self.dtype)
            program.mislice(queue, (res.size,), None, indices.data, self.data, res.data)
            return res
        elif isinstance(index, myclArray) and self.ndim>0:
            program = programs.getndbyids(self.dtype, index.dtype)
            resshape = (index.size,) + self.shape[1:] if index.size>1 else self.shape[1:]
            dims = (int(np.prod(self.shape[1:])), int(index.size),)
            res = empty(resshape, self.dtype)
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
                val = arr_from_np(np.array(_vl, dtype=self.dtype))
            elif isinstance(_vl, np.ndarray):
                val = arr_from_np(_vl).astype(self.dtype)
            else:
                assert True==False, "Can not determine value type in setitem of {0}".format(_value)
            return val
        if isinstance(subscript, myclArray):
            value = fix_val(_value)
            if subscript.dtype == dtbool:
                cs = 0 if value.size==self.size else value.size
                programs.setif(cs, self.dtype, subscript.dtype)\
                        .setif(queue, (self.size,), None, subscript.data, self.data, value.data)
            else:
                #valsz, dtype, idtype
                #print("going setndbyids")
                cs = int(np.prod(self.shape[1:]))
                # need Assert subscript.size <= cs
                programs.setndbyids(cs, self.dtype, subscript.dtype)\
                        .setbyids(queue, (cs, subscript.size,), None, subscript.data, self.data, value.data)
        #elif isinstance(subscript, myclArray) and subscript.ndim > 1:
        #    clarray.Array.setitem(self.reshape(self.size), subscript.reshape(subscript.size), value, queue=queue)
        elif isinstance(subscript, tuple) or isinstance(subscript, slice):
            value = fix_val(_value)
            indices, newshape = self.createshapes(subscript)
            newsize = int(np.prod(newshape))
            assert newshape[-value.ndim:] == value.shape or value.size == newsize or value.size == 1,\
                                     "Size of value array {0} does not match size of result indices {1}"\
                                                                 .format(value.shape, newshape)
            if value.size == newsize: 
                programs.sliceset(self.dtype, self.ndim, 1)\
                        .mislice(queue, (newsize,), None, indices.data, self.data, value.data)
            elif value.size == 1:
                programs.sliceset(self.dtype, self.ndim, 1)\
                        .mislicesingle(queue, (newsize,), None, indices.data, self.data, value.data, np.int32(0))
            elif newshape[-value.ndim:] == value.shape:
                sizes = (int(np.prod(newshape[:-value.ndim])), int(np.prod(value.shape)),)
                programs.sliceset(self.dtype, self.ndim - value.ndim, sizes[-1])\
                        .mislicesingle(queue, sizes, None, indices.data, self.data, value.base_data, np.int32(value.offset))
        elif isinstance(_value, myclArray) and type(subscript) == int and self.shape[-_value.ndim:] == _value.shape:
            count = int(np.prod(self.shape[-_value.ndim:]))
            programs.singleset(self.dtype)\
                    .prg(queue, (count,), None, self.data, _value.base_data, np.int32(0), global_offset=(count*subscript,))
        else:
            try:
                clarray.Array.setitem(self, subscript, _value, queue=queue)
            except:
                assert False==True, "Can not set array {0} by value {1} on [psition {2}".format(self, _value, subscript)
        #return self


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

#randomeer.uniform(queue, (10,2,), float_, a=-0.5, b=0.5)
#np.random.uniform(-0.5, 0.5, (10, 2))

class myrandom():
    def __init__(self):
        #np.random.__init__(self)
        self.randomeer = clrandom.RanluxGenerator(queue)
    def random(self, size=None):
        _size = size if size else 1
        res = clrandom.rand(queue, _size, float_, a=0.0, b=1.0)
        res.__class__ = myclArray
        res.reinit()
        return res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def uniform(self, low=0.0, high=1.0, size=1):
        res = self.randomeer.uniform(queue, size, float_, a=low, b=high)
        res.__class__ = myclArray
        res.reinit()
        return res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def randint(self, low, high=None, size=1):
        #_size, reshape = szs(size)
        if high:
            a, b = low, high
        else:
            a, b = 0, low
        res = clrandom.rand(queue, size, np.int32, a=a, b=b)
        res.__class__ = myclArray
        res.reinit()
        return res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def rand(self, *args):
        dtype=float_
        shape = args if len(args) else 1
        res = clrandom.rand(queue, shape, dtype, a=0.0, b=1.0)
        res.__class__ = myclArray
        res.reinit()
        return res #myclArray(queue, _res.shape, _res.dtype, data=_res.data)
    def randn(self, *args, dtype=float_):
        shape = args if len(args) else 1
        res = clrandom.rand(queue, shape, dtype, a=-1.0, b=1.0)
        res.__class__ = myclArray
        res.reinit()
        return res#myclArray(queue, _res.shape, _res.dtype, data=_res.data)

class vectorize(np.lib.function_base.vectorize):
    def __init__(self, *args, **kwargs):
        np.lib.function_base.vectorize.__init__(self, *args, **kwargs)
    def __call__(self, *args, **kwargs):
        return arr_from_np(np.lib.function_base.vectorize.__call__(self, *args, **kwargs))

ndarray = myclArray

@justtime        
def arr_from_np(nparr):
    if nparr.dtype == np.object:
        nparr = np.concatenate(nparr)
    buf = myBuffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=nparr)
    return myclArray(queue, nparr.shape, nparr.dtype, data=buf)
@justtime        
def bool2int(arr):
    return myclArray(queue, arr.shape, np.uint8, data=arr.data)

class nprandom():
    def random(self, *args, **kwargs):
        kwargs.update(dtype=float_)
        return arr_from_np( np.random.random(*args, **kwargs) )
    def uniform(self, *args, **kwargs):
        kwargs.update(dtype=float_)
        return arr_from_np( np.random.uniform(*args, **kwargs) )
    def randint(self, *args, **kwargs):
        kwargs.update(dtype=np.int32)
        return arr_from_np( np.random.randint(*args, **kwargs) )
    def rand(self, *args, **kwargs):
        return arr_from_np( np.random.rand(*args, **kwargs).astype(float_) )
    def randn(self, *args, **kwargs):
        return arr_from_np( np.random.randn(*args, **kwargs).astype(float_) )

random = myrandom()

@chkfunc
def ones(shape, dtype=None, order='C'):
    if not dtype: dtype = float_
    res = myclArray(queue, shape, dtype)
    res.fill(1)
    return res

@chkfunc
def delete(_arr, obj, axis=None):
    if type(obj) == int:
        rc = 1
        removed = np.uint32(obj)
    else:
        rc = len(obj)
        removed = array(obj, dtype=np.uint32)
    if axis==None:
        dimr = 0
        newshape = (_arr.size - rc,)
        arr = _arr.reshape(_arr.size)
    else:
        dimr = axis
        arr = _arr
        newshape = list(_arr.shape)
        newshape[axis] = newshape[axis]-rc
    dst = empty(shape=newshape, dtype=arr.dtype)
    prg = programs.delete(arr.ndim, dimr, rc, arr.dtype, str(arr.shape)[1:-1], str(newshape)[1:-1])
    prg.delete(queue, (dst.size,), None, removed.data, arr.data, dst.data)
    return dst

@chkfunc
def vstack(arrays):
    return concatenate(arrays, axis=0)


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
        out.dtype = dtbool
        return out
    else:
        res = empty(a.shape, dtype=dtbool)
        #res = clarray.empty_like(a)
        program.isneginf(queue, (a.size,), None, a.data, res.data)
        res.dtype = dtbool
        return res
    #return np.isneginf(*args, **kwargs)


@chkfunc
def ones_like(a, dtype=float_, order='K', subok=True):
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
def asfarray(a, dtype=float_):
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
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=float_):
    #TODO: create native function
    if num<2: return array([start])
    if endpoint:
        mnum = num-1
    else:
        mnum = num
    diff = (stop - start) / mnum
    if endpoint:
        stop = stop + diff
    res = clarray.arange(queue, start, stop, diff, dtype=float_)[:num]
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
        out.dtype = dtbool
        return out
    else:
        res = empty(a.shape, dtype=dtbool)
        program.isposinf(queue, (a.size,), None, a.data, res.data)
        res.dtype = dtbool
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
def empty(shape, dtype=float_):
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
def zeros(shape, dtype=float_, order='C'):
    res = clarray.zeros(queue, shape, dtype, order)
    res.__class__ = myclArray
    res.reinit()
    return res

@chkfunc
def array(*args, **kwargs):
    if not 'dtype' in kwargs.keys():
        kwargs['dtype'] = float_
    return arr_from_np( np.array(*args, **kwargs) )

@chkfunc
def asarray(*args, **kwargs):
    if not 'dtype' in kwargs.keys():
        kwargs['dtype'] = float_
    return arr_from_np( np.asarray(*args, **kwargs) )

