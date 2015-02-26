import time
from numpy import ndarray
import numpy as np
from pyopencl import array 

class collector():
    def __init__(self, funcname): 
        self.bage = [funcname, 0.0, 0, 0.0]
    def __del__(self):
        if self.bage[2]: self.bage[3] = self.bage[1]/self.bage[2]
        print("Function {0}. Total time {1}, total calls {2}. Average time {3}.".format(*self.bage))


def justtime(func):
    bage = collector(func.__name__)
    def wrapper(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()
        bage.bage[1] += te-ts
        bage.bage[2] += 1
        return result
    return wrapper

def convertinst(inst, varbls):
    newvs = []
    for varbl in varbls:
        if isinstance(varbl, inst):
            v = varbl.get()
            if varbl.is_boolean:
                newvs.append(v>0)
            else:
                newvs.append(v)
        else:
            newvs.append(varbl)
    return tuple(newvs)

def chkvoidmethod(func):
    npfunc = ndarray.__dict__[func.__name__]
    def wrapper(*args, **kw):
        result = func(*args, **kw)
        newargs = convertinst(array.Array, args)
        npfunc(*newargs, **kw)
        npres = newargs[0] 
        clres = args[0].get()
        tst = False
        if isinstance(npres, ndarray):
            tst = ((abs(clres-npres))<0.00001).all()
        else:
            tst = (abs(clres-npres))<0.00001
        assert tst==True, "Error in void method {2}. Result from cl \n{0}\n does not equal result from np\n{1}. Args was {3}"\
                          .format(clres, npres, func.__name__, args)
        return result
    return wrapper

def chkmethod(func):
    npfunc = ndarray.__dict__[func.__name__]
    def wrapper(*args, **kw):
        result = func(*args, **kw)
        newargs = convertinst(array.Array, args)
        #print("wrapper "+func.__name__, newargs, kw)
        npres = npfunc(*newargs, **kw)
        clres = result.get()
        #print(npres)
        #print(clres)
        tst = False
        if isinstance(npres, ndarray):
            tst = ((abs(clres-npres))<0.00001).all()
        else:
            tst = (abs(clres-npres))<0.00001
        assert tst==True, "Error in method {2}. Result from cl \n{0}\n does not equal result from np\n{1}. Args was {3}"\
                          .format(clres, npres, func.__name__, args)
        return result
    return wrapper

def chkfunc(func):
    npfunc = np.__dict__[func.__name__]
    def wrapper(*args, **kw):
        #print("wrapper "+func.__name__, args, kw)
        result = func(*args, **kw)
        newargs = convertinst(array.Array, args)
        npres = npfunc(*newargs, **kw)
        clres = result.get()
        #print(npres)
        #print(clres)
        if isinstance(npres, ndarray):
            tst = ((abs(clres-npres))<0.00001).all()
        else:
            tst = (abs(clres-npres))<0.00001
        assert tst==True, "Error in func {2}. Result from cl \n{0}\n does not equal result from np\n{1}. Args was {3}"\
                          .format(clres, npres, func.__name__, args)
        return result
    return wrapper

