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
            newvs.append(varbl.get())
        else:
            newvs.append(varbl)
    return tuple(newvs)

def chkvoidmethod(func):
    npfunc = ndarray.__dict__[func.__name__]
    def wrapper(*args, **kw):
        #print("wrapper", args, kw)
        newargs = convertinst(array.Array, args)
        func(*args, **kw)
        npfunc(*newargs, **kw)
        npres = newargs[0] 
        clres = args[0].get()
        #print(npres)
        #print(clres)
        tst = False
        if isinstance(npres, ndarray):
            tst = ((abs(clres-npres))<0.00001).all()
        else:
            tst = (abs(clres-npres))<0.00001
        assert tst==True, "Error in {2}. Result from cl \n{0}\n does not equal result from np\n{1}"\
                          .format(clres, npres, func.__name__)
    return wrapper

def chkmethod(func):
    npfunc = ndarray.__dict__[func.__name__]
    def wrapper(*args, **kw):
        #print("wrapper", args, kw)
        result = func(*args, **kw)
        newargs = convertinst(array.Array, args)
        npres = npfunc(*newargs, **kw)
        clres = result.get()
        #print(npres)
        #print(clres)
        tst = False
        if isinstance(npres, ndarray):
            tst = ((abs(clres-npres))<0.00001).all()
        else:
            tst = (abs(clres-npres))<0.00001
        assert tst==True, "Error in {2}. Result from cl \n{0}\n does not equal result from np\n{1}"\
                          .format(clres, npres, func.__name__)
        return result
    return wrapper

def chkfunc(func):
    npfunc = np.__dict__[func.__name__]
    def wrapper(*args, **kw):
        #print("wrapper", args, kw)
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
        assert tst==True, "Result from cl \n{0}\n does not equal result from np\n{1}"\
                          .format(clres, npres)
        return result
    return wrapper


