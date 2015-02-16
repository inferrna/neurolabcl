import time
from numpy import ndarray
import numpy as np

class justtime():
    def __init__(self, func):
        self.func = func
        self.bage = [func.__name__, 0.0, 0]
    def __call__(self, *args, **kw):
        ts = time.time()
        result = self.func(*args, **kw)
        te = time.time()
        self.bage[1] += te-ts
        self.bage[2] += 1
        return result
    def __del__(self):
        self.bage.append(self.bage[1]/self.bage[2])
        print("Function {0}. Total time {1}, total calls {2}. Average time {3}.".format(*self.bage))

def convertinst(inst, varbls):
    newvs = []
    for varbl in varbls:
        if isinstance(varbl, inst):
            newvs.append(varbl.get())
        else:
            newvs.append(varbl)
    return tuple(newvs)

def chkmethod(func):
    npfunc = ndarray.__dict__[func.__name__]
    def wrapper(*args, **kw):
        #print("wrapper", args, kw)
        result = func(*args, **kw)
        newargs = convertinst(ndarray, args)
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

def chkfunc(func):
    npfunc = np.__dict__[func.__name__]
    def wrapper(*args, **kw):
        #print("wrapper", args, kw)
        result = func(*args, **kw)
        newargs = convertinst(ndarray, args)
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


