import time
from numpy import ndarray


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

class chkmethod():
    def __init__(self, *args):
        func = args[0]
        self.func = func
        self.npfunc = ndarray.__dict__[func.__name__]
        self.bage = [func.__name__, 0.0, 0, 0]
    def __call__(self, *args, **kw):
        print(self, args, kw)
        ts = time.time()
        result = self.func(obj, *args, **kw)
        te = time.time()
        newargs = [obj.get()] + args[1:]
        npres = self.npfunc(*newargs, **kw)
        clres = result.get()
        if isinstance(npres, ndarray):
            print(((abs(clres-npres))>0.00001).all())
        else:
            print((abs(clres-npres))>0.00001)
        self.bage[1] += te-ts
        self.bage[2] += 1
        return result
    def __del__(self):
        if self.bage[2]: 
            self.bage[3] = self.bage[1]/self.bage[2]
        print("Function myArray.{0}. Total time {1}, total calls {2}. Average time {3}.".format(*self.bage))

