import clsrc
import pyopencl as cl
import numpy as np

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

class programs():
    def __init__(self, context):
        self.ctx = context

    def sliceset(self, *args):
        if not args in programcache.keys():
            dtype, ndim, oper = args
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], ndim) + clsrc.slicesrc + clsrc.slicesetsrc
            programcache[args] = cl.Program(self.ctx, ksource).build()
        return programcache[args]

    def sliceget(self, *args):
        if not args in programcache.keys():
            dtype, ndim, oper = args
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], ndim) + clsrc.slicesrc + clsrc.slicegetsrc
            programcache[args] = cl.Program(self.ctx, ksource).build()
        return programcache[args]

    def transpose(self, *args):
        if not args in programcache.keys():
            dtype, ndim, oper = args
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], ndim) + clsrc.transpsrc
            programcache[args] = cl.Program(self.ctx, ksource).build()
        return programcache[args]

    def sum(self, *args):
        if not args in programcache.keys():
            dtype, nsum, oper = args
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], nsum) + clsrc.sumsrc
            programcache[args] = cl.Program(self.ctx, ksource).build()
        return programcache[args]

    def singlesms(self, *args):
        if not args in programcache.keys():
            dtype, oper = args
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0) +\
                                                   clsrc.singlesumsrc +\
                                                   clsrc.singlemulsrc +\
                                                   clsrc.singlesubsrc +\
                                                   clsrc.singlenegsubsrc
            programcache[args] = cl.Program(self.ctx, ksource).build()
        return programcache[args]
