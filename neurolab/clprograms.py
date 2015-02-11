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
            nds = ndim-1
            slicesrcl = [clsrc.norecslicesrc.format("", "", min(1, nds), nds)]+\
                        [clsrc.norecslicesrc.format(a, "", a+1, nds-a) for a in range(1, nds)]+\
                        [clsrc.norecslicesrc.format(nds, "//", 0, 0)]
            slicesrcl.reverse()
            slicesrc = "\n".join(slicesrcl).replace("<%", "{").replace("%>", "}")+"\n"
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], ndim) + slicesrc + clsrc.slicesetsrc
            programcache[args] = cl.Program(self.ctx, ksource).build()
        return programcache[args]

    def sliceget(self, *args):
        if not args in programcache.keys():
            dtype, ndim, oper = args
            nds = ndim-1
            slicesrcl = [clsrc.norecslicesrc.format("", "", min(1, nds), nds)]+\
                        [clsrc.norecslicesrc.format(a, "", a+1, nds-a) for a in range(1, nds)]+\
                        [clsrc.norecslicesrc.format(nds, "//", 0, 0)]
            slicesrcl.reverse()
            slicesrc = "\n".join(slicesrcl).replace("<%", "{").replace("%>", "}")+"\n"
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], ndim) + slicesrc + clsrc.slicegetsrc
            programcache[args] = cl.Program(self.ctx, ksource).build()
        return programcache[args]

    def transpose(self, *args):
        if not args in programcache.keys():
            dtype, ndim, oper = args
            nds = ndim-1
            findpsrcl = [clsrc.findposnorecsrc.format("", "", min(1, nds), nds)]+\
                        [clsrc.findposnorecsrc.format(a, "", a+1, nds-a) for a in range(1, nds)]+\
                        [clsrc.findposnorecsrc.format(nds, "//", 0, 0)]
            findpsrcl.reverse()
            findpsrc = "\n".join(findpsrcl).replace("<%", "{").replace("%>", "}")+"\n"
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], ndim) + findpsrc + clsrc.transpsrc
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
