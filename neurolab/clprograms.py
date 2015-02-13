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
        key = args+('sliceset',)
        if not key in programcache.keys():
            dtype, ndim = args
            nds = ndim-1
            slicesrcl = [clsrc.norecslicesrc.format("", "", min(1, nds), nds)]+\
                        [clsrc.norecslicesrc.format(a, "", a+1, nds-a) for a in range(1, nds)]+\
                        [clsrc.norecslicesrc.format(nds, "//", 0, 0)]
            slicesrcl.reverse()
            slicesrc = "\n".join(slicesrcl).replace("<%", "{").replace("%>", "}")+"\n"
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], ndim) + slicesrc + clsrc.slicesetsrc
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def sliceget(self, *args):
        key = args+('sliceget',)
        if not key in programcache.keys():
            dtype, ndim = args
            nds = ndim-1
            slicesrcl = [clsrc.norecslicesrc.format("", "", min(1, nds), nds)]+\
                        [clsrc.norecslicesrc.format(a, "", a+1, nds-a) for a in range(1, nds)]+\
                        [clsrc.norecslicesrc.format(nds, "//", 0, 0)]
            slicesrcl.reverse()
            slicesrc = "\n".join(slicesrcl).replace("<%", "{").replace("%>", "}")+"\n"
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], ndim) + slicesrc + clsrc.slicegetsrc
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def transpose(self, *args):
        key = args+('transpose',)
        if not key in programcache.keys():
            dtype, ndim = args
            nds = ndim-1
            findpsrcl = [clsrc.findposnorecsrc.format("", "", min(1, nds), nds)]+\
                        [clsrc.findposnorecsrc.format(a, "", a+1, nds-a) for a in range(1, nds)]+\
                        [clsrc.findposnorecsrc.format(nds, "//", 0, 0)]
            findpsrcl.reverse()
            findpsrc = "\n".join(findpsrcl).replace("<%", "{").replace("%>", "}")+"\n"
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], ndim) + findpsrc + clsrc.transpsrc
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def sum(self, *args):
        key = args+('sum',)
        if not key in programcache.keys():
            dtype, nsum = args
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], nsum) + clsrc.sumsrc
            #print(ksource)
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]
    def min(self, *args):
        key = args+('min',)
        if not key in programcache.keys():
            dtype, nsum = args
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], nsum) + clsrc.minsrc
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]
    def max(self, *args):
        key = args+('max',)
        if not key in programcache.keys():
            dtype, nsum  = args
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], nsum) + clsrc.maxsrc
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def singlesms(self, *args):
        key = args+('singlesms',)
        if not key in programcache.keys():
            dtype = args[0]
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0) +\
                                                   clsrc.singlesumsrc +\
                                                   clsrc.singlemulsrc +\
                                                   clsrc.singlesubsrc +\
                                                   clsrc.singlenegsubsrc
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]
