import clsrc
import pyopencl as cl
import numpy as np
from mako.template import Template

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

operators = {
    'sub': '-',
    'add': '+',
    'mul': '*',
    'div': '/',
    'ge':  '>=',
    'le':  '<=',
    'eq':  '==',
    'ne':  '!=',
    'lt':  '<',
    'gt':  '>'
}


class programs():
    def __init__(self, context):
        self.ctx = context

    def sliceset(self, *args):
        key = args+('sliceset',)
        if not key in programcache.keys():
            dtype, ndim = args
            nds = ndim-1
            if nds>0:
                slicesrcl = [clsrc.norecslicesrc.format("", "", min(1, nds), nds)]+\
                            [clsrc.norecslicesrc.format(a, "", a+1, nds-a) for a in range(1, nds)]+\
                            [clsrc.norecslicesrc.format(nds, "//", 0, 0)]
                slicesrcl.reverse()
            else:
                slicesrcl = [clsrc.norecslicesrc.format("", "//", 0, nds)]

            slicesrc = "\n".join(slicesrcl).replace("<%", "{").replace("%>", "}")+"\n"
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0, ndim) + slicesrc + clsrc.slicesetsrc
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def sliceget(self, *args):
        key = args+('sliceget',)
        if not key in programcache.keys():
            dtype, ndim = args
            nds = ndim-1
            if nds>0:
                slicesrcl = [clsrc.norecslicesrc.format("", "", min(1, nds), nds)]+\
                            [clsrc.norecslicesrc.format(a, "", a+1, nds-a) for a in range(1, nds)]+\
                            [clsrc.norecslicesrc.format(nds, "//", 0, 0)]
                slicesrcl.reverse()
            else:
                slicesrcl = [clsrc.norecslicesrc.format("", "//", 0, nds)]
            slicesrc = "\n".join(slicesrcl).replace("<%", "{").replace("%>", "}")+"\n"
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0, ndim) + slicesrc + clsrc.slicegetsrc
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
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0, ndim) + findpsrc + clsrc.transpsrc
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def sum(self, *args):
        key = args+('sum',)
        if not key in programcache.keys():
            dtype, nsum = args
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0, nsum) + clsrc.sumsrc
            #print(ksource)
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def sign(self, dtype):
        key = (dtype, 'sign',)
        if not key in programcache.keys():
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0, 0) + clsrc.signsrc
            #print(ksource)
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def isinf(self, dtype):
        key = (dtype, 'isinf',)
        if not key in programcache.keys():
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0, 0) + clsrc.isinfsrc
            #print(ksource)
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def min(self, *args):
        key = args+('min',)
        if not key in programcache.keys():
            dtype, nsum = args
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0, nsum) + clsrc.minsrc
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]
    def max(self, *args):
        key = args+('max',)
        if not key in programcache.keys():
            dtype, nsum  = args
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0, nsum) + clsrc.maxsrc
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def ndsms(self, *args):
        key = args+('ndsms',)
        if not key in programcache.keys():
            dtype, action = args
            dtypecl = typemaps[dtype.name]
            if action in ('lt', 'gt', 'le', 'ge', 'eq', 'ne',):
                idtypecl = 'char'
            else:
                idtypecl = dtypecl
            operator = operators[action]
            ksourcetpl = Template(clsrc.slicedefs.format(dtypecl, idtypecl, 0) + clsrc.ndsrc)
            ksource = ksourcetpl.render(action=action, operator=operator, idtype=idtypecl)
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def ndrsms(self, *args):
        key = args+('ndrsms',)
        if not key in programcache.keys():
            dtype, N, action = args
            dtypecl = typemaps[dtype.name]
            if action in ('lt', 'gt', 'le', 'ge', 'eq', 'ne',):
                idtypecl = 'char'
            else:
                idtypecl = dtypecl
            operator = operators[action]
            ksourcetpl = Template(clsrc.slicedefs.format(dtypecl, idtypecl, N) + clsrc.ndrsrc)
            ksource = ksourcetpl.render(action=action, operator=operator, idtype=idtypecl)
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def getndbyids(self, dtype, idtype):
        key = (dtype, idtype, 'getndbyids',)
        if not key in programcache.keys():
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], typemaps[idtype.name], 0) + clsrc.getbyidssrc
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def singlesms(self, *args):
        key = args+('singlesms',)
        if not key in programcache.keys():
            dtype, action, neg = args
            dtypecl = typemaps[dtype.name]
            if action in ('lt', 'gt', 'le', 'ge', 'eq', 'ne',):
                idtypecl = 'char'
            else:
                idtypecl = dtypecl
            operator = operators[action]
            ksourcetpl = Template(clsrc.slicedefs.format(dtypecl, idtypecl, 0) + clsrc.singlesrc)
            ksource = ksourcetpl.render(action=action, operator=operator, idtype=idtypecl, neg=neg)
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def argsort(self, dtype):
        key = (dtype, 'argsort',)
        if not key in programcache.keys():
            programcache[key] = cl.algorithm.RadixSort(self.ctx, typemaps[dtype.name]+" *mkey, unsigned int *tosort",\
                                                                         "mkey[i]", ["mkey", "tosort"])
        return programcache[key]

    def dot(self, *args):
        key = args+('dot',)
        if not key in programcache.keys():
            dtype, nums = args
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0, nums) + clsrc.smalldotsrc;
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]
