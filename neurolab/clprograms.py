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
np.float64.__name__: "double",
np.dtype('bool').name: "char"}

operators = {
    'sub': '-',
    'add': '+',
    'mul': '*',
    'truediv': '/',
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
            dtype, ndim, cs, noidx = args          # cs as chunk size. Usual size of value if it shape may be fitted into source shapes
            nds = ndim-1
            if noidx:
                slicesrcl = ''
            elif nds>0:
                slicesrcl = [clsrc.norecslicesrc.format("", "", min(1, nds), nds)]+\
                            [clsrc.norecslicesrc.format(a, "", a+1, nds-a) for a in range(1, nds)]+\
                            [clsrc.norecslicesrc.format(nds, "//", 0, 0)]
                slicesrcl.reverse()
            else:
                slicesrcl = [clsrc.norecslicesrc.format("", "//", 0, nds)]
            slicesrc = "\n".join(slicesrcl).replace("<%", "{").replace("%>", "}")+"\n"
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0, ndim)\
                                            + slicesrc\
                                            + Template(clsrc.slicesetsrc).render(cs=cs, noidx=noidx)
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def sliceget(self, *args):
        key = args+('sliceget',)
        if not key in programcache.keys():
            dtype, ndim, noidx = args
            nds = ndim-1
            if noidx:
                slicesrcl = ''
            elif nds>0:
                slicesrcl = [clsrc.norecslicesrc.format("", "", min(1, nds), nds)]+\
                            [clsrc.norecslicesrc.format(a, "", a+1, nds-a) for a in range(1, nds)]+\
                            [clsrc.norecslicesrc.format(nds, "//", 0, 0)]
                slicesrcl.reverse()
            else:
                slicesrcl = [clsrc.norecslicesrc.format("", "//", 0, nds)]
            slicesrc = "\n".join(slicesrcl).replace("<%", "{").replace("%>", "}")+"\n"
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0, ndim) + slicesrc + Template(clsrc.slicegetsrc).render(noidx=noidx)
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def transpose(self, *args):
        key = args+('transpose',)
        if not key in programcache.keys():
            dtype, shape, repls = args
            ndim = len(shape)
            scales = [0]*ndim
            scales[ndim-1] = 1
            for i in range(ndim-1, 0, -1):
                scales[i-1] = scales[i]*shape[repls[i]]
            ksourcetpl = Template(clsrc.slicedefs.format(typemaps[dtype.name], 0, ndim) + clsrc.transpsrc)
            ksource = ksourcetpl.render(scales=scales, replaces=repls, olddims=shape, ndim=ndim)
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def delete(self, *args):
        key = args+('delete',)
        if not key in programcache.keys():
            ndims, dimr, rc, dtype, olddims, newdims = args
            dtypecl = typemaps[dtype.name]
            ksourcetpl = Template(clsrc.slicedefs.format(dtypecl, dtypecl, 0) + clsrc.deletesrc)
            ksource = ksourcetpl.render(ndims=ndims, dimr=dimr, rc=rc, olddims=olddims, newdims=newdims)
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]

    def singleset(self, dtype):
        key = (dtype, 'singleset',)
        if not key in programcache.keys():
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0, 0) + clsrc.singlesetsrc
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
            idtypecl = dtypecl
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

    def setndbyids(self, valsz, dtype, idtype):
        key = (dtype, idtype, 'setndbyids',)
        if not key in programcache.keys():
            ksourcetpl = Template(clsrc.slicedefs.format(typemaps[dtype.name], typemaps[idtype.name], 0) + clsrc.setbyidssrc)
            ksource = ksourcetpl.render(cs=valsz)
            programcache[key] = cl.Program(self.ctx, ksource).build()
        return programcache[key]
    def setif(self, valsz, dtype, idtype):
        key = (dtype, idtype, 'setif',)
        if not key in programcache.keys():
            ksourcetpl = Template(clsrc.slicedefs.format(typemaps[dtype.name], typemaps[idtype.name], 0) + clsrc.setifsrc)
            ksource = ksourcetpl.render(cs=valsz)
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
            programcache[key] = cl.algorithm.RadixSort(self.ctx, typemaps[dtype.name]+" *mkey, int *tosort",\
                                                                         "mkey[i]", ["mkey", "tosort"], key_dtype=np.int32)
        return programcache[key]

    def dot(self, *args):
        key = args+('dot',)
        if not key in programcache.keys():
            dtype, nums = args
            ksource = clsrc.slicedefs.format(typemaps[dtype.name], 0, nums) + clsrc.doubledotsrc;
            programcache[key] = cl.Program(self.ctx, ksource).build()
            #print(ksource)
        return programcache[key]
