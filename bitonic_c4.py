import mynp as np
import time
import bitonic_templates
from mako.template import Template
from operator import mul
from functools import reduce

#np.cl.Program(np.ctx, tplsrc).build()
defstpl = Template(bitonic_templates.defines)
sz = pow(2, 20)
arr = np.random.randn(sz) #.astype(np.np.float64)
out = np.empty(sz, dtype=arr.dtype)
arrc = arr.get()

sa = 0 #Sort axis

arrs = np.np.sort(arrc, axis=sa)
#arrc[99] = 0.199
tsc = time.time()
arrs = np.np.sort(arrc, axis=sa)
tec = time.time()
indexes = np.arange(sz)

cached_defs = {}
cached_progs = {'B2':{}, 'B4':{}, 'B8':{}, 'B16':{}, 'C4':{}}
kernels_srcs = {'B2': bitonic_templates.ParallelBitonic_B2,
                'B4': bitonic_templates.ParallelBitonic_B4,
                'B8': bitonic_templates.ParallelBitonic_B8,
                'B16':bitonic_templates.ParallelBitonic_B16,
                'C4': bitonic_templates.ParallelBitonic_C4}

argsort=1

def get_program(letter, params):
    if params in cached_progs[letter].keys():
        return cached_progs[letter][params]
    else:
        if params in cached_defs.keys():
            defs = cached_defs[params]
        else:
            defs = defstpl.render(NS="\\", argsort=argsort, inc=params[0], dir=params[1],\
                                           dtype=params[2], idxtype=params[3])
            cached_defs[params] = defs
        kid = Template(kernels_srcs[letter]).render(argsort=argsort)
        prg = np.cl.Program(np.ctx, defs + kid).build()
        cached_progs[letter][params] = prg
        return prg

def sort_b_prepare(shape, axis):
    run_queue = []
    ds = int(shape[axis])
    size = reduce(mul, shape)
    ndim = len(shape)
    #m = int(size/shape[axis])
    ns = reduce(mul, shape[(axis+1):]) if axis<ndim-1 else 1
    allowb4  = True
    allowb8  = True
    allowb16 = True
    length = 1
    while length<ds:
        inc = length;
        while inc > 0:
            ninc = 0;
            direction = length<<1
            if allowb16 and inc >= 8 and ninc == 0:
                letter = 'B16'
                ninc = 4;
            elif allowb8 and inc >= 4 and ninc == 0:
                letter = 'B8'
                ninc = 3;
            elif allowb4 and inc >= 2 and ninc == 0:
                letter = 'B4'
                ninc = 2;
            elif inc >= 0:
                letter = 'B2'
                ninc = 1;
            nThreads = (size) >> ninc;
            print("dsize == {0}, nsize == {1}, nThreads == {2}".format(ds, ns, nThreads))
            prg = get_program(letter, (inc, direction, 'float', 'uint',  ds, ns))
            run_queue.append((prg, nThreads,))
            inc >>= ninc;
        length<<=1
    return run_queue

def sort_b_run(arr, rq, idx=None):
    if argsort:
        for p, nt in rq:
            p.run(np.queue, (nt,), None, arr.data, idx.data)
    else:
        for p, nt in rq:
            p.run(np.queue, (nt,), None, arr.data)


def sort_c4_prepare(shape, axis):
    run_queue = []
    size = shape[0]
    print("size", size)
    n = size
    ds = n
    ns = 1
    mwg = np.ctx.devices[0].max_work_group_size
    allowb4  = True 
    allowb8  = True 
    allowb16 = True 
    length = 1
    while length<size:
        inc = length
        strategy = []
        ii = inc
        while ii > 0:
            if ii in [128, 32, 8]:
                strategy.append(-1)
                break
            d = 1
            if ii==256: d = 1;
            elif ii==512 and allowb4: d = 2;
            elif ii==1024 and allowb8: d = 3;
            elif ii==2048 and allowb16: d = 4;
            elif ii>=8 and allowb16: d = 4;
            elif ii>=4 and allowb8: d = 3;
            elif ii>=2 and allowb4: d = 2;
            else: d = 1;
            strategy.append(d);
            ii >>= d
        while inc > 0:
            ninc = 0;
            kid = -1;
            doLocal = 0;
            nThreads = 0;
            d = strategy.pop(0)
            direction = length<<1
            #defs = defstpl.render(inc=inc, dir=direction)  #Define custom parameters
            if d == -1:
                letter = 'C4' #PARALLEL_BITONIC_C4_KERNEL;
                ninc = -1; # reduce all bits
                doLocal = 4;
                nThreads = n >> 2;
            elif d == 4:
                letter = 'B16' #PARALLEL_BITONIC_B16_KERNEL;
                ninc = 4;
                nThreads = n >> ninc;
            elif d == 3:
                letter = 'B8' #PARALLEL_BITONIC_B8_KERNEL;
                ninc = 3;
                nThreads = n >> ninc;
            elif d == 2:
                letter = 'B4' #PARALLEL_BITONIC_B4_KERNEL;
                ninc = 2;
                nThreads = n >> ninc;
            elif d == 1:
                letter = 'B2' #PARALLEL_BITONIC_B2_KERNEL;
                ninc = 1;
                nThreads = n >> ninc;
            else:
                print("Strategy error!");
            #print("inc ==", inc)
            wg = min(mwg,256)
            wg = min(wg,nThreads)
            prg = get_program(letter, (inc, direction, 'float', 'uint',  ds, ns))
            #if doLocal>0:
            #    localmemx = np.cl.LocalMemory(wg*doLocal*arr.dtype.itemsize)
            #    if argsort:
            #        localmemy = np.cl.LocalMemory(wg*doLocal*indexes.dtype.itemsize)
            #        prg.run(np.queue, (nThreads,), (wg,), arr.data, idx.data, localmemx, localmemy)
            #    else:
            #        prg.run(np.queue, (nThreads,), (wg,), arr.data, localmemx)
            #else:
            #    if argsort:
            #        prg.run(np.queue, (nThreads,), (wg,), arr.data, idx.data)
            #    else:
            #        prg.run(np.queue, (nThreads,), (wg,), arr.data)
            #np.cl.enqueue_barrier(np.queue)
            run_queue.append((prg, nThreads, wg, doLocal>0,))
            #c->enqueueBarrier(targetDevice); // sync
            # if (mLastN != n) printf("LENGTH=%d INC=%d KID=%d\n",length,inc,kid); // DEBUG
            if ninc < 0: break
            inc >>= ninc
        length<<=1
    return {'q': run_queue, 'mems': (np.cl.LocalMemory(mwg*4*4), np.cl.LocalMemory(mwg*4*4),)}
        #print("length =", length)

def sort_c4_run(arr, rqm, idx=None):
    rq = rqm['q']
    lx, ly = rqm['mems']
    if argsort:
        for p, nt, wg, dl in rq:
            if dl:
                p.run(np.queue, (nt,), (wg,), arr.data, idx.data, lx, ly)
            else:                           
                p.run(np.queue, (nt,), (wg,), arr.data, idx.data)
            np.cl.enqueue_barrier(np.queue)
    else:
        for p, nt, wg, dl in rq:
            if dl:
                p.run(np.queue, (nt,), (wg,), arr.data, lx)
            else:                           
                p.run(np.queue, (nt,), (wg,), arr.data)
            np.cl.enqueue_barrier(np.queue)

rq = sort_c4_prepare(arr.shape, sa)
tsg = time.time()
if argsort:
    sort_c4_run(arr, rq, indexes)
else:
    sort_c4_run(arr, rq)
teg = time.time()

print("Sorting {0} samples. Got {1} sec on CPU and {2} sec on GPU".format(sz, tec - tsc, teg - tsg))

#singleprogram.ParallelBitonic_C4(np.queue, (arr.size,), None, arr.data, np.np.int32(32), np.np.int32(0), localmem)
#singleprogram.ParallelMerge_Local(np.queue, (arr.size,), None, arr.data, out.data, localmem)

