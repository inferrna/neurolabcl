import mynp as np
import time
import bitonic_templates
from mako.template import Template

#np.cl.Program(np.ctx, tplsrc).build()
defstpl = Template(bitonic_templates.defines)
sz = pow(2, 10)
arr = np.random.randn(2, sz, 3)
out = np.empty(sz, dtype=arr.dtype)
arrc = arr.get()

sa = 1 #Sort axis

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
                                           dtype=params[2], idxtype=params[3],\
                                           dsize=params[4], nsize=params[5])
            cached_defs[params] = defs
        kid = Template(kernels_srcs[letter]).render(argsort=argsort)
        prg = np.cl.Program(np.ctx, defs + kid).build()
        cached_progs[letter][params] = prg
        if letter=='B2':
            print(kid)
        return prg

def sort_b(arr, axis, idx):
    n = arr.shape[axis]
    ds = arr.shape[axis]
    m = int(np.np.prod(arr.shape)/arr.shape[axis])
    ns = np.np.prod(arr.shape[(axis+1):]) if axis<arr.ndim-1 else 1
    ns = int(ns)
    ds = int(ds)
    allowb4  = True
    allowb8  = False 
    allowb16 = False 
    length = 1
    while length<n:
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
            nThreads = (arr.size) >> ninc;
            print("dsize == {0}, nsize == {1}, nThreads == {2}".format(ds, ns, nThreads))
            wg = np.ctx.devices[0].max_work_group_size
            wg = min(wg,256)
            wg = min(wg,nThreads)
            prg = get_program(letter, (inc, direction, 'float', 'uint',  ds, ns))
            if argsort:
                prg.run(np.queue, (nThreads,), (wg,), arr.data, idx.data)
            else:
                prg.run(np.queue, (nThreads,), (wg,), arr.data)
            np.cl.enqueue_barrier(np.queue)
            inc >>= ninc;
        length<<=1

def sort_c4(arr, idx):
    n = arr.size
    allowb4  = True 
    allowb8  = True 
    allowb16 = True 
    length = 1
    while length<n:
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
            wg = np.ctx.devices[0].max_work_group_size
            wg = min(wg,256)
            wg = min(wg,nThreads)
            #prg = np.cl.Program(np.ctx, defs + kid).build()
            prg = get_program(letter, (inc, direction, 'float', 'uint',))
            if doLocal>0:
                localmemx = np.cl.LocalMemory(wg*doLocal*arr.dtype.itemsize)
                if argsort:
                    localmemy = np.cl.LocalMemory(wg*doLocal*indexes.dtype.itemsize)
                    prg.run(np.queue, (nThreads,), (wg,), arr.data, idx.data, localmemx, localmemy)
                else:
                    prg.run(np.queue, (nThreads,), (wg,), arr.data, localmemx)
            else:
                if argsort:
                    prg.run(np.queue, (nThreads,), (wg,), arr.data, idx.data)
                else:
                    prg.run(np.queue, (nThreads,), (wg,), arr.data)
            np.cl.enqueue_barrier(np.queue)
            #c->enqueueBarrier(targetDevice); // sync
            # if (mLastN != n) printf("LENGTH=%d INC=%d KID=%d\n",length,inc,kid); // DEBUG
            if ninc < 0: break
            inc >>= ninc
        length<<=1
        #print("length =", length)

sort_b(arr.copy(), sa, indexes.copy())
tsg = time.time()
sort_b(arr, sa, indexes)
teg = time.time()

print("Sorting {0} samples. Got {1} sec on CPU and {2} sec on GPU".format(sz, tec - tsc, teg - tsg))

#singleprogram.ParallelBitonic_C4(np.queue, (arr.size,), None, arr.data, np.np.int32(32), np.np.int32(0), localmem)
#singleprogram.ParallelMerge_Local(np.queue, (arr.size,), None, arr.data, out.data, localmem)

