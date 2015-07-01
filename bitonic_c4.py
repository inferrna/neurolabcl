import mynp as np
import time
import bitonic_templates
from mako.template import Template

#np.cl.Program(np.ctx, tplsrc).build()
defstpl = Template(bitonic_templates.defines)
sz = pow(2, 15)
arr = np.random.randn(sz)
out = np.empty(sz, dtype=arr.dtype)
arrc = arr.get()

arrs = np.np.sort(arrc)
#arrc[99] = 0.199
tsc = time.time()
arrs = np.np.sort(arrc)
tec = time.time()
indexes = np.arange(sz)

cached_defs = {}
cached_progs = {'B2':{}, 'B4':{}, 'B8':{}, 'B16':{}, 'C4':{}}
kernels_srcs = {'B2': bitonic_templates.ParallelBitonic_B2,
                'B4': bitonic_templates.ParallelBitonic_B4,
                'B8': bitonic_templates.ParallelBitonic_B8,
                'B16':bitonic_templates.ParallelBitonic_B16,
                'C4': bitonic_templates.ParallelBitonic_C4}


def get_program(letter, params):
    if params in cached_progs[letter].keys():
        return cached_progs[letter][params]
    else:
        if params in cached_defs.keys():
            defs = cached_defs[params]
        else:
            defs = defstpl.render(NS="\\", inc=params[0], dir=params[1], dtype=params[2], idxtype=params[3])
            cached_defs[params] = defs
        kid = kernels_srcs[letter]
        prg = np.cl.Program(np.ctx, defs + kid).build()
        cached_progs[letter][params] = prg
        return prg


def sort_c4(arr):
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
            prg = get_program(letter, (inc, direction, 'float', 'uint'))
            if doLocal>0:
                localmemx = np.cl.LocalMemory(wg*doLocal*arr.dtype.itemsize)
                localmemy = np.cl.LocalMemory(wg*doLocal*indexes.dtype.itemsize)
                prg.run(np.queue, (nThreads,), (wg,), arr.data, indexes.data, localmemx, localmemy)
            else: 
                prg.run(np.queue, (nThreads,), (wg,), arr.data, indexes.data)
            np.cl.enqueue_barrier(np.queue)
            #c->enqueueBarrier(targetDevice); // sync
            # if (mLastN != n) printf("LENGTH=%d INC=%d KID=%d\n",length,inc,kid); // DEBUG
            if ninc < 0: break
            inc >>= ninc
        length<<=1
        #print("length =", length)

sort_c4(arr.copy())
tsg = time.time()        
sort_c4(arr)
teg = time.time()

print("Sorting {0} samples. Got {1} sec on CPU and {2} sec on GPU".format(sz, tec - tsc, teg - tsg))

#singleprogram.ParallelBitonic_C4(np.queue, (arr.size,), None, arr.data, np.np.int32(32), np.np.int32(0), localmem)
#singleprogram.ParallelMerge_Local(np.queue, (arr.size,), None, arr.data, out.data, localmem)

