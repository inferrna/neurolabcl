import mynp as np
tplsrc = open("SortKernels.cl", "r").read()

singleprogram = np.cl.Program(np.ctx, tplsrc).build()
localmem = np.cl.LocalMemory(1024*4)
arr = np.random.randn(1024)
out = np.empty(1024, dtype=arr.dtype)
arrc = arr.copy()
print(arr)
arrs = np.np.sort(arr.get())

def sort_c4(arr):
    n = arr.size
    allowb4  = False 
    allowb8  = False 
    allowb16 = False 
    length = 1
    while length<n:
        inc = length
        strategy = []
        ii = inc
        while ii > 0:
            if ii in [126, 32, 8]:
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
            print("ii =", ii)
        length<<=1
        print("length =", length)
    while inc > 0:
        ninc = 0;
        kid = -1;
        doLocal = 0;
        nThreads = 0;
        d = strategy.pop(0);

        if d == -1:
          kid = singleprogram.ParallelBitonic_C4 #PARALLEL_BITONIC_C4_KERNEL;
          ninc = -1; # reduce all bits
          doLocal = 4;
          nThreads = n >> 2;
        elif d == 4:
          kid = singleprogram.ParallelBitonic_B16 #PARALLEL_BITONIC_B16_KERNEL;
          ninc = 4;
          nThreads = n >> ninc;
        elif d == 3:
          kid = singleprogram.ParallelBitonic_B8 #PARALLEL_BITONIC_B8_KERNEL;
          ninc = 3;
          nThreads = n >> ninc;
        elif d == 2:
          kid = singleprogram.ParallelBitonic_B4 #PARALLEL_BITONIC_B4_KERNEL;
          ninc = 2;
          nThreads = n >> ninc;
        elif d == 1:
          kid = singleprogram.ParallelBitonic_B2 #PARALLEL_BITONIC_B2_KERNEL;
          ninc = 1;
          nThreads = n >> ninc;
        else:
          print("Strategy error!");
        wg = np.ctx.devices[0].max_work_group_size
        wg = min(wg,256);
        wg = min(wg,nThreads);
        #c->clearArgs(kid);
        #c->pushArg(kid,out);
        #c->pushArg(kid,inc); // INC passed to kernel
        #c->pushArg(kid,length<<1); // DIR passed to kernel
        localmem = np.cl.LocalMemory(wg*doLocal*arr.dtype.itemsize)
        #c->enqueueKernel(targetDevice,kid,nThreads,1,wg,1,EventVector());
        if doLocal>0: kid(np.queue, (nThreads,), (wg,), arr.data, np.np.int32(inc), np.np.int32(length<<1), localmem)
        else: kid(np.queue, (nThreads,), (wg,), arr.data, np.np.int32(inc), np.np.int32(length<<1))
        np.cl.enqueue_barrier(np.queue)
        #c->enqueueBarrier(targetDevice); // sync
        # if (mLastN != n) printf("LENGTH=%d INC=%d KID=%d\n",length,inc,kid); // DEBUG
        if ninc < 0: break
        inc >>= ninc

        
sort_c4(arr)
#singleprogram.ParallelBitonic_C4(np.queue, (arr.size,), None, arr.data, np.np.int32(32), np.np.int32(0), localmem)
#singleprogram.ParallelMerge_Local(np.queue, (arr.size,), None, arr.data, out.data, localmem)

