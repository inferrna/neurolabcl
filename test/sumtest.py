import numpy as np
import pyopencl as cl
from pyopencl import array

arr = np.random.randint(55, size=(55, 77, 88)).astype(np.uint32)
rarr = arr.reshape((arr.size,))


gpusrc = """
#define dtype uint
#define PC 3 //Dimensions count

uint slice(uint id, __global uint4 *params, uint c){
    uint N = params[c].s0;
    uint x = params[c].s1;
    uint y = params[c].s2;
    uint d = params[c].s3;
    //printf("N=%d\\n", N);
    uint ipg = 1+(min(N, y)-(x%N)-1)/d;
    uint s = x/N;
    uint group = s+id/ipg;
    if(c>0) group = slice(group, params, c-1);
    uint groupstart = group*N;
    uint cmd = id%ipg;
    uint groupid = x%N+cmd*d;
    return  groupid+groupstart;
}

__kernel void misum(__global uint4 *params, __global dtype *data, __global dtype *result){
    uint gid = get_global_id(0);
    result[gid] = data[slice(gid, params, PC-1)];
}
"""

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags


run = cl.Program(ctx, gpusrc).build()
dvdr = np.cumsum(np.array(arr.shape[:axis], dtype=np.uint32))
#params = np.array([[8, 2, 2, 3], [7, 3, 1, 4], [5, 2, 0, 5]], dtype=np.uint32)
summ = arr.sum(axis=axis)
clarr = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=arr.reshape((arr.size,)))
clparams = cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=params)
clresult = cl.Buffer(ctx, mf.READ_WRITE, sliced.nbytes)

result = np.empty(sliced.size).astype(np.uint32)

run.mislice(queue, (sliced.size,), None, clparams, clarr, clresult)
cl.enqueue_copy(queue, result, clresult)
print(result)









