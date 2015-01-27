import numpy as np
import pyopencl as cl
from pyopencl import array


gpusrc = """
#define dtype uint
#define PC 3 //Dimensions count

void findpos(uint id, __global uint *olddims, uint *currposs, uint c){
    uint ipg = olddims[c];
    uint group = id/ipg;
    uint gid = id%ipg;
    currposs[c] = gid;
    if(c>0) findpos(group, olddims, currposs, c-1);
}

__kernel void mitransp(__global uint *olddims, __global uint *replaces, __global dtype *data, __global dtype *result){
    uint gid = get_global_id(0);
    uint i,j;
    uint currposs[PC];
    uint newposs[PC];
    uint newdims[PC];
    uint newid = 0;
    uint scales[PC];

    findpos(gid, olddims, currposs, PC-1);
    for(i=0; i<PC; i++){ //i as current dim
        j = replaces[i];
        newposs[j] = currposs[i];
        newdims[j] = olddims[i];
    }
    scales[PC-1] = 1;
    for(i=PC-1; i>0; i--){ //i as current dim
        scales[i-1] = scales[i]*newdims[i];
    }
    for(i=0; i<PC; i++){ //i as current dim
        newid += scales[i]*newposs[i];
    }
    result[newid] = data[gid];
}
"""

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

arr = np.random.randint(55, size=(3, 4, 5)).astype(np.uint32)
rarr = arr.reshape((arr.size,))

run = cl.Program(ctx, gpusrc).build()
replaces = np.array([1,0,2], dtype=np.uint32)
olddims = np.array(arr.shape, dtype=np.uint32)
#params = np.array([[8, 2, 2, 3], [7, 3, 1, 4], [5, 2, 0, 5]], dtype=np.uint32)
transposed = arr.transpose(replaces)
clarr = cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=arr.reshape((arr.size,)))
clreplaces = cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=replaces)
clolddims = cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=olddims)
clresult = cl.Buffer(ctx, mf.READ_WRITE, transposed.nbytes)

result = np.empty(transposed.shape).astype(np.uint32)

run.mitransp(queue, (arr.size,), None, clolddims, clreplaces, clarr, clresult)
cl.enqueue_copy(queue, result, clresult)
print(result)









