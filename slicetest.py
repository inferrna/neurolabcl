import numpy as np
import pyopencl as cl
from pyopencl import array

def mislice(gid, params):
    
        N, x, y, d = params[-1]
        ipg = 1+(min(N, y)-(x%N)-1)//d  #Items per group
        print("Items per group is", ipg)
        s = x//N                        #Group shift
        print("Shift is", s)
        group = s+gid//ipg              #Current group
        if len(params)>1: group = mislice(group, params[:-1])
        print("Current group is", group)
        groupstart = group*N            #Start index of group
        print("Start index of group is", groupstart)
        cmd = gid%ipg                   #Current modulo
        print("Current modulo is", cmd)
        groupid = x%N+cmd*d             #Index of modulo in current group
        print("Index of modulo in current grop is", groupid)
        return  groupid+groupstart      #Index in all array

"""
x%N+cmd*d + group*N
x%N+(gid%ipg)*d + group*N
x%N+(gid%ipg)*d + (s+gid//ipg)*N
x%N+(gid%ipg)*d + (x//N+gid//ipg)*N
"""

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

__kernel void mislice(__global uint4 *params, __global dtype *data, __global dtype *result){
    uint gid = get_global_id(0);
    result[gid] = data[slice(gid, params, PC-1)];
}
"""

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags


run = cl.Program(ctx, gpusrc).build()
params = np.array([[55, 0, 55, 2], [77, 1, 4, 3], [88, 2, 3, 1]], dtype=np.uint32)
#params = np.array([[8, 2, 2, 3], [7, 3, 1, 4], [5, 2, 0, 5]], dtype=np.uint32)
sliced = arr[::2, 1:4:3, 2]
print( arr.reshape((arr.size,))[mislice(0, params )])
clarr = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=arr.reshape((arr.size,)))
clparams = cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=params)
clresult = cl.Buffer(ctx, mf.READ_WRITE, sliced.nbytes)

result = np.empty(sliced.size).astype(np.uint32)

run.mislice(queue, (sliced.size,), None, clparams, clarr, clresult)
cl.enqueue_copy(queue, result, clresult)
print(result)









