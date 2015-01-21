import numpy as np

def mislice(gid, params):
    N, d, x, y = params[-1]
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

arr = np.random.randint(55, size=(5, 7, 8))
rarr = arr.reshape((arr.size,))


"""
#define dtype uint
#define PC 3 //Dimensions count

uint slice(uint id, uint4 *params){
    uint N = params[id].s0;
    uint d = params[id].s1;
    uint x = params[id].s2;
    uint y = params[id].s3;
    uint ipg = 1+(min(N, y)-(x%N)-1)/d;
    uint s = x//N;
    uint group = s+gid/ipg;
    for(uint i=id; i>=0; i--){

    }


}

__kernel void mislice(__global uint4 *params, __global dtype *data){

}
"""
