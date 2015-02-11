slicedefs = """
#define dtype {0}
#define PC {1} //Dimensions count
"""
norecslicesrc = """
uint slice{0}(uint id, __global int4 *params, const uint c)<%
    int N = params[{3}].s0;
    int x = params[{3}].s1;
    int y = params[{3}].s2;
    int d = abs(params[{3}].s3);
    int minny = min(N, y);
    int sd = params[{3}].s3/d; //Sign
    int ipg = 1+(minny-(x%N)-1)/d;
    int group = id/ipg;
    {1}group = slice{2}(group, params, c-1);
    int groupstart = group*N;
    int cmd = id%ipg;
    int groupid = x+(minny-1-x)*((1-sd)/2)+sd*cmd*d;
    return  groupid+groupstart;
%>
"""
slicesrc = """
uint slice(uint id, __global int4 *params, const uint c){
    int N = params[c].s0;
    int x = params[c].s1;
    int y = params[c].s2;
    int d = abs(params[c].s3);
    int minny = min(N, y);
    int sd = params[c].s3/d; //Sign
    int ipg = 1+(minny-(x%N)-1)/d;
    int group = id/ipg;
    if(c>0) group = slice(group, params, c-1);
    int groupstart = group*N;
    int cmd = id%ipg;
    int groupid = x+(minny-1-x)*((1-sd)/2)+sd*cmd*d;
    return  groupid+groupstart;
}
"""
slicegetsrc = """
__kernel void mislice(__global int4 *params, __global dtype *data, __global dtype *result){
    uint gid = get_global_id(0);
    result[gid] = data[slice(gid, params, PC-1)];
}
"""
slicesetsrc = """
__kernel void mislice(__global int4 *params, __global dtype *data, __global dtype *source){
    uint gid = get_global_id(0);
    data[slice(gid, params, PC-1)] = source[gid];
}
__kernel void mislicesingle(__global int4 *params, __global dtype *data, __global dtype *source){
    uint gid = get_global_id(0);
    __local dtype value;
    if(get_local_id(0) == 0){
        value = source[0];
    }
    data[slice(gid, params, PC-1)] = value;
}
"""


signsrc = """
__kernel void asign(__global float *inpt, __global float *outpt){
    uint gid = get_global_id(0);
    float res = copysign(1, inpt[gid]);
    outpt[gid] = res; 
}\n
"""
isinfsrc = """
__kernel void isposinf(__global float *inpt, __global uint *outpt){
    uint gid = get_global_id(0);
    float val = inpt[gid];
    float res = isinf(val);
    outpt[gid] = res;
}\n
__kernel void isneginf(__global float *inpt, __global uint *outpt){
    uint gid = get_global_id(0);
    float val = inpt[gid];
    float res =  signbit(val) * isinf(val);
    outpt[gid] = res;
}\n
"""

findposnorecsrc = """
void findpos{0}(uint id, __global uint *olddims, uint *currposs, uint c)<%
    uint ipg = olddims[{3}];
    uint group = id/ipg;
    uint gid = id%ipg;
    currposs[{3}] = gid;
    {1}findpos{2}(group, olddims, currposs, c-1);
%>
"""
findpossrc = """
void findpos(uint id, __global uint *olddims, uint *currposs, uint c){
    uint ipg = olddims[c];
    uint group = id/ipg;
    uint gid = id%ipg;
    currposs[c] = gid;
    if(c>0) findpos(group, olddims, currposs, c-1);
}
"""

transpsrc = """

__kernel void mitransp(__global uint *olddims, __global uint *replaces,
                       __global dtype *data, __global dtype *result){
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint wid = get_group_id(0);
    uint i,j;
    uint currposs[PC];
    uint newposs[PC];
    uint newdims[PC];
    uint newid = 0;
    uint scales[PC];

    findpos(gid, olddims, currposs, PC-1);
    for(i=0; i<PC; i++){ //i as current dim
        j = replaces[i];
        newposs[i] = currposs[j];
        newdims[i] = olddims[j];
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
sumsrc = """
__kernel void misum(__global dtype *data, __global dtype *result){
    uint gid = get_global_id(0);
    dtype res = 0;
    for(uint i = gid*PC; i<gid*PC+PC; i++){
        res += data[i];
    }
    result[gid] = res;
}
"""
singlesumsrc = """
__kernel void misinglesum(__global dtype *data, __global dtype *result, __global dtype *gparam){
    uint gid = get_global_id(0);
    __local dtype param;
    if(get_local_id(0)==0) param = gparam[0];
    dtype res = data[gid] + param;
    result[gid] = res;
}
"""
singlemulsrc = """
__kernel void misinglemul(__global dtype *data, __global dtype *result, __global dtype *gparam){
    uint gid = get_global_id(0);
    __local dtype param;
    if(get_local_id(0)==0) param = gparam[0];
    result[gid] = data[gid]*param;
}
"""
singlenegsubsrc = """
__kernel void misinglenegsub(__global dtype *data, __global dtype *result, __global dtype *gparam){
    uint gid = get_global_id(0);
    __local dtype param;
    if(get_local_id(0)==0) param = gparam[0];
    result[gid] = param - data[gid];
}
"""
singlesubsrc = """
__kernel void misinglesub(__global dtype *data, __global dtype *result, __global dtype *gparam){
    uint gid = get_global_id(0);
    __local dtype param;
    if(get_local_id(0)==0) param = gparam[0];
    result[gid] = data[gid] - param;
}
"""

