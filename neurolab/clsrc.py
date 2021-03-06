slicedefs = """
#define dtype {0} //Main datatype
#define idtype {1} //Type for indexing
#define PC {2} //Dimensions count
#define ucond uint
"""
norecslicesrc = """
uint slice{0}(uint id, __global int4 *params, const uint c)<%
    int N = params[{3}].s0;      //Size
    int x = params[{3}].s1;      //Start
    int y = params[{3}].s2;      //End
    int d = abs(params[{3}].s3); //Step
    int minny = min(N, y);       //Real end
    int sd = params[{3}].s3/d;   //Sign
    int ipg = 1+(minny-x-1)/d;   //Items per group
    int group = id/ipg;  //Current group as total
    {1}group = slice{2}(group, params, c-1); //Current group as subgroup
    int groupstart = group*N;
    int cgi = id%ipg;    //Index in current group
    int groupid;
    if(sd>0)
        groupid = x+cgi*d;
    else
        groupid = (minny-1-x)-cgi*d;
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
% if noidx:
__kernel void mislice(int dummy, __global dtype *data, __global dtype *result){
% else:
__kernel void mislice(__global int4 *params, __global dtype *data, __global dtype *result){
% endif
    uint gid = get_global_id(0);
% if noidx:
    result[gid] = data[gid];
% else:
    result[gid] = data[slice(gid, params, PC-1)];
% endif
}
"""
slicesetsrc = """
% if noidx:
__kernel void mislice(int dummy, __global dtype *data, __global dtype *source, uint data_offset, uint srcoffset){
% else:
__kernel void mislice(__global int4 *params, __global dtype *data, __global dtype *source, uint data_offset, uint srcoffset){
% endif
    uint gid = get_global_id(0);
% if noidx:
    data[gid+data_offset] = source[gid + srcoffset];
% else:
    data[slice(gid, params, PC-1)+data_offset] = source[gid + srcoffset];
% endif
}
% if noidx:
__kernel void mislicesingle(int dummy, __global dtype *data, __global dtype *_source, uint srcoffset){
% else:
__kernel void mislicesingle(__global int4 *params, __global dtype *data, __global dtype *_source, uint srcoffset){
% endif
    uint gid = get_global_id(0);
    uint gid1 = get_global_id(1);
    __global dtype *source = _source + srcoffset;
    __local dtype value[${cs}];
    if(get_local_id(0) == 0){
        for(uint i=0; i<${cs}; i++){
            value[i] = source[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
% if noidx:
    data[gid*${cs}+gid1] = value[gid1];
% else:
    data[slice(gid, params, PC-1)*${cs}+gid1] = value[gid1];
% endif
}
"""

deletesrc = """
__kernel void delete(
% if rc > 1:
                     __global uint *removed,  //Removed positions fom spec dimension
% else:
                     uint removed,            //Removed position fom spec dimension
% endif
                     __global dtype *src,
                     __global dtype *dst
){
    uint newdims[${ndims}] = {${newdims}}; //New shape
    uint olddims[${ndims}] = {${olddims}}; //Old shape
    //find positions in new dims
    uint oldpos[${ndims}];
    uint gid = get_global_id(0);
    uint newgid = gid;
    int mult = 1;
    uint idx = 0;
    int i;
    for(i = ${ndims-1}; i>${dimr}; i--){
        oldpos[i] = newgid % newdims[i]; //same as currpos
        newgid = newgid/newdims[i];
    }
    i = ${dimr};
    oldpos[i] = newgid % newdims[i];
    newgid = newgid/newdims[i];
% if rc > 1:
    uint op = oldpos[i];
    for(int j=0; j<${rc}; j++){
        oldpos[i] += (uint) ((removed[j]) - j <= op); //New position is shifted, substract j to compensate
    }
% else:
    oldpos[i] += (uint) (removed <= oldpos[i]); //New position is shifted, substract j to compensate
% endif
    for(i=${dimr-1}; i>=0; i--){
        oldpos[i] = newgid % newdims[i];
        newgid = newgid/newdims[i];
    }
    for(i=${ndims - 1}; i>=0; i--){
        idx += oldpos[i]*mult;
        mult *=  olddims[i];
    }
    dst[gid] = src[idx]; 
    //map it to position in old dims
    //and get it
}
"""

getbyidssrc = """
__kernel void getbyids(__global idtype *ids, __global dtype *data, __global dtype *result){
    size_t gid0 = get_global_id(0);  //Addr inside block
    size_t gs0 = get_global_size(0); //Block size  
    size_t gid1 = get_global_id(1);  //Block addr in result array
    size_t idx = ids[gid1];          //Block addr in source data
    size_t didx = idx*gs0+gid0;      //Addr in source data
    size_t ridx = gid1*gs0+gid0;     //Addr in result array
    dtype res = data[didx];
    result[ridx] = res; 
}
"""
setbyidssrc = """
__kernel void setbyids(__global idtype *ids, __global dtype *data, __global dtype *_value){
    size_t gid0 = get_global_id(0);  //Addr inside block
    size_t gs0 = get_global_size(0); //Block size  
    size_t gid1 = get_global_id(1);  //Block index
    size_t idx = ids[gid1];          //Block addr in source data
    size_t didx = idx*gs0+gid0;      //Addr in source data
    __local dtype value[${cs}];
    % if cs>1:
    if(get_local_id(0) == 0){
        for(uint i=0; i<${cs}; i++){
            value[i] = _value[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    data[didx] = value[gid0];
    % else:
    if(get_local_id(0) == 0){
        value[0] = _value[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    data[didx] = value[0];
    % endif
}
"""
setifsrc = """
__kernel void setif(__global idtype *ids, __global dtype *data, __global dtype *_value){
    size_t gid0 = get_global_id(0);  //Addr inside block
    dtype ires = data[gid0];
    ucond id = (ucond) ids[gid0];
    % if cs>1:
    data[gid0] = select(ires, _value[gid0 % cs], id);
    % elif cs==1:
    __local dtype value;
    if(get_local_id(0) == 0){
        value = _value[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    data[gid0] = select(ires, value, id);
    % elif cs==0:
    data[gid0] = select(ires, _value[gid0], id);;
    % endif
}
"""

signsrc = """
__kernel void asign(__global dtype *inpt, __global dtype *outpt){
    uint gid = get_global_id(0);
    dtype linpt = inpt[gid];
    dtype res = copysign(1, linpt);
    res = select(res, (dtype) 0, (ucond)(linpt == (dtype)0));
    outpt[gid] = res; 
}\n
"""
isinfsrc = """
__kernel void isposinf(__global dtype *inpt, __global char *outpt){
    uint gid = get_global_id(0);
    dtype val = inpt[gid];
    int res = isinf(val);
    outpt[gid] = (char) res;
}\n
__kernel void isneginf(__global float *inpt, __global char *outpt){
    uint gid = get_global_id(0);
    dtype val = inpt[gid];
    int res =  signbit(val) * isinf(val);
    outpt[gid] = (char) res;
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

__kernel void mitransp(__global dtype *data, __global dtype *result){
    uint gid = get_global_id(0);
    uint newgid = gid;
    uint lid = get_local_id(0);
    uint wid = get_group_id(0);
    int i,j;
    uint currposs[${ndim}];
    uint newposs[${ndim}];
    uint newid = 0;

% for i in range(ndim-1, -1, -1):
    currposs[${i}] = newgid % ${olddims[i]}; //same as currpos
    newgid = newgid/${olddims[i]};
% endfor
% for i in range(0, ndim):
    <% j = replaces[i] %>
    newposs[${i}] = currposs[${j}];
% endfor
% for i in range(0, ndim):
    newid += ${scales[i]}*newposs[${i}];
% endfor
    result[newid] = data[gid];
}
"""
minsrc = """
__kernel void misum(__global dtype *data, __global dtype *result){
    uint gid = get_global_id(0);
    dtype res = data[gid*PC];
    for(uint i = gid*PC+1; i<gid*PC+PC; i++){
        res = min(data[i], res);
    }
    result[gid] = res;
}
"""
maxsrc = """
__kernel void misum(__global dtype *data, __global dtype *result){
    uint gid = get_global_id(0);
    dtype res = data[gid*PC];
    for(uint i = gid*PC+1; i<gid*PC+PC; i++){
        res = max(data[i], res);
    }
    result[gid] = res;
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
singlesetsrc = """
__kernel void prg(__global dtype *data, __global dtype *_source, uint srcoffset, uint goff){
    uint gid = get_global_id(0);
    __global dtype *source = _source+srcoffset;
    data[gid+goff] = source[gid];
}
"""
singlesrc = """

__kernel void prg(__global dtype *data, __global idtype *result, __global dtype *gparam){
    uint gid = get_global_id(0);
    __local dtype param;
    if(get_local_id(0)==0) param = gparam[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    dtype res = (${idtype}) ((${neg} data[gid]) ${operator} param);
    result[gid] = res;
}
"""
ndsrc = """
__kernel void prg(__global dtype *data, __global idtype *result, __global dtype *gparam){
    uint gid0 = get_global_id(0);
    uint gid1 = get_global_id(1);
    uint gs1 = get_global_size(1);
    uint did = gid0*gs1 + gid1;
    result[did] = (${idtype}) (data[did] ${operator} gparam[gid1]);
}
"""
ndrsrc = """
__kernel void prg(__global dtype *data, __global idtype *result, __global dtype *gparam){
    uint gid = get_global_id(0);
    uint did = gid/PC;
    result[gid] = (${idtype}) (data[gid] ${operator} gparam[did]);
}
"""

doubledotsrc = """
__kernel void midot(__global dtype *data, __global dtype *gparam, __global dtype *result){
    uint gid0 = get_global_id(0);  //Addressing data
    uint gid1 = get_global_id(1);  //Adressing multiplier
    uint gs0 = get_global_size(0); //Count of data banks
    uint gs1 = get_global_size(1); //Count of multiplier banks
    uint sgd0 = gid0*PC;           //Data shift
    uint sgd1 = gid1*PC;           //Multiplier shift
    uint did = gid0*gs1 + gid1;    //Result address
    dtype res = 0;
    for(uint i=0; i<PC; i++){
        res += data[sgd0+i] * gparam[gid1+i*gs1];
    }
    result[did] = res;
}
"""

smalldotsrc = """
__kernel void midot(__global dtype *data, __global dtype *gparam, __global dtype *result){
    uint gid = get_global_id(0);
    uint sgd = gid*PC;
    dtype res = 0;
    __local dtype lparam[PC];
    event_t a[2];
    a[0] = async_work_group_copy(lparam, gparam, PC, 0);
    wait_group_events(1, a);
    for(uint i=0; i<PC; i++){
        res += data[sgd+i] * lparam[i];
    }
    result[gid] = res;
}
"""

