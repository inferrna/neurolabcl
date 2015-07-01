"""
// Sort kernels
// EB Jun 2011
"""

defines = """
typedef ${dtype} data_t;
typedef ${idxtype} idx_t;
typedef ${idxtype}2 idx_t2;
#if CONFIG_USE_VALUE
#define getKey(a) ((a).x)
#define getValue(a) ((a).y)
#define makeData(k,v) ((${dtype}2)((k),(v)))
#else
#define getKey(a) (a)
#define getValue(a) (0)
#define makeData(k,v) (k)
#endif

#ifndef BLOCK_FACTOR
#define BLOCK_FACTOR 1
#endif
#define ORDER(a,b,ai,bj) { bool swap = reverse ^ (getKey(a)<getKey(b));${NS}
                         data_t auxa = a; data_t auxb = b;${NS}
                         idx_t auxi = ai; idx_t auxj = bj;${NS}
                         a = (swap)?auxb:auxa; b = (swap)?auxa:auxb;${NS}
                         ai = (swap)?auxj:auxi; bj = (swap)?auxi:auxj;}

#define inc  ${inc}
#define hinc ${inc>>1} //Half inc
#define qinc ${inc>>2} //Quarter inc
#define einc ${inc>>3} //Eighth of inc
#define dir  ${dir}


#define ORDERV(x,y,a,b) { bool swap = reverse ^ (getKey(x[a])<getKey(x[b]));${NS}
                        data_t auxa = x[a]; data_t auxb = x[b];${NS}
                        idx_t auxi = y[a]; idx_t auxj = y[b];${NS}
                        x[a] = (swap)?auxb:auxa; x[b] = (swap)?auxa:auxb;${NS}
                        y[a] = (swap)?auxj:auxi; y[b] = (swap)?auxi:auxj;}
#define B2V(x,y,a)  { ORDERV(x,y,a,a+1) }
#define B4V(x,y,a)  { for (int i4=0;i4<2;i4++) { ORDERV(x,y,a+i4,a+i4+2) } B2V(x,y,a) B2V(x,y,a+2) }
#define B8V(x,y,a)  { for (int i8=0;i8<4;i8++) { ORDERV(x,y,a+i8,a+i8+4) } B4V(x,y,a) B4V(x,y,a+4) }
#define B16V(x,y,a) { for (int i16=0;i16<8;i16++) { ORDERV(x,y,a+i16,a+i16+8) } B8V(x,y,a) B8V(x,y,a+8) }
"""

ParallelBitonic_B2 = """
// N/2 threads
//ParallelBitonic_B2 
__kernel void run(__global data_t * data, __global idx_t * index)
{
  int t = get_global_id(0); // thread index
  int low = t & (inc - 1); // low order bits (below INC)
  int i = (t<<1) - low; // insert 0 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  data  += i; // translate to first value
  index += i; // translate to first value

  // Load data
  data_t x0 = data[  0];
  data_t x1 = data[inc];
  // Load index
  idx_t i0 = index[  0];
  idx_t i1 = index[inc];

  // Sort
  ORDER(x0,x1,i0,i1)

  // Store data
  data[0  ] = x0;
  data[inc] = x1;
  // Store index
  index[  0] = i0;
  index[inc] = i1;
}
"""

ParallelBitonic_B4 = """
// N/4 threads
//ParallelBitonic_B4 
__kernel void run(__global data_t * data, __global idx_t * index)
{
  int t = get_global_id(0); // thread index
  int low = t & (hinc - 1); // low order bits (below INC)
  int i = ((t - low) << 2) + low; // insert 00 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  data  += i; // translate to first value
  index += i; // translate to first value

  // Load data
  data_t x0 = data[     0];
  data_t x1 = data[  hinc];
  data_t x2 = data[2*hinc];
  data_t x3 = data[3*hinc];
  // Load index
  idx_t i0 = index[     0];
  idx_t i1 = index[  hinc];
  idx_t i2 = index[2*hinc];
  idx_t i3 = index[3*hinc];

  // Sort
  ORDER(x0,x2,i0,i2)
  ORDER(x1,x3,i1,i3)
  ORDER(x0,x1,i0,i1)
  ORDER(x2,x3,i2,i3)

  // Store data
  data[     0] = x0;
  data[  hinc] = x1;
  data[2*hinc] = x2;
  data[3*hinc] = x3;
  // Store index
  index[     0] = i0;
  index[  hinc] = i1;
  index[2*hinc] = i2;
  index[3*hinc] = i3;
}
"""

ParallelBitonic_B8 = """
// N/8 threads
//ParallelBitonic_B8 
__kernel void run(__global data_t * data, __global idx_t * index)
{
  int t = get_global_id(0); // thread index
  int low = t & (qinc - 1); // low order bits (below INC)
  int i = ((t - low) << 3) + low; // insert 000 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  data  += i; // translate to first value
  index += i; // translate to first value

  // Load
  data_t x[8];
  idx_t y[8];
  for (int k=0;k<8;k++) x[k] = data[k*qinc];
  for (int k=0;k<8;k++) y[k] = index[k*qinc];

  // Sort
  B8V(x,y,0)

  // Store
  for (int k=0;k<8;k++) data[k*qinc] = x[k];
  for (int k=0;k<8;k++) index[k*qinc] = y[k];
}
"""

ParallelBitonic_B16 = """
// N/16 threads
//ParallelBitonic_B16 
__kernel void run(__global data_t * data, __global idx_t * index)
{
  int t = get_global_id(0); // thread index
  int low = t & (einc - 1); // low order bits (below INC)
  int i = ((t - low) << 4) + low; // insert 0000 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  data  += i; // translate to first value
  index += i; // translate to first value

  // Load
  data_t x[16];
  idx_t y[16];
  for (int k=0;k<16;k++) x[k] = data[k*einc];
  for (int k=0;k<16;k++) y[k] = index[k*einc];

  // Sort
  B16V(x,y,0)

  // Store
  for (int k=0;k<16;k++) data[k*einc] = x[k];
  for (int k=0;k<16;k++) index[k*einc] = y[k];
}
"""

ParallelBitonic_C4 = """
//ParallelBitonic_C4 
__kernel void run(__global data_t * data, __global idx_t * index, __local data_t * aux, __local idx_t * auy)
{
  int t = get_global_id(0); // thread index
  int wgBits = 4*get_local_size(0) - 1; // bit mask to get index in local memory AUX (size is 4*WG)
  int linc,low,i;
  bool reverse;
  data_t x[4];
  idx_t y[4];

  // First iteration, global input, local output
  linc = hinc;
  low = t & (linc - 1); // low order bits (below INC)
  i = ((t - low) << 2) + low; // insert 00 at position INC
  reverse = ((dir & i) == 0); // asc/desc order
  for (int k=0;k<4;k++) x[k] = data[i+k*linc];
  for (int k=0;k<4;k++) y[k] = index[i+k*linc];
  B4V(x,y,0);
  for (int k=0;k<4;k++) aux[(i+k*linc) & wgBits] = x[k];
  for (int k=0;k<4;k++) auy[(i+k*linc) & wgBits] = y[k];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Internal iterations, local input and output
  for ( ;linc>1;linc>>=2)
  {
    low = t & (linc - 1); // low order bits (below INC)
    i = ((t - low) << 2) + low; // insert 00 at position INC
    reverse = ((dir & i) == 0); // asc/desc order
    for (int k=0;k<4;k++) x[k] = aux[(i+k*linc) & wgBits];
    for (int k=0;k<4;k++) y[k] = auy[(i+k*linc) & wgBits];
    B4V(x,y,0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k=0;k<4;k++) aux[(i+k*linc) & wgBits] = x[k];
    for (int k=0;k<4;k++) auy[(i+k*linc) & wgBits] = y[k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Final iteration, local input, global output, INC=1
  i = t << 2;
  reverse = ((dir & i) == 0); // asc/desc order
  for (int k=0;k<4;k++) x[k] = aux[(i+k) & wgBits];
  for (int k=0;k<4;k++) y[k] = auy[(i+k) & wgBits];
  B4V(x,y,0);
  for (int k=0;k<4;k++) data[i+k] = x[k];
  for (int k=0;k<4;k++) index[i+k] = y[k];
}
"""
