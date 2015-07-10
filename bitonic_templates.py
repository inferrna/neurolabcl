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

#define inc  ${inc}
#define hinc ${inc>>1} //Half inc
#define qinc ${inc>>2} //Quarter inc
#define einc ${inc>>3} //Eighth of inc
#define dir  ${dir}

% if argsort:
#define ORDER(a,b,ay,by) { bool swap = reverse ^ (getKey(a)<getKey(b));${NS}
                         data_t auxa = a; data_t auxb = b;${NS}
                         idx_t auya = ay; idx_t auyb = by;${NS}
                         a = (swap)?auxb:auxa; b = (swap)?auxa:auxb;${NS}
                         ay = (swap)?auyb:auya; by = (swap)?auya:auyb;}
#define ORDERV(x,y,a,b) { bool swap = reverse ^ (getKey(x[a])<getKey(x[b]));${NS}
                        data_t auxa = x[a]; data_t auxb = x[b];${NS}
                        idx_t auya = y[a]; idx_t auyb = y[b];${NS}
                        x[a] = (swap)?auxb:auxa; x[b] = (swap)?auxa:auxb;${NS}
                        y[a] = (swap)?auyb:auya; y[b] = (swap)?auya:auyb;}
#define B2V(x,y,a)  { ORDERV(x,y,a,a+1) }
#define B4V(x,y,a)  { for (int i4=0;i4<2;i4++) { ORDERV(x,y,a+i4,a+i4+2) } B2V(x,y,a) B2V(x,y,a+2) }
#define B8V(x,y,a)  { for (int i8=0;i8<4;i8++) { ORDERV(x,y,a+i8,a+i8+4) } B4V(x,y,a) B4V(x,y,a+4) }
#define B16V(x,y,a) { for (int i16=0;i16<8;i16++) { ORDERV(x,y,a+i16,a+i16+8) } B8V(x,y,a) B8V(x,y,a+8) }
% else:
#define ORDER(a,b) { bool swap = reverse ^ (getKey(a)<getKey(b)); data_t auxa = a; data_t auxb = b; a = (swap)?auxb:auxa; b = (swap)?auxa:auxb; }
#define ORDERV(x,a,b) { bool swap = reverse ^ (getKey(x[a])<getKey(x[b]));${NS}
      data_t auxa = x[a]; data_t auxb = x[b];${NS}
      x[a] = (swap)?auxb:auxa; x[b] = (swap)?auxa:auxb; }
#define B2V(x,a) { ORDERV(x,a,a+1) }
#define B4V(x,a) { for (int i4=0;i4<2;i4++) { ORDERV(x,a+i4,a+i4+2) } B2V(x,a) B2V(x,a+2) }
#define B8V(x,a) { for (int i8=0;i8<4;i8++) { ORDERV(x,a+i8,a+i8+4) } B4V(x,a) B4V(x,a+4) }
#define B16V(x,a) { for (int i16=0;i16<8;i16++) { ORDERV(x,a+i16,a+i16+8) } B8V(x,a) B8V(x,a+8) }
% endif
#define nsize ${nsize}   //Total next dimensions sizes sum. (Block size)
#define dsize ${dsize}   //Dimension size
"""

ParallelBitonic_B2 = """
// N/2 threads
//ParallelBitonic_B2 
__kernel void run(__global data_t * data\\
% if argsort:
, __global idx_t * index)
% else:
)
% endif
{
  int t  = get_global_id(0) % (dsize>>1); // thread index
  int gt = get_global_id(0) / (dsize>>1);
  int low = t & (inc - 1); // low order bits (below INC)
  int i = (t<<1) - low; // insert 0 at position INC
  int gi = i/dsize; // block index
  bool reverse = ((dir & i) == 0);// ^ (gi%2); // asc/desc order

  int offset = (gt/nsize)*nsize*dsize+(gt%nsize);
  data  += i*nsize + offset; // translate to first value
% if argsort:
  index += i*nsize + offset; // translate to first value
% endif

  // Load data
  data_t x0 = data[  0];
  data_t x1 = data[inc*nsize];
% if argsort:
  // Load index
  idx_t i0 = index[  0];
  idx_t i1 = index[inc*nsize];
% endif

  // Sort
% if argsort:
  ORDER(x0,x1,i0,i1)
% else:
  ORDER(x0,x1)
% endif

  // Store data
  data[0  ] = x0;
  data[inc*nsize] = x1;
% if argsort:
  // Store index
  index[  0] = i0;
  index[inc*nsize] = i1;
% endif
}
"""

ParallelBitonic_B4 = """
// N/4 threads
//ParallelBitonic_B4 
__kernel void run(__global data_t * data\\
% if argsort:
, __global idx_t * index)
% else:
)
% endif
{
  int t  = get_global_id(0) % (dsize>>2); // thread index
  int gt = get_global_id(0) / (dsize>>2);
  int low = t & (hinc - 1); // low order bits (below INC)
  int i = ((t - low) << 2) + low; // insert 00 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  int offset = (gt/nsize)*nsize*dsize+(gt%nsize);
  data  += i*nsize + offset; // translate to first value
% if argsort:
  index += i*nsize + offset; // translate to first value
% endif

  // Load data
  data_t x0 = data[     0];
  data_t x1 = data[  hinc*nsize];
  data_t x2 = data[2*hinc*nsize];
  data_t x3 = data[3*hinc*nsize];
% if argsort:
  // Load index
  idx_t i0 = index[     0];
  idx_t i1 = index[  hinc*nsize];
  idx_t i2 = index[2*hinc*nsize];
  idx_t i3 = index[3*hinc*nsize];
% endif

  // Sort
% if argsort:
  ORDER(x0,x2,i0,i2)
  ORDER(x1,x3,i1,i3)
  ORDER(x0,x1,i0,i1)
  ORDER(x2,x3,i2,i3)
% else:
  ORDER(x0,x2)
  ORDER(x1,x3)
  ORDER(x0,x1)
  ORDER(x2,x3)
% endif

  // Store data
  data[     0] = x0;
  data[  hinc*nsize] = x1;
  data[2*hinc*nsize] = x2;
  data[3*hinc*nsize] = x3;
% if argsort:
  // Store index
  index[     0] = i0;
  index[  hinc*nsize] = i1;
  index[2*hinc*nsize] = i2;
  index[3*hinc*nsize] = i3;
% endif
}
"""

ParallelBitonic_B8 = """
// N/8 threads
//ParallelBitonic_B8 
__kernel void run(__global data_t * data\\
% if argsort:
, __global idx_t * index)
% else:
)
% endif
{
  int t  = get_global_id(0) % (dsize>>3); // thread index
  int gt = get_global_id(0) / (dsize>>3);
  int low = t & (qinc - 1); // low order bits (below INC)
  int i = ((t - low) << 3) + low; // insert 000 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  int offset = (gt/nsize)*nsize*dsize+(gt%nsize);

  data  += i*nsize + offset; // translate to first value
% if argsort:
  index += i*nsize + offset; // translate to first value
% endif

  // Load
  data_t x[8];
% if argsort:
  idx_t y[8];
% endif
  for (int k=0;k<8;k++) x[k] = data[k*qinc*nsize];
% if argsort:
  for (int k=0;k<8;k++) y[k] = index[k*qinc*nsize];
% endif

  // Sort
% if argsort:
  B8V(x,y,0)
% else:
  B8V(x,0)
% endif

  // Store
  for (int k=0;k<8;k++) data[k*qinc*nsize] = x[k];
% if argsort:
  for (int k=0;k<8;k++) index[k*qinc*nsize] = y[k];
% endif
}
"""

ParallelBitonic_B16 = """
// N/16 threads
//ParallelBitonic_B16 
__kernel void run(__global data_t * data\\
% if argsort:
, __global idx_t * index)
% else:
)
% endif
{
  int t  = get_global_id(0) % (dsize>>4); // thread index
  int gt = get_global_id(0) / (dsize>>4);
  int low = t & (einc - 1); // low order bits (below INC)
  int i = ((t - low) << 4) + low; // insert 0000 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  int offset = (gt/nsize)*nsize*dsize+(gt%nsize);

  data  += i*nsize + offset; // translate to first value
% if argsort:
  index += i*nsize + offset; // translate to first value
% endif

  // Load
  data_t x[16];
% if argsort:
  idx_t y[16];
% endif
  for (int k=0;k<16;k++) x[k] = data[k*einc*nsize];
% if argsort:
  for (int k=0;k<16;k++) y[k] = index[k*einc*nsize];
% endif

  // Sort
% if argsort:
  B16V(x,y,0)
% else:
  B16V(x,0)
% endif

  // Store
  for (int k=0;k<16;k++) data[k*einc*nsize] = x[k];
% if argsort:
  for (int k=0;k<16;k++) index[k*einc*nsize] = y[k];
% endif
}
"""

ParallelBitonic_C4 = """
//ParallelBitonic_C4 
__kernel void run\\
% if argsort:
(__global data_t * data, __global idx_t * index, __local data_t * aux, __local idx_t * auy)
% else:
(__global data_t * data, __local data_t * aux)
% endif
{
  int t = get_global_id(0); // thread index
  int wgBits = 4*get_local_size(0) - 1; // bit mask to get index in local memory AUX (size is 4*WG)
  int linc,low,i;
  bool reverse;
  data_t x[4];
% if argsort:
  idx_t y[4];
% endif

  // First iteration, global input, local output
  linc = hinc;
  low = t & (linc - 1); // low order bits (below INC)
  i = ((t - low) << 2) + low; // insert 00 at position INC
  reverse = ((dir & i) == 0); // asc/desc order
  for (int k=0;k<4;k++) x[k] = data[i+k*linc];
% if argsort:
  for (int k=0;k<4;k++) y[k] = index[i+k*linc];
  B4V(x,y,0);
  for (int k=0;k<4;k++) auy[(i+k*linc) & wgBits] = y[k];
% else:
  B4V(x,0);
% endif
  for (int k=0;k<4;k++) aux[(i+k*linc) & wgBits] = x[k];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Internal iterations, local input and output
  for ( ;linc>1;linc>>=2)
  {
    low = t & (linc - 1); // low order bits (below INC)
    i = ((t - low) << 2) + low; // insert 00 at position INC
    reverse = ((dir & i) == 0); // asc/desc order
    for (int k=0;k<4;k++) x[k] = aux[(i+k*linc) & wgBits];
% if argsort:
    for (int k=0;k<4;k++) y[k] = auy[(i+k*linc) & wgBits];
    B4V(x,y,0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k=0;k<4;k++) auy[(i+k*linc) & wgBits] = y[k];
% else:
    B4V(x,0);
    barrier(CLK_LOCAL_MEM_FENCE);
% endif
    for (int k=0;k<4;k++) aux[(i+k*linc) & wgBits] = x[k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Final iteration, local input, global output, INC=1
  i = t << 2;
  reverse = ((dir & i) == 0); // asc/desc order
  for (int k=0;k<4;k++) x[k] = aux[(i+k) & wgBits];
% if argsort:
  for (int k=0;k<4;k++) y[k] = auy[(i+k) & wgBits];
  B4V(x,y,0);
  for (int k=0;k<4;k++) index[i+k] = y[k];
% else:
  B4V(x,0);
% endif
  for (int k=0;k<4;k++) data[i+k] = x[k];
}
"""
