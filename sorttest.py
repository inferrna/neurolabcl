import numpy as np
import neurolab as nl
import time
from pyopencl import array

harr = np.random.randint(599, size=(30, 55, 4,))
carr = nl.mynp.arr_from_np(harr)
farr = carr.reshape((carr.size,))
hsa = harr[5::9,::7,::2]
csa = carr[5::9,::7,::2]
t1 = time.time()
hs = harr[5::7,::7,::2]
t2 = time.time()
cs = carr[5::7,::7,::2]
t3 = time.time()
#harr = np.random.randint(99, size=(3, 4, 5, 6,))
#carr = nl.mynp.arr_from_np(harr)
#carr.transpose(1,0,2,3)
arng = nl.mynp.clarray.arange(nl.mynp.queue, 0, cs.size, 1, dtype=np.int32)
csat = cs.reshape(cs.size).astype(np.float32)*1.3
srt = nl.mynp.algorithm.RadixSort(nl.mynp.ctx, "float *mkey, int *tosort", "mkey[i]", ["mkey", "tosort"])#,\
#                                  index_dtype=np.int32, key_dtype=np.float32)
csat, arng = srt(csat, arng, key_bits=32)[0]
#print(k)
#csa.__class__ = array.Array
#print(csa[k])
