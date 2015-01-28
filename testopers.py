import numpy as np
import neurolab as nl
import time

harr = np.random.randint(99, size=(330, 500, 9,))
carr = nl.mynp.arr_from_np(harr)
farr = carr.reshape((carr.size,))
hsa = harr[5::9,::21,::2]
csa = carr[5::9,::21,::2]
t1 = time.time()
hs = harr[5::7,::21,::2]
t2 = time.time()
cs = carr[5::7,::21,::2]
t3 = time.time()
print("{0} for cpu and {1} for gpu".format(t2-t1, t3-t2))
harr = np.random.randint(99, size=(3, 4, 5, 6,))
carr = nl.mynp.arr_from_np(harr)
carr.transpose(1,0,2,3)
