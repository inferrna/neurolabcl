import numpy as np
import mynp
import time

harr = np.random.randint(99, size=(30, 55, 4,))
carr = mynp.arr_from_np(harr)
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
