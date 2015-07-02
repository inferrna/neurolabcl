import numpy as np
import neurolab as nl
import time

harr = np.random.randint(99, size=(3, 5, 4,))
carr = nl.mynp.arr_from_np(harr)
decr = np.random.randint(99, size=(5, 4,))
cld = nl.mynp.arr_from_np(decr)
print(carr - cld)

