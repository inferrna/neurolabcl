import numpy as np
import neurolab as nl
harr = np.array([[1, 2, 3, 4, 5], [3,4,5,6,7]])
carr = nl.mynp.arr_from_np(harr)
farr = carr.reshape((carr.size,))
print(carr[:,2])
