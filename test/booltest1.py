import mynp as np
arr = np.array([ -0.5,   0.2], dtype=np.np.float32)
ids = arr > 0
val = np.array([-1.], dtype=np.np.float32)
arr[ids] = val
