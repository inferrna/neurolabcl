import numpy as np

def mislice(gid, N, y):
    #return gid*3-((gid*3)//N)*(N - 3*(min(N, y)//3))
    ipg = min(N, y)//3         #Items per group
    group = gid//ipg           #Current group
    print("Current group is", group)
    groupstart = group*N       #Start index of group
    print("Start index of group is", groupstart)
    cmd = gid%ipg              #Current modulo
    print("Current modulo is", cmd)
    groupid = cmd*3            #Index of modulo in current grop
    print("Index of modulo in current grop is", groupid)
    return  groupid+groupstart #Index in all array


arr = np.random.randint(55, size=(4, 9))
rarr = arr.reshape((4*9,))
