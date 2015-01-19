import numpy as np

def mislice(gid, N, d, x, y):
    ipg = 1+(min(N, y)-(x%N)-1)//d  #Items per group
    print("Items per group is", ipg)
    s = x//N                        #Group shift
    print("Shift is", s)
    group = s+gid//ipg              #Current group
    print("Current group is", group)
    groupstart = group*N            #Start index of group
    print("Start index of group is", groupstart)
    cmd = gid%ipg                   #Current modulo
    print("Current modulo is", cmd)
    groupid = x%N+cmd*d             #Index of modulo in current grop
    print("Index of modulo in current grop is", groupid)
    return  groupid+groupstart      #Index in all array

"""
x%N+cmd*d + group*N
x%N+(gid%ipg)*d + group*N
x%N+(gid%ipg)*d + (s+gid//ipg)*N
x%N+(gid%ipg)*d + (x//N+gid//ipg)*N
"""

arr = np.random.randint(55, size=(4, 9))
rarr = arr.reshape((4*9,))
