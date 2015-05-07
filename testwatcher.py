from watcher import watcher
import numpy as np
import urllib.request, urllib.error, urllib.parse
class X(object):
    def __init__(self, foo):
        self.foo = foo

class Y(object):
    def __init__(self, x):
        self.xoo = x

    def boom(self):
        self.xoo.foo = "xoo foo!"
def main():
    x = X(50)
    watcher(x, 'foo', log_file='log.txt', include =['/home/inferno/.dev/neurolabcl/'])
    x.foo = 500
    x.goo = 300
    y = Y(x)
    y.boom()
    arr = np.arange(0,100,0.1)
    arr = arr**2
    for i in range(3):
        print('a')
        x.foo = i

    for i in range(1):
        i = i+1

main()
