import time

name = "david"
davidLen = len(name)
appendDavid = ""
times = []
x = []
x.append(0)
times.append(0)
for mult in range(1,20000):
    totalLen = davidLen * mult
    appendDavid = appendDavid + name
    list1 = []
    start = time.time()
    for c in appendDavid:
        list1.append(ord(c) % 17)
    end = time.time() - start
    times.append(end)
    x.append(totalLen)

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.gcf()
plt.plot(x,times)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_xlim((min(x),max(x)))
plt.savefig('plot.png')
