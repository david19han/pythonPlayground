import time

#input
name = "david"
davidLen = len(name)
appendDavid = ""
#hold the elapsed times for different lengths
times = []
x = []
x.append(0)
times.append(0)

#run for different lengths
for mult in range(1,20000):
    totalLen = davidLen * mult
    appendDavid = appendDavid + name
    list1 = []
    start = time.time()
    for c in appendDavid:
        #find the hash value
        list1.append(ord(c) % 17)
    #record length
    end = time.time() - start
    times.append(end)
    x.append(totalLen)

#print results
for i in range(0,10000):
    print ("length: ", x[i],"| time: ",times[i])

#plot
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.gcf()
plt.plot(x,times)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_xlim((min(x),max(x)))
plt.savefig('plot.png')
