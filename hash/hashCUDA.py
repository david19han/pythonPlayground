#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit
import pycuda.driver as drv

import time

#kernel code to find hash
kernel_code_template = """
__global__ void findHash(char *a, int *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = (int) a[i];
    c[i] = j % 17;
}
"""

#input name
david = "david"
davidLen = len(david)
appendDavid = ""
#hold the elapsed times for different lengths
times = []
x = []
x.append(0)
times.append(0)
totalLen = 0
for mult in range(1,10000):
    totalLen = davidLen * mult
    #initialize the array holding array of chars with my name
    a_cpu = np.chararray(totalLen, )
    appendDavid = appendDavid + david
    for i in range(totalLen):
        a_cpu[i] = appendDavid[i]

    c_cpu = np.zeros(totalLen)
    # transfer host (CPU) memory to device (GPU) memory
    a_gpu = gpuarray.to_gpu(a_cpu)

    # create empty gpu array for the result
    c_gpu = gpuarray.empty(totalLen, np.int32)

#    c_gpu = gpuarray.empty_like(david, np.int32)
#np.array(list(namestr))
#ascii_name = ord(npname)


#print results
for i in range(1,len(x)):
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
plt.savefig('plotCUDA.png')
