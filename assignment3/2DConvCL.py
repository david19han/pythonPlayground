#!/usr/bin/env python
"""
Vector addition using PyOpenCL.
"""

import time

import pyopencl as cl
import pyopencl.array
import numpy as np
import copy
import scipy
from scipy import signal

# Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

# Set up a command queue; we need to enable profiling to time GPU operations:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Define the OpenCL kernel you wish to run; most of the interesting stuff you
# will be doing involves modifying or writing kernels:
kernel = """
__kernel void func(__global char* a, __global int* c) {
    unsigned int i = get_global_id(0);
    int j = (int) a[i];
    c[i] = j % 17;
}
"""

# Load some random data to process. Note that setting the data type is
# important; if your data is stored using one type and your kernel expects a
# different type, your program might either produce the wrong results or fail to
# run.  Note that Numerical Python uses names for certain types that differ from
# those used in OpenCL. For example, np.float32 corresponds to the float type in
# OpenCL:

#input
david = "david"
davidLen = len(david)
appendDavid = ""
#hold the elapsed times for different lengths
times = []
x = []

x.append(0)
times.append(0)

#run for different input lengths
for mult in range(1,10000):
    totalLen = davidLen * mult
    a = np.chararray(totalLen, )
    appendDavid = appendDavid + david
    for i in range(totalLen):
        a[i] = appendDavid[i]

    a_gpu = cl.array.to_device(queue, a)
    dt = np.dtype(np.int32)
    c_gpu = cl.array.empty(queue, a.shape, dt)

    prg = cl.Program(ctx, kernel).build()

    #record elapsed time
    start = time.time()
    evt = prg.func(queue, a.shape, None, a_gpu.data, c_gpu.data)
    times.append(time.time()-start)
    #evt = prg.func(queue, a.shape, None, a_gpu.data, c_gpu.data)
    #times.append(1e-9 * (evt.profile.end - evt.profile.start))
    x.append(totalLen)

    # Retrieve the results from the GPU:
    c = c_gpu.get()

#print results
for i in range(0,10000):
    print ("length: ", x[i],"| time: ",times[i])

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.gcf()
plt.plot(x,times)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_xlim((min(x),max(x)))
plt.savefig('plotOpenCL.png')
