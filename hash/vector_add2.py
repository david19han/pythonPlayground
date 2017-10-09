#!/usr/bin/env python
"""
Vector addition using PyOpenCL.
"""

import time

import pyopencl as cl
import pyopencl.array
import numpy as np

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
    printf("%d\n",j);
    c[i] = j % 17;
}
"""

# Load some random data to process. Note that setting the data type is
# important; if your data is stored using one type and your kernel expects a
# different type, your program might either produce the wrong results or fail to
# run.  Note that Numerical Python uses names for certain types that differ from
# those used in OpenCL. For example, np.float32 corresponds to the float type in
# OpenCL:
a = np.chararray(5, )
a[0] = 'd'
a[1] = 'a'
a[2] = 'v'
a[3] = 'i'
a[4] = 'd'
print a
# We can use PyOpenCL's Array type to easily transfer data from numpy arrays to
# GPU memory (and vice versa):
a_gpu = cl.array.to_device(queue, a)
c_gpu = cl.array.empty(queue, a.shape, a.dtype)

# Launch the kernel; notice that you must specify the global and locals to
# determine how many threads of execution are run. We can take advantage of Numpy to
# use the shape of one of the input arrays as the global size. Since our kernel
# only accesses the global work item ID, we simply set the local size to None:
prg = cl.Program(ctx, kernel).build()
prg.func(queue, a.shape, None, a_gpu.data, c_gpu.data)

# Retrieve the results from the GPU:
c = c_gpu.get()

print 'input (a):    ', a
print 'opencl (c): ', c

# Compare the results from the GPU with those obtained using Numerical Python;
# this should print True:
