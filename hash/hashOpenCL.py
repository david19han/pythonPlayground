#!/usr/bin/env python

"""
Vector addition using PyOpenCL.
"""

import time

import pyopencl as cl
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
queue = cl.CommandQueue(ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

# Define the OpenCL kernel you wish to run; most of the interesting stuff you
# will be doing involves modifying or writing kernels:
kernel = """
__kernel void func(__global char* a, __global int* c) {
    unsigned int i = get_global_id(0);
        int i;
        for(i=0;i<strlen(a);i++){
            char c = a[i];
            int ascii = (int) c;
            c[i] = ascii % 17;
        }
        int j;
        for(j=0;j<strlen(a);j++){
            printf("key %c | index %d\n",a[j],n[j]);
        }
}
"""

# Load some random data to process. Note that setting the data type is
# important; if your data is stored using one type and your kernel expects a
# different type, your program might either produce the wrong results or fail to
# run.  Note that Numerical Python uses names for certain types that differ from
# those used in OpenCL. For example, np.float32 corresponds to the float type in
# OpenCL:
N = 16
a = np.chararray((1,5))
a = "david"

# You need to set the flags of the buffers you create properly; otherwise,
# you might not be able to read or write them as needed:
mf = cl.mem_flags
c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

# Launch the kernel; notice that you must specify the global and locals to
# determine how many threads of execution are run. We can take advantage of Numpy to
# use the shape of one of the input arrays as the global size. Since our kernel
# only accesses the global work item ID, we simply set the local size to None:
prg = cl.Program(ctx, kernel).build()
prg.func(queue, a.shape, None, a_buf, c_buf)

# Retrieve the results from the GPU:
c = np.empty_like(a)
cl.enqueue_copy(queue, c, c_buf)

print 'input (a):    ', a
print 'opencl (a+b): ', c

# Compare the results from the GPU with those obtained using Numerical Python;
# this should print True:
print 'equal:        ', np.allclose(a, c)
