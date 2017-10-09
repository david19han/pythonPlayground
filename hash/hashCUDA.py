#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multiplies two square matrices together using a *single* block of threads and
global memory only. Each thread computes one element of the resulting matrix.
"""

import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
__global__ void findHash(char *a, int *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = (int) a[i];
    c[i] = j % 17;
}
"""

david = "david"
davidLen = len(david)
appendDavid = ""
times = []
x = []
x.append(0)
times.append(0)

for mult in range(1,10000):
    totalLen = davidLen * mult
    a_cpu = np.chararray(totalLen, )
    appendDavid = appendDavid + david
    for i in range(totalLen):
        a[i] = appendDavid[i]

    c_cpu = np.zeros(totalLen)
    # transfer host (CPU) memory to device (GPU) memory
    a_gpu = gpuarray.to_gpu(a_cpu)

    # create empty gpu array for the result (C = A * B)
    c_gpu = gpuarray.empty(5, np.int32)

    # compile the kernel code
    mod = compiler.SourceModule(kernel_code_template)

    # get the kernel function from the compiled module
    findhash = mod.get_function("findHash")
    start = time.time()
    # call the kernel on the card
    findhash(
        # inputs
        a_gpu,
        c_gpu,
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (5, 1, 1),grid = (1,1)
    )
    end = time.time() - start
    times.append(end)
    x.append(totalLen)

    # print the results
    print "A (GPU):"
    print a_gpu.get()
    print "Matrix C (GPU):"
    print c_gpu.get()

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.gcf()
plt.plot(x,times)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_xlim((min(x),max(x)))
plt.savefig('plot.png')
