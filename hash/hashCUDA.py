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

# define the (square) matrix size
#  note that we'll only use *one* block of threads here
#  as a consequence this number (squared) can't exceed max_threads,
#  see http://documen.tician.de/pycuda/util.html#pycuda.tools.DeviceData
#  for more information on how to get this number for your device

# create two random square matrices
a_cpu = np.chararray(5, )
a_cpu[0] = 'd'
a_cpu[1] = 'a'
a_cpu[2] = 'v'
a_cpu[3] = 'i'
a_cpu[4] = 'd'
c_cpu = np.zeros(5)
print a_cpu
print a_cpu.shape
print c_cpu
print c_cpu.shape
# transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)

# create empty gpu array for the result (C = A * B)
c_gpu = gpuarray.to_gpu(c_cpu)

# compile the kernel code
mod = compiler.SourceModule(kernel_code_template)

# get the kernel function from the compiled module
findhash = mod.get_function("findHash")

# call the kernel on the card
findhash(
    # inputs
    a_gpu,
    c_gpu,
    # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
    block = (5, 1, 1),grid = (1,1)
    )

# print the results
print "Matrix A (GPU):"
print a_gpu.get()


print "Matrix C (GPU):"
print c_gpu.get()
