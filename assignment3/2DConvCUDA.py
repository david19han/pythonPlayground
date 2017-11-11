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
#include <stdio.h>
__global__ void dilateConv(float *input,float *output,float *mask, int maskWidth, int width, int height)
{
    int tx = threadIdx.x
    int ty = threadIdy.y
    printf("tx is %d | ty is %d",tx,ty)
}
"""

maskWidth = 3
numRows = 7
numCols = 7
stride = 1

input_cpu = np.random.rand(numRows,numCols)
output_cpu = np.ones_like(input_cpu)
mask_cpu = [[-1,-2,-1],[0,0,0],[1,2,1]]

input_gpu = gpuarray.to_gpu(input_cpu)
output_gpu = gpuarray.to_gpu(output_cpu)
mask_gpu = gpuarray.to_gpu(mask_cpu)

mod = compiler.SourceModule(kernel_code_template)

dilateConv = mod.get_function("dilateConv")

block = (numRows,numCols,1)
grid = (1,1)
findhash(
    # inputs
    input_gpu,
    output_gpu,
    mask_gpu,
    maskWidth,
    numCols,
    numRows,
    block=block,grid=grid
)
