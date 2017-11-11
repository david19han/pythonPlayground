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
__global__ void DilateConKernel(float *input,float *output,float *mask, int maskWidth)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    printf("tx is %d | ty is %d",tx,ty);
    printf("%d %d %d",MASK_WIDTH,ROWS,COLS);
}
"""

maskWidth = 3
numRows = 7
numCols = 7
stride = 1


input_cpu = np.random.randn(numRows,numCols).astype(np.float32)
output_cpu = np.random.randn(numRows,numCols).astype(np.float32)
mask_cpu = np.random.randn(maskWidth,maskWidth).astype(np.float32)

input_gpu = gpuarray.to_gpu(input_cpu)
output_gpu = gpuarray.to_gpu(output_cpu)
mask_gpu = gpuarray.to_gpu(mask_cpu)


kernel_code = kernel_code_template % {
         'MASK_WIDTH': maskWidth,
         'ROWS' : numRows,
         'COLS' : numCols
         }

mod = compiler.SourceModule(kernel_code)

dilateConv = mod.get_function("DilateConKernel")

block = (numRows,numCols,1)
grid = (1,1)
dilateConv(
    # inputs
    input_gpu,
    output_gpu,
    mask_gpu,
    block = (4,4, 1),
    grid = grid
)
