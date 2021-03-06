#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

 # -- initialize the device
import pycuda.autoinit

matrixSize = 4
MATRIX_SIZE = matrixSize
kernel_code_template = """
 #include <stdio.h>
 __global__ void MatrixMulKernel(float *a, float *c)
 {
     // 2D Thread ID (assuming that only *one* block will be executed)
     int tx = threadIdx.x;
     int ty = threadIdx.y;

     printf("tx is %d|ty is %d",tx,ty);
 }
 """
times = []
matrixSize = []

for matrixSize in range(2,11):
     a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
     a_gpu = gpuarray.to_gpu(a_cpu)

     c_gpu = gpuarray.empty((matrixSize,matrixSize), np.float32)


     mod = compiler.SourceModule(kernel_code_template)

     matrixTranspose = mod.get_function("MatrixMulKernel")

     start = time.time()
     matrixTranspose(
        a_gpu,#input
        c_gpu,#output
        block = (matrixSize, matrixSize, 1),
     )
     end = time.time() - start
     matrixSize.append(matrixSize)
     times.append(end)
