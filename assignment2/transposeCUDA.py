#!/usr/bin/env python
# -*- coding: utf-8 -*-

 import numpy as np
 from pycuda import driver, compiler, gpuarray, tools

 # -- initialize the device
 import pycuda.autoinit

 kernel_code_template = """
 __global__ void MatrixMulKernel(float *a, float *c)
 {
     // 2D Thread ID (assuming that only *one* block will be executed)
     int tx = threadIdx.x;
     int ty = threadIdx.y;



     // Each thread loads one row of M and one column of N,
     //   to produce one element of P.
     for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
         float Aelement = a[ty * %(MATRIX_SIZE)s + k];
         float Belement = b[k * %(MATRIX_SIZE)s + tx];
         Pvalue += Aelement * Belement;
     }

     // Write the matrix to device memory;
     // each thread writes one element
     c[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
 }
 """
 times = []
 matrixSize = []

 for matrixSize in range(2,11):
     a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
     a_gpu = gpuarray.to_gpu(a_cpu)

     c_gpu = gpuarray.empty((matrixSize,matrixSize), np.float32)
     kernel_code = kernel_code_template % {
         'MATRIX_SIZE': matrixSize
         }

     mod = compiler.SourceModule(kernel_code)
     matrixTranspose = mod.get_function("matrixTranspose")

     start = time.time()
     matrixTranspose(
        a_gpu,#input
        c_gpu,#output
        block = (matrixSize, matrixSize, 1),
     )
     end = time.time() - start
     matrixSize.append(matrixSize)
     times.append(end)
