import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit
import pycuda.driver as drv

import time

from tabulate import tabulate
import pandas as pd

def CustomPrintTime(py_time, naive_time, opt_time):
## Print running time for cpu, naive and optimized algorithms
## Arguments: Each argument is a list of length 3 that contains the running times for three cases
 
    if len(py_time) != len(naive_time) or len(py_time) != len(opt_time) or len(py_time) != 3:
        raise Exception('All lists should be 3, but get {}, {}, {}'.format(len(py_time), len(naive_time), len(opt_time)))
    headers = ['small', 'medium', 'large']
    py_time = ['python'] + py_time
    naive_time = ['naive'] + naive_time
    opt_time = ['opt'] + opt_time
    table = [py_time, naive_time, opt_time]
    print "Time Taken by Kernels:"
    print tabulate(table, headers, tablefmt='fancy_grid').encode('utf-8')
 
##----------------------------------------------------------------------------------------------------------------------
 
def CustomPrintHistogram(histogram):
## Print the histogram
## Argument: a list of length 18 in which each element i represents the number of elements fall into the bin i.
 
    if len(histogram) != 18:
        raise Exception('length of the histogram must be 18, but get{}'.format(len(histogram)))
    header = ["{}".format(i) for i in range(18)]
    table = [histogram]
    print 'Histogram:'
    print tabulate(table, header, tablefmt='fancy_grid').encode('utf-8')
 
##----------------------------------------------------------------------------------------------------------------------
 
def CustomPrintSpeedUp(naive_kernel, opt_kernel):
## Print the speed up
## Arguments: The first argument is the naive kernel running time and the second is optimized version
## Each argument is the length of 3.
 
    if len(opt_kernel) != len(naive_kernel) or len(opt_kernel) != 3:
        raise Exception('lenght of naive_kernel and opt_kernel must be 3, but get {}, {}'.format(len(naive_kernel), len(opt_kernel)))
    speedup = [[s * 1.0/t for s, t in zip(naive_kernel, opt_kernel)]]
    print "Speedup(Naive/Optimized):"
    header = ['small_image', 'medium_image', 'large_image']
    print tabulate(speedup, header, tablefmt='fancy_grid').encode('utf-8')
 
##----------------------------------------------------------------------------------------------------------------------
 
def getData(path, mode):
## Get the input data
## Path: The path from which we extract data
## mode: size of teh data returned. 0 for first 2^20, 1 for 2^26 and 2 for full data
 
    data = np.memmap(path, dtype=np.uint16, mode='r')
    data = data.reshape(2**15, 2**15)
    if mode == 0:
        return data[:2**10, :2**10]
    if mode == 1:
        return data[:2**13, :2**13]
    if mode == 2:
        return data
    raise Exception('mode must be one of 0, 1, 2, but get {}'.format(mode))
 
##----------------------------------------------------------------------------------------------------------------------
 
def CustomHistEqual(py, naive, opt):
## Check the equality of the histograms
## Arguments: each argument is a list containing same amount of bins
 
    if len(py) != len(naive) or len(py) != len(opt):
        raise Exception('All length must be equal, but get {}, {}, {}'.format(len(py), len(naive), len(opt)))
    py_naive = True
    py_opt = True
    if not np.all(py == naive):
        print "Python != Naive"
        py_naive = False
    if not np.all(py == opt):
        print "Python != Opt"
        py_opt = False
    if py_naive and py_opt:
        print "All the histograms are equal!!"
 
##----------------------------------------------------------------------------------------------------------------------
 
def histogram(data, exponent = 10):
## Calculate the histogram
## data: A 2-D numpy array
## exponent: exponent of two. The sub-region size is 2^exponet by 2^exponent
## This function outputs a 1D array rather than a list. You must transform it to a list before you
## call CustomPrintFunction
 
    base = np.power(2, exponent).astype(np.int32)
    side = int(data.shape[0] / base)
    num_bins = side**2
    bins = np.zeros((num_bins, 18))
    for i in range(side):
        for j in range(side):
            hist = np.histogram(data[i*base:(i+1)*base, j*base:(j+1)*base], np.arange(0, 181, 10))
            bin_idx = i * side + j
            bins[bin_idx,:] = hist[0]
    bins = bins.reshape(-1)
    return bins

# print("Sequential 2^10x2^10")
# data0 = getData('hist_data.dat',0)
# hgram10 = histogram(data0)
# CustomPrintHistogram(list(hgram10))

# print("Sequential 2^13x2^13")
# data1 = getData('hist_data.dat',1)
# hgram13 = histogram(data1)
# CustomPrintHistogram(list(hgram13[:18]))
# len13 = len(hgram13)
# CustomPrintHistogram(list(hgram13[18:36]))
# CustomPrintHistogram(list(hgram13[len13-18:len13+1]))

print("Sequential 2^15x2^15")
data2 = getData('hist_data.dat',2)
hgram15 = histogram(data2)
len15 = len(hgram15)
CustomPrintHistogram(list(hgram15[:18]))
CustomPrintHistogram(list(hgram15[len15-18:len15+1]))


kernel_code_template = """
#include <stdio.h>
#include <math.h>

__global__ void naiveHisto(int *data,int* histogram,int size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < size && row < size){
        int index = col + row * size;
        int value = data[index];
        int bIndex = value/10;

        int rowRegion = row/1024;
        int colRegion = col/1024;

        int numBox = size/1024;

        int binRegion = colRegion + rowRegion * numBox;
        bIndex += binRegion*18;

        atomicAdd(&histogram[bIndex],1);
    }    
}
"""

# kernel_code_template = """
# #include <stdio.h>
# #include <math.h>

# __global__ void naiveHisto(int *data,int* histogram,int size)
# {
#     printf("Value is %d",data[threadIdx.x]);
#     int value = data[threadIdx.x];
#     int bIndex = value/10;
#     printf("bIndex is %d",bIndex);
# }   
# """

smallBins = 18
medBins = 18*64
largeBins = 18*1024

smallMatrix = 1024
medMatrix = np.power(2,13)
largeMatrix = np.power(2,15)

# compile the kernel code
mod = compiler.SourceModule(kernel_code_template)
# get the kernel function from the compiled module
naiveHisto = mod.get_function("naiveHisto")
blockSize = 32

# small_gpu = gpuarray.zeros(smallBins, np.int32)
# input_gpu_small = gpuarray.to_gpu(data0.astype('int32')) 

# print("GPU for Small Matrix:")
# naiveHisto(
#             # inputs
#             input_gpu_small, #1024x1024
#             small_gpu,
#             np.int32(smallMatrix),
#             block = (blockSize,blockSize,1),
#             grid = (smallMatrix/blockSize,smallMatrix/blockSize,1)
#             )
# print(np.array_equal(small_gpu.get(),hgram10.astype('int32')))

# print("GPU for Medium Matrix:")
# med_gpu = gpuarray.zeros(medBins,np.int32)
# input_gpu_med = gpuarray.to_gpu(data1.astype('int32'))
# naiveHisto(
#             # inputs
#             input_gpu_med, 
#             med_gpu,
#             np.int32(medMatrix),
#             block = (blockSize,blockSize,1),
#             grid = (medMatrix/blockSize,medMatrix/blockSize,1)
#             )
# CustomPrintHistogram(list(hgram13[:18]))
# CustomPrintHistogram(med_gpu.get()[:18])
# print(np.array_equal(med_gpu.get(),hgram13.astype('int32')))

# print("GPU for Large Matrix:")
# large_gpu = gpuarray.zeros(largeBins,np.int32)
input_gpu_large = gpuarray.to_gpu(data2.astype('int32'))
# naiveHisto(
#             # inputs
#             input_gpu_large, #1024x1024
#             large_gpu,
#             np.int32(largeMatrix),
#             block = (blockSize,blockSize,1),
#             grid = (largeMatrix/blockSize,largeMatrix/blockSize,1)
#             )
# print(np.array_equal(large_gpu.get(),hgram15.astype('int32')))


kernel_opt_template = """
#include <stdio.h>
#include <math.h>

__global__ void optimizeHisto(int *data,int* globalHisto,int size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


    __shared__ unsigned int localHisto[18];

    for(int i = 0;i<18;i++){
        localHisto[i] = 0;
    }
    __syncthreads();

    if(col<size && row<size){
        int index = col + row * size;
        int value = data[index];
        int bIndex = value/10;
        atomicAdd(&localHisto[bIndex],1);
    }
    __syncthreads();


    //index into right 18 bin set 

    if(threadIdx.x < 19 && threadIdx.y==0){

        int rowRegion = row/1024;
        int colRegion = col/1024;

        int numBox = size/1024;
        int binRegion = colRegion + rowRegion * numBox;
        int gIndex = threadIdx.x + binRegion*18;

        atomicAdd(&globalHisto[gIndex],localHisto[threadIdx.x]);
    }
}
"""

# compile the kernel code
mod = compiler.SourceModule(kernel_opt_template)
# get the kernel function from the compiled module
optoHisto = mod.get_function("optimizeHisto")

# small_gpu_opt = gpuarray.zeros(smallBins, np.int32)
# print("Optimized GPU for Small Matrix:")
# optoHisto(
#             # inputs
#             input_gpu_small, 
#             small_gpu_opt,
#             np.int32(smallMatrix),
#             block = (blockSize,blockSize,1),
#             grid = (smallMatrix/blockSize,smallMatrix/blockSize,1)
#             )
# print(np.array_equal(small_gpu_opt.get(),hgram10.astype('int32')))

# med_gpu_opt = gpuarray.zeros(medBins, np.int32)
# print("Optimized GPU for Medium Matrix:")
# optoHisto(
#             # inputs
#             input_gpu_med, 
#             med_gpu_opt,
#             np.int32(medMatrix),
#             block = (blockSize,blockSize,1),
#             grid = (medMatrix/blockSize,medMatrix/blockSize,1)
#             )
# print(np.array_equal(med_gpu_opt.get(),hgram13.astype('int32')))
# CustomPrintHistogram(med_gpu_opt.get()[:18])
# print("optGPU")
# CustomPrintHistogram(med_gpu_opt.get()[18:36])
# CustomPrintHistogram(med_gpu_opt.get()[len13-18:len13+1])

large_gpu_opt = gpuarray.zeros(largeBins, np.int32)
print("Optimized GPU for Medium Matrix:")
optoHisto(
            # inputs
            input_gpu_large, 
            large_gpu_opt,
            np.int32(largeMatrix),
            block = (blockSize,blockSize,1),
            grid = (largeMatrix/blockSize,largeMatrix/blockSize,1)
            )
print(np.array_equal(large_gpu_opt.get(),hgram15.astype('int32')))


