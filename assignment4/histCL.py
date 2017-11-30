from tabulate import tabulate
import numpy as np

import time
 
import pyopencl as cl
import pyopencl.array


 
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
naiveKernel = """
__kernel void func(__global int* data, __global int* histogram, int size) {
    int row = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int col = get_group_id(1) * get_local_size(1) + get_local_id(1);
    
    if(col<size && row < size){
        int index = col + row * size;
        int value = data[index];
        int bIndex = value/10;

        int rowRegion = row/1024;
        int colRegion = col/1024;

        int numBox = size/1024;

        int binRegion = colRegion + rowRegion * numBox;
        bIndex += binRegion*18;

        atomic_add(&histogram[bIndex],1);
    }
}
"""
# naiveKernel = """
# __kernel void func(__global int* histogram,int size) {
#     int col = get_group_id(0) * get_local_size(0) + get_local_id(0);
#     int row = get_group_id(1) * get_local_size(1) + get_local_id(1);
#     int index = col + row * size;
#     histogram[index] = index; 
# }
# """

print("Sequential 2^10x2^10")
data0 = getData('hist_data.dat',0)
hgram10 = histogram(data0)
CustomPrintHistogram(list(hgram10))

print("Sequential 2^13x2^13")
data1 = getData('hist_data.dat',1)
hgram13 = histogram(data1)
CustomPrintHistogram(list(hgram13[:18]))
len13 = len(hgram13)
CustomPrintHistogram(list(hgram13[len13-18:len13+1]))

# print("Sequential 2^15x2^15")
# data2 = getData('hist_data.dat',2)
# hgram15 = histogram(data2)
# len15 = len(hgram15)
# CustomPrintHistogram(list(hgram15[:18]))
# CustomPrintHistogram(list(hgram15[len15-18:len15+1]))

smallBins = 18
medBins = 18*64
largeBins = 18*1024

smallMatrix = 1024
medMatrix = np.power(2,13)
largeMatrix = np.power(2,15)

print("Naive GPU for Small Matrix:")
input_gpu_small = cl.array.to_device(queue,data0.astype('int32'))
output_gpu_small = cl.array.empty(queue, (18,), 'int32')

prg = cl.Program(ctx, naiveKernel).build()
prg.func(queue,(smallMatrix,smallMatrix),(32,32),input_gpu_small.data,output_gpu_small.data,np.int32(smallMatrix))

print(np.array_equal(output_gpu_small.get(),hgram10.astype('int32')))
CustomPrintHistogram(output_gpu_small.get()[:18])

# print("Naive GPU for Medium Matrix:")
# input_gpu_med = cl.array.to_device(queue,data1.astype('int32'))
# output_gpu_med = cl.array.empty(queue, (medBins,), 'int32')

# prg = cl.Program(ctx, naiveKernel).build()
# prg.func(queue,(medMatrix,medMatrix),(32,32),input_gpu_med.data,output_gpu_med.data,np.int32(medMatrix))

# print(np.array_equal(output_gpu_med.get(),hgram13.astype('int32')))
# # print(len(output_gpu_med.get()))
# CustomPrintHistogram(output_gpu_med.get()[:18])
# CustomPrintHistogram(output_gpu_med.get()[len13-18:len13+1])


print("Naive GPU for Large Matrix:")


