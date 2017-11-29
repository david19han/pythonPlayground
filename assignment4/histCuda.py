from tabulate import tabulate
import numpy as np
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

data0 = getData('hist_data.dat',0)
hgram10 = histogram(data0)
CustomPrintHistogram(list(hgram10))
print(data0.shape)
print(len(hgram10))
for i in xrange(len(hgram10)):
    print((hgram10[i]))

data1 = getData('hist_data.dat',1)
hgram13 = histogram(data1,13)
CustomPrintHistogram(list(hgram13))
print(data1.shape)
print(len(hgram13))
for i in xrange(len(hgram13)):
    print((hgram13[i]))

data2 = getData('hist_data.dat',2)
hgram15 = histogram(data2,15)
CustomPrintHistogram(list(hgram15))
print(data2.shape)
print(len(hgram15))
for i in xrange(len(hgram15)):
    print((hgram15[i]))

#naive kernel
kernel_code_template = """
#include <stdio.h>
#include <math.h>

__global__ void naiveHisto(const char* const data,int* histogram)
{
    int id_x = blockidx.x * blockDim.x + threadIdx.x;
    int id_y = blockidx.y * blockDim.y + threadIdx.y;

    int index = floor(x/10);
    atomicAdd(&histogram[index],1);

}
"""


