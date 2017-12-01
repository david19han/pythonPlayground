from tabulate import tabulate
import numpy as np

import time
 
import pyopencl as cl
import pyopencl.array

print("OPENCL")
 
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
## Each argument is tthe length of 3.
 
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
    int col = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int row = get_group_id(1) * get_local_size(1) + get_local_id(1);

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

smallBins = 18
medBins = 18*64
largeBins = 18*1024

smallMatrix = 1024
medMatrix = np.power(2,13)
largeMatrix = np.power(2,15)

arraysize_x = []
arraysize_x.append(smallMatrix)
arraysize_x.append(medMatrix)
arraysize_x.append(largeMatrix)

seqTimes = []


print("Sequential 2^10x2^10")
data0 = getData('hist_data.dat',0)
start = time.time()
hgram10 = histogram(data0)
seqTimes.append(time.time()-start)
CustomPrintHistogram(list(hgram10))

print "-" * 80

print("Sequential 2^13x2^13")
data1 = getData('hist_data.dat',1)
start = time.time()
hgram13 = histogram(data1)
seqTimes.append(time.time()-start)
CustomPrintHistogram(list(hgram13[:18]))
len13 = len(hgram13)
CustomPrintHistogram(list(hgram13[len13-18:len13+1]))

print "-" * 80

print("Sequential 2^15x2^15")
data2 = getData('hist_data.dat',2)
start = time.time()
hgram15 = histogram(data2)
seqTimes.append(time.time()-start)
len15 = len(hgram15)
CustomPrintHistogram(list(hgram15[:18]))
CustomPrintHistogram(list(hgram15[len15-18:len15+1]))

print "-" * 80

naiveTimes = [] 

print("Naive GPU for Small Matrix:")
input_gpu_small = cl.array.to_device(queue,data0.astype('int32'))
output_gpu_zeros_small = np.zeros(smallBins,'int32') 
output_gpu_small = cl.array.to_device(queue,output_gpu_zeros_small.astype('int32'))

prg = cl.Program(ctx, naiveKernel).build()
start = time.time() 
prg.func(queue,(smallMatrix,smallMatrix),(32,32),input_gpu_small.data,output_gpu_small.data,np.int32(smallMatrix))
naiveTimes.append(time.time()-start)
CustomPrintHistogram(output_gpu_small.get()[:18])
print(np.array_equal(output_gpu_small.get(),hgram10.astype('int32')))

print "-" * 80

print("Naive GPU for Medium Matrix:")
input_gpu_med = cl.array.to_device(queue,data1.astype('int32'))
output_gpu_zeros_med = np.zeros(medBins,'int32') 
output_gpu_med = cl.array.to_device(queue,output_gpu_zeros_med.astype('int32'))

prg = cl.Program(ctx, naiveKernel).build()
start = time.time()
prg.func(queue,(medMatrix,medMatrix),(32,32),input_gpu_med.data,output_gpu_med.data,np.int32(medMatrix))
naiveTimes.append(time.time()-start)
CustomPrintHistogram(output_gpu_med.get()[:18])
CustomPrintHistogram(output_gpu_med.get()[len13-18:len13+1])
print(np.array_equal(output_gpu_med.get(),hgram13.astype('int32')))

print "-" * 80

print("Naive GPU for Large Matrix:")
input_gpu_large = cl.array.to_device(queue,data2.astype('int32'))
output_gpu_zeros_large = np.zeros(largeBins,'int32') 
output_gpu_large = cl.array.to_device(queue,output_gpu_zeros_large.astype('int32'))

prg = cl.Program(ctx, naiveKernel).build()
start = time.time()
prg.func(queue,(largeMatrix,largeMatrix),(32,32),input_gpu_large.data,output_gpu_large.data,np.int32(largeMatrix))
naiveTimes.append(time.time()-start)
CustomPrintHistogram(output_gpu_large.get()[:18])
CustomPrintHistogram(output_gpu_large.get()[len15-18:len15+1])
print(np.array_equal(output_gpu_large.get(),hgram15.astype('int32')))

print "-" * 80

# optKernel = """
# __kernel void func(__global int* data, __global int* histogram, int size) {
#     int col = get_group_id(0) * get_local_size(0) + get_local_id(0);
#     int row = get_group_id(1) * get_local_size(1) + get_local_id(1);

#     __local int localHisto[18];

#     for(int i = 0;i<18;i++){
#         localHisto[i] = 0;
#     }

#     barrier(CLK_LOCAL_MEM_FENCE);

#     if(col<size && row<size){
#         int index = col + row * size;
#         int value = data[index];
#         int bIndex = value/10;
#         atomic_add(&localHisto[bIndex],1);
#     }

#     barrier(CLK_LOCAL_MEM_FENCE);

#    //index into right 18 bin set 

#     if(get_local_id(0) < 18 && get_local_id(1)==0){

#         int rowRegion = row/1024;
#         int colRegion = col/1024;

#         int numBox = size/1024;
#         int binRegion = colRegion + rowRegion * numBox;
#         int gIndex = get_local_id(0) + binRegion*18;

#         atomic_add(&histogram[gIndex],localHisto[get_local_id(0)]);
#     }
# }
# """

optKernel = """
__kernel void func(__global int* data, __global int* histogram, int size) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    int x = get_local_id(0);
    int y = get_local_id(1);

    __local int localHisto[18];

    for(int i = 0;i<18;i++){
        localHisto[i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(col<size && row<size){
        int index = col + row * size;
        int value = data[index];
        int bIndex = value/10;
        atomic_add(&localHisto[bIndex],1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

   //index into right 18 bin set 

    if(x < 18 && y==0){

        int rowRegion = row/1024;
        int colRegion = col/1024;

        int numBox = size/1024;
        int binRegion = colRegion + rowRegion * numBox;
        int gIndex = x + binRegion*18;

        atomic_add(&histogram[gIndex],localHisto[x]);
    }
}
"""

optiTimes = []

print("Optimized GPU for Small Matrix:")
input_gpu_small = cl.array.to_device(queue,data0.astype('int32'))
output_gpu_zeros_small = np.zeros(smallBins,'int32') 
opt_gpu_small = cl.array.to_device(queue,output_gpu_zeros_small.astype('int32'))

prg = cl.Program(ctx, optKernel).build()
start = time.time()
prg.func(queue,(smallMatrix,smallMatrix),(32,32),input_gpu_small.data,opt_gpu_small.data,np.int32(smallMatrix))
optiTimes.append(time.time()-start)

CustomPrintHistogram(opt_gpu_small.get()[:18])
print(np.array_equal(opt_gpu_small.get(),hgram10.astype('int32')))

print "-" * 80

print("Optimized GPU for Medium Matrix:")
input_gpu_med = cl.array.to_device(queue,data1.astype('int32'))
output_gpu_zeros_med = np.zeros(medBins,'int32') 
opt_gpu_med = cl.array.to_device(queue,output_gpu_zeros_med.astype('int32'))

prg = cl.Program(ctx, optKernel).build()
start=time.time()
prg.func(queue,(medMatrix,medMatrix),(32,32),input_gpu_med.data,opt_gpu_med.data,np.int32(medMatrix))
optiTimes.append(time.time()-start)

CustomPrintHistogram(opt_gpu_med.get()[:18])
CustomPrintHistogram(opt_gpu_med.get()[len13-18:len13+1])
print(np.array_equal(opt_gpu_med.get(),hgram13.astype('int32')))

print "-" * 80

print("Optimized GPU for Large Matrix:")
input_gpu_large = cl.array.to_device(queue,data2.astype('int32'))
output_gpu_zeros_large = np.zeros(largeBins,'int32') 
opt_gpu_large = cl.array.to_device(queue,output_gpu_zeros_large.astype('int32'))

start=time.time()
prg = cl.Program(ctx, optKernel).build()
optiTimes.append(time.time()-start)
prg.func(queue,(largeMatrix,largeMatrix),(32,32),input_gpu_large.data,opt_gpu_large.data,np.int32(largeMatrix))

CustomPrintHistogram(opt_gpu_large.get()[:18])
CustomPrintHistogram(opt_gpu_large.get()[len15-18:len15+1])
print(np.array_equal(opt_gpu_large.get(),hgram15.astype('int32')))

print "-" * 80
print("Custom Print Time")
CustomPrintTime(seqTimes,naiveTimes,optiTimes)
print "-" * 80
print("SeqTimes: ")
print(seqTimes)
print "-" * 80
print("NaiveTimes: ")
print(naiveTimes)
print "-" * 80
print("OptiTimes: ")
print(optiTimes)

print "-" * 80
print("Custom Print Speedup")
CustomPrintSpeedUp(naiveTimes,optiTimes)

print "-" * 80
print("Custom Histogram Equal")
print("Size 2^10x2^10")
CustomHistEqual(hgram10, output_gpu_small.get(), opt_gpu_small.get())

print("Size 2^13x2^13")
CustomHistEqual(hgram13, output_gpu_med.get(), opt_gpu_med.get())

print("Size 2^15x2^15")
CustomHistEqual(hgram15, output_gpu_large.get(), opt_gpu_large.get())

print "-" * 80

#plot
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#####################  
plt.gcf() 
plt.plot(arraysize_x, seqTimes, 'r')
plt.xlabel('Array Size')
plt.ylabel('SeqTime')
plt.gca().set_ylim((min(seqTimes),max(seqTimes)))
plt.legend(handles=[])
plt.savefig('SeqCL.png')

#####################  
plt.gcf() 
plt.plot(arraysize_x, naiveTimes, 'r')
plt.xlabel('Array Size')
plt.ylabel('NaiveTime')
plt.legend(handles=[])
plt.gca().set_ylim((min(naiveTimes),max(naiveTimes)))
plt.savefig('NaiveCL.png')

##################### 
plt.gcf()  
plt.plot(arraysize_x, optiTimes, 'r')
plt.xlabel('Array Size')
plt.ylabel('OptimizedTime')
plt.gca().set_ylim((min(optiTimes),max(optTimes)))
plt.legend(handles=[])
plt.savefig('OptimizedCL.png')

# red dashes, blue squares and green triangles
plt.gcf()  
plt.plot(arraysize_x, seqTimes, 'r', arraysize_x, naiveTimes, 'b', arraysize_x, optiTimes, 'g')
plt.xlabel('Array Size')
plt.ylabel('Time')

red_patch = mpatches.Patch(color='red', label='Sequential')
blue_patch = mpatches.Patch(color='blue', label='Naive')
green_patch = mpatches.Patch(color='green', label='Optimized')

handles = []
handles.append(red_patch)
handles.append(blue_patch)
handles.append(green_patch)
plt.legend(handles=handles)
plt.savefig('CUDA.png')


# plt.gcf()
# plt.plot(arraysize_x,seqTimes)
# plt.xlabel('Array Size')
# plt.ylabel('Time')
# plt.gca().set_xlim((min(x),max(x)))
# plt.savefig('seqTimesCUDA.png')

# plt.gcf()
# plt.plot(arraysize_x,naiveTimes)
# plt.xlabel('Array Size')
# plt.ylabel('Time')
# plt.gca().set_xlim((min(x),max(x)))
# plt.savefig('seqTimesCUDA.png')

# plt.gcf()
# plt.plot(arraysize_x,optiTimes)
# plt.xlabel('Array Size')
# plt.ylabel('Time')
# plt.gca().set_xlim((min(x),max(x)))
# plt.savefig('seqTimesCUDA.png')







