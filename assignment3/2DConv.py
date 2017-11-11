import random
import numpy as np
import time
import copy
import scipy
from scipy import signal

#inputArr

maskSize = 3
numRows = 7
numCols = 9

stride = 2

inputArr = [[int(100*random.random()) for i in range(numCols)] for j in range(numRows)]
#inputArr = [[1, 2, 3], [4,5, 6], [7, 8, 9]]

mask = [[int(100*random.random()) for i in range(maskSize)] for j in range(maskSize)]
#mask = [[-1,-2,-1],[0,0,0],[1,2,1]]


#flip mask: return flip

#flip horizontally
flipHor= [[0 for i in range(maskSize)] for j in range(maskSize)]
for row in range(maskSize):
	window = maskSize - 1
	for col in range(maskSize-1,-1,-1):
		colIndex = abs(col-window)
		flipHor[row][colIndex] = mask[row][col]
#flip vertically
flip = copy.deepcopy(flipHor)
for col in range(maskSize):
	window = maskSize - 1
	for row in range(maskSize-1,-1,-1):
		rowIndex = abs(row-window)
		flip[rowIndex][col] = flipHor[row][col]

flip1d = []
mask1d = []
input1d = []
for i in range(maskSize):
	for j in range(maskSize):
		flip1d.append(flip[i][j])
		mask1d.append(mask[i][j])
		input1d.append(inputArr[i][j])

#for each cell, calculate new sum and store it into output 
output= [[0 for i in range(numCols)] for j in range(numRows)]

#calculated weighted sum each cell
for row in range(numRows):
	for col in range(numCols):
		surrValues = []
		inputValue = inputArr[row][col]

		for i in range(row-stride,row+stride+1,stride):
			for j in range(col-stride,col+stride+1,stride):
				if i < 0 or i>numRows-1 or j<0 or j>numCols-1:
					surrValues.append(0)
				else:
					surrValues.append(inputArr[i][j])

		#calculate output value
		outputValue = 0
		for i in range(len(flip1d)):
			#print(flip1d[i],surrValues[i])
			prod = flip1d[i] * surrValues[i]
			outputValue = outputValue + prod
		output[row][col] = outputValue
print output

maskAdj = [[0 for i in range(stride*2+1)] for j in range(stride*2+1)]
a = 0
for i in range(0,stride*2+1,stride):
	for j in range(0,stride*2+1,stride):
		maskAdj[i][j] = mask1d[a]
		a = a +1


print signal.convolve2d(inputArr,mask,mode='same', boundary='fill', fillvalue=0)
print np.array_equal(output,signal.convolve2d(inputArr,maskAdj,mode='same', boundary='fill', fillvalue=0))













	
		
