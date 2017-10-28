import numpy as np
import time



for MATRIX_SIZE in range(2,11):
    print "matrix size is",MATRIX_SIZE
    # create two random square matrices
    matrixA = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    print "MatrixA is :"
    #print matrixA

    #print len(matrixA)#num of rows
    #print len(matrixA[0])#number of cols

    matrixAt = np.empty([MATRIX_SIZE, MATRIX_SIZE])

    start = time.time()
    for i in range(0,len(matrixA)):
        for j in range(0,len(matrixA[0])):
            matrixAt[i][j] = matrixA[j][i]
    end = time.time()-start

    print "time elapsed is ",end
    print "MatrixAt is :"
    #print matrixAt
    print " "
