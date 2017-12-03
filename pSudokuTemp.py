
import time
from tabulate import tabulate
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
 
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
 
class Node:
    position = (0,0)
    possibleValues = []

def buildGrid(gridString,size):
    grid = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            grid[i][j] = gridString[i*size+j]
    return grid

def buildGrid2(list2,size):
    grid = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            grid[i][j] = int(list2[i*size+j])
    return grid

def getEmptySpaces(grid):
    list = []
    for i in range(size):
        for j in range(size):
            if(grid[i][j]==0):
                list.append((i,j))
    return list

#create nodes for empty spaces
def createEmptySpacesNode(grid,size):
    empty_spaces = []
    for i in range(size):
        for j in range(size):
            if(grid[i][j] == 0):
                list = findPossibleValues(grid,(i,j))
                if(len(list)==1):
                    grid[i][j] = list[0]
                else:
                    n = Node()
                    n.position = (i,j)
                    n.possibleValues = list
                    empty_spaces.append(n)
    return empty_spaces

def createEmptySpacesPositionList(grid,size):
    empty_spaces_position_i = []
    empty_spaces_position_j = []
    for i in range(size):
        for j in range(size):
            if(grid[i][j] == 0):
                list = findPossibleValues(grid,(i,j))
                if(len(list)==1):
                    grid[i][j] = list[0]
                else:
                    empty_spaces_position_i.append(i)
                    empty_spaces_position_j.append(j)
    return (empty_spaces_position_i,empty_spaces_position_j)

def findPossibleValues(grid,position):
    list = [x for x in range(1,size+1)]
    row = position[0]
    col = position[1]
    
    #check for row
    for i in range(size):
        if(grid[row][i] in list):
            list.remove(grid[row][i])
    #check for column
    for i in range(size):
        if(grid[i][col] in list):
            list.remove(grid[i][col])
    #this square
    sqrtSize=np.sqrt(size).astype(int)
    startRow=(row//sqrtSize)*sqrtSize
    startCol=(col//sqrtSize)*sqrtSize
    for i in range(sqrtSize):
        for j in range(sqrtSize):
            if(grid[i+startRow][j+startCol] in list):
                list.remove(grid[i+startRow][j+startCol])
    return list

#sort empty_spaces list by number of possible values
def getPossibleValues(n):
    list = n.possibleValues
    return len(list)

#run sudoku
def runSudoku(grid,empty_spaces):
    stack = []
    while(len(empty_spaces)!=0):
        #move empty_space node to stack
        stack.append(empty_spaces[0])
        empty_spaces.pop(0)
        
        #get bottom of stack 
        currentStackIndex=len(stack)-1
        stackNode = stack[currentStackIndex]
        
        #check possibleValues
        pValues = stackNode.possibleValues
        while (len(pValues) == 0):
            #backtrack
            n = stack.pop()
            nPos = n.position
            grid[nPos[0]][nPos[1]] = 0
            
            empty_spaces.insert(0,n)
            
            currentStackIndex=len(stack)-1
            stackNode = stack[currentStackIndex]
            pValues = stackNode.possibleValues
            #change value of bottom of current nodes 
  
        #else:
        pos = stackNode.position
        grid[pos[0]][pos[1]] = stackNode.possibleValues[0]

        pValues.pop(0)
        stackNode.possibleValues = pValues # just in case 

        #recalcualte possible values for top of empty_spaces
        if(len(empty_spaces)>0):
            for n in empty_spaces:
                n.possibleValues = findPossibleValues(grid,n.position)
            empty_spaces.sort(key=getPossibleValues)
            
    return grid
def checkGrid(grid,size):
    #check rows
    for i in range(size):
        listR = [x for x in range(1,size+1)]
        listC = [x for x in range(1,size+1)]
        for j in range(size):
            if grid[i][j] in listR:
                listR.remove(grid[i][j])
            else:
                return False
            if grid[j][i] in listC:
                listC.remove(grid[j][i])
            else:
                return False
    #check squares:
    sqrtSize=np.sqrt(size).astype(int)
    rowList=[x*sqrtSize for x in range(0,sqrtSize)]
    colList=[x*sqrtSize for x in range(0,sqrtSize)]
    print(rowList,colList)
    for row in rowList:
        for col in colList:
            list = [x for x in range(1,size+1)]
            startRow=(row//sqrtSize)*sqrtSize
            startCol=(col//sqrtSize)*sqrtSize
            for i in range(sqrtSize):
                for j in range(sqrtSize):
                    if(grid[i+startRow][j+startCol] in list):
                        list.remove(grid[i+startRow][j+startCol])
                    else:
                        return False
    return True


# Define the CUDA kernel
kernel_code_template = """
#include<stdio.h> 


//Compute the Possible Values List
//True-constraints exist
//False-constraints do not exist, possible to be the value
//Set all values that appeared in the particular row,column and box to true
__device__ void computePossibleValues(int* grid, bool* possibleValueList,unsigned int i,unsigned int j)
{
    unsigned int a,b;
    unsigned int sqrtSize=(unsigned int)sqrt((float)9);

    for(a=0;a<9;a++)
    {
        possibleValueList[a]=false;
    }
    for(a=0;a<9;a++)
    {
        possibleValueList[grid[i*9+a]-1]=true;
        possibleValueList[grid[a*9+j]-1]=true;
    }
    unsigned int startRow=(i/sqrtSize)*sqrtSize;
    unsigned int startCol=(j/sqrtSize)*sqrtSize;
    for(a=0;a<sqrtSize;a++)
    {
        for(b=0;b<sqrtSize;b++)
        {
            possibleValueList[grid[(a+startRow)*9+b+startCol]-1]=true;
        }
    }
}
//if found possibleValue that dont have constraints:
//set path=true to indicate there is valid path to move on
//otherwise, break loop
__device__ bool findPossibleValues(unsigned int &idx, bool* possibleValueList)
{
    bool path=false;
    for(;idx<9;idx++)
    {
        if(possibleValueList[idx]==false)
        {
            path=true;
            break;
        }
    }
    return path;
}
__device__ void fillSpaces(int* grid)
{
    
}

__global__ void generateBoard(int* grid,int* empty_spaces_i, int* empty_spaces_j,int* out,int empty_spaces_length,int* threadCount) {
    unsigned int empty_spaces_count=0;
    bool path=false;
    unsigned int x,y;

    unsigned int idx;
    unsigned int i,j;

    bool possibleValueList[9];

    unsigned int emptyspaces_fillcount=0;

    unsigned int possibleValuesIdx[9*9];

    //initialise possibleValueIdx array to 0
    for(unsigned int a=0;a<9*9;a++)
    {
        possibleValuesIdx[a]=0;
    }
    while((threadCount[0]<3000)&&(emptyspaces_fillcount<empty_spaces_length))
    {
        //fill in emptyspaces and call more
        i = empty_spaces_i[empty_spaces_count];
        j = empty_spaces_j[empty_spaces_count];

        computePossibleValues(grid,possibleValueList,i,j);

        //start from possibleValuesIdx to prevent doing the same node again for backtracking
        idx=possibleValuesIdx[i*9+j];
        path=findPossibleValues(idx,possibleValueList);

        if(empty_spaces_count==4) //store results and start backtrack
        {
            path=false;

            //output grid for every new board
            for (x=0;x<9;x++)
            {
                for(y=0;y<9;y++)
                {
                    out[threadCount[0]*9*9+y*9+x]=grid[y*9+x];
                }
            }
            threadCount[0]+=1;
        }
        while(path==false)
        {
            //backtrack
            possibleValuesIdx[i*9+j]=0; //reset start index
            grid[i*9+j] = 0;  //reset grid
            empty_spaces_count -=1;

            if(empty_spaces_count==-1)
                break;

            i = empty_spaces_i[empty_spaces_count];
            j = empty_spaces_j[empty_spaces_count];

            computePossibleValues(grid,possibleValueList,i,j);

            idx=possibleValuesIdx[i*9+j];
            path=findPossibleValues(idx,possibleValueList);
        }    
        if(empty_spaces_count==-1)
            break;
        grid[i*9+j] = idx+1; //set value
        possibleValuesIdx[i*9+j]=idx+1; //next time check for possible values, start from next value
        empty_spaces_count +=1;
    }
    grid[0]=3;
}

__global__ void runSudokuKernel(int* more_grid,int* empty_spaces_i,int* empty_spaces_j,int empty_spaces_length,int* flag) {
    //keep private copy of specific board/grid
    const int totalSize = 9*9;
    int grid[totalSize];
    for(int i =0;i<totalSize;i++){
        grid[i] = more_grid[threadIdx.x*totalSize+i];
    }
    for(int i = 0;i<totalSize;i++){
       printf("%d",grid[i]);
    }

    printf("finished\\n");
    unsigned int empty_spaces_count=0;
    unsigned int idx;
    unsigned int i,j;

    unsigned int possibleValuesIdx[9*9];

    //initialise possibleValueIdx array to 0
    for(unsigned int a=0;a<9*9;a++)
    {
        possibleValuesIdx[a]=0;
    }

    bool possibleValueList[9];
    bool path;
    while(flag[0]==0 && empty_spaces_count<empty_spaces_length)
    {    
        printf("%s %d\\n","empty_spaces_count is ", empty_spaces_count);
        i = empty_spaces_i[empty_spaces_count];
        j = empty_spaces_j[empty_spaces_count];

        computePossibleValues(grid,possibleValueList,i,j);

        //start from possibleValuesIdx to prevent doing the same node again for backtracking
        idx=possibleValuesIdx[i*9+j];
        path=findPossibleValues(idx,possibleValueList);

        while(path==false)
        {
            //backtrack
            possibleValuesIdx[i*9+j]=0; //reset start index
            printf("%s %d\\n","i*9+j is",i*9+j);
            grid[i*9+j] = 0;  //reset grid
            empty_spaces_count -=1;

            printf("%s %d\\n","empty_spaces_count is ", empty_spaces_count);
            i = empty_spaces_i[empty_spaces_count];
            j = empty_spaces_j[empty_spaces_count];

            computePossibleValues(grid,possibleValueList,i,j);

            idx=possibleValuesIdx[i*9+j];
            path=findPossibleValues(idx,possibleValueList);

        }    
        printf("%s %d\\n","i*9+j is",i*9+j);
        grid[i*9+j] = idx+1; //set value
        possibleValuesIdx[i*9+j]=idx+1; //next time check for possible values, start from next value
        empty_spaces_count +=1;
    }
    flag[0]=1;
    printf("KERNEL DONE\\n");
}
"""
######################################## Sequential Code ###########################################
MAXBOARD = 64

# gridString = '003020600900305001001806400008102900700000008006708200002609500800203009005010300'
gridString = '000001000020000008691200000000000014102506003800020506005000000730000000006319405'
size = int(np.sqrt(len(gridString)))
grid = buildGrid(gridString,size)

# grid16 = '''0 15 0 1 0 2 10 14 12 0 0 0 0 0 0 0 
# 0 6 3 16 12 0 8 4 14 15 1 0 2 0 0 0
# 14 0 9 7 11 3 15 0 0 0 0 0 0 0 0 0
# 4 13 2 12 0 0 0 0 6 0 0 0 0 15 0 0
# 0 0 0 0 14 1 11 7 3 5 10 0 0 8 0 12
# 3 16 0 0 2 4 0 0 0 14 7 13 0 0 5 15
# 11 0 5 0 0 0 0 0 0 9 4 0 0 6 0 0
# 0 0 0 0 13 0 16 5 15 0 0 12 0 0 0 0
# 0 0 0 0 9 0 1 12 0 8 3 10 11 0 15 0
# 2 12 0 11 0 0 14 3 5 4 0 0 0 0 9 0
# 6 3 0 4 0 0 13 0 0 11 9 1 0 12 16 2
# 0 0 10 9 0 0 0 0 0 0 12 0 8 0 6 7
# 12 8 0 0 16 0 0 10 0 13 0 0 0 5 0 0
# 5 0 0 0 3 0 4 6 0 1 15 0 0 0 0 0
# 0 9 1 6 0 14 0 11 0 0 2 0 0 0 10 8
# 0 14 0 0 0 13 9 0 4 12 11 8 0 0 2 0'''
# temp = grid16.replace('\n'," ")
# nums16= temp.split(" ")
# nums16.remove('')
# size=int(np.sqrt(len(nums16)))
# grid = buildGrid2(nums16,size)

empty_spaces=createEmptySpacesNode(grid,size)
empty_spaces.sort(key=getPossibleValues)

start = time.time()
runSudoku(grid,empty_spaces)
pythontime=time.time()-start
print(checkGrid(grid,size))

# grid2 = buildGrid('358471629427963158691285347569738214142596783873124596915647832734852961286319475',size)

# print(np.equal(grid,grid2))
completedgrid=grid

# grid16 = '''0 15 0 1 0 2 10 14 12 0 0 0 0 0 0 0 
# 0 6 3 16 12 0 8 4 14 15 1 0 2 0 0 0
# 14 0 9 7 11 3 15 0 0 0 0 0 0 0 0 0
# 4 13 2 12 0 0 0 0 6 0 0 0 0 15 0 0
# 0 0 0 0 14 1 11 7 3 5 10 0 0 8 0 12
# 3 16 0 0 2 4 0 0 0 14 7 13 0 0 5 15
# 11 0 5 0 0 0 0 0 0 9 4 0 0 6 0 0
# 0 0 0 0 13 0 16 5 15 0 0 12 0 0 0 0
# 0 0 0 0 9 0 1 12 0 8 3 10 11 0 15 0
# 2 12 0 11 0 0 14 3 5 4 0 0 0 0 9 0
# 6 3 0 4 0 0 13 0 0 11 9 1 0 12 16 2
# 0 0 10 9 0 0 0 0 0 0 12 0 8 0 6 7
# 12 8 0 0 16 0 0 10 0 13 0 0 0 5 0 0
# 5 0 0 0 3 0 4 6 0 1 15 0 0 0 0 0
# 0 9 1 6 0 14 0 11 0 0 2 0 0 0 10 8
# 0 14 0 0 0 13 9 0 4 12 11 8 0 0 2 0'''
# temp = grid16.replace('\n'," ")
# nums16= temp.split(" ")
# nums16.remove('')
# size=int(np.sqrt(len(nums16)))
# grid = buildGrid2(nums16,size)

grid = buildGrid(gridString,size)

h_empty_spaces_i,h_empty_spaces_j= createEmptySpacesPositionList(grid,size)

h_boardcount=np.zeros((1,1)).astype(np.int32)
h_flag=np.zeros((1,1)).astype(np.int32)
d_boardcount = gpuarray.to_gpu(h_boardcount)
d_flag = gpuarray.to_gpu(h_flag)
# Define the input and output to process.
h_grid = np.array(grid).flatten().astype(np.int32)

h_possibleValuesIdx=np.zeros((size,size)).astype(np.int32)
h_moregrid = np.zeros((MAXBOARD,size,size)).astype(np.int32)
# print(len(h_moregrid),shape(h_moregrid))
d_grid = gpuarray.to_gpu(h_grid)
d_empty_spaces_i = gpuarray.to_gpu(np.array(h_empty_spaces_i).astype(np.int32))
d_empty_spaces_j = gpuarray.to_gpu(np.array(h_empty_spaces_j).astype(np.int32))
d_possibleValuesIdx=gpuarray.to_gpu(h_possibleValuesIdx)

d_moregrid = gpuarray.to_gpu(h_moregrid)
######################################## Parallel Code ###########################################
# get the kernel code from the template 
# by specifying the constant MATRIX_SIZE
    
# compile the kernel code
mod = compiler.SourceModule(kernel_code_template)

# # get the kernel function from the compiled module
runSudokuKernelFn = mod.get_function("runSudokuKernel")
generateBoardFn = mod.get_function("generateBoard")

start = time.time()
generateBoardFn(
    d_grid,
    d_empty_spaces_i,
    d_empty_spaces_j,
    d_moregrid,
    np.int32(len(h_empty_spaces_i)),
    d_boardcount,
    block = (1, 1,1), )

print(len(d_moregrid.get()))
# print(d_empty_spaces_i.get())
# print(d_empty_spaces_i.get()[4:len(h_empty_spaces_i)])
# for i in range(2):
# boardcount=d_boardcount.get()[0]
runSudokuKernelFn(
    d_grid, 
    d_empty_spaces_i,
    d_empty_spaces_j,
    np.int32(len(h_empty_spaces_i)),
    d_flag,
    block = (2, 1,1), )
cudanaivetime=time.time()-start
# print("number of board:",d_boardcount.get())
