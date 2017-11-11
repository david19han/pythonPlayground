#result gives the largest distance of values farthest

def solution(A):
    #hash = {}
    N = len(A)
    currMax = 0
    max = N-1

    for i in xrange(N):
        key = A[i]
        buffResult = N - 1 - i
        window = 0
        for j in range(N-1,i,-1):
            curr = A[j]
            if(key == curr):
                buffResult = buffResult - window
            else:
                window = window +1

        currMax = max(currMax,buffResult)

    return currMax
solution([1,2,3,4])
