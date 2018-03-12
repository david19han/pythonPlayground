
# coding: utf-8

# In[70]:


import gzip
import re
import pandas as pd
import matplotlib.pyplot as plt
from queue import PriorityQueue
import queue as Q
get_ipython().magic('matplotlib inline')
# nathan@petalcard.com.
fnArr = ['transactions1.csv.gz','transactions2.csv.gz','transactions3.csv.gz']
# fnArr = ['trans1a.csv.gz']


# In[95]:


class Transaction:
    def __init__(self, amount,desc,date,typeTrans,misc):
        self.amount = amount
        self.desc = desc
        self.typeTrans = typeTrans
        self.misc = misc
        self.date = date
        self.value = int(date[0:4])*365 + int(date[5:7])*30 + int(date[8:])
    def __str__(self):
        s = "Trans: "
        s += self.amount
        s += self.desc
        s += self.date
        s += self.typeTrans
        s += self.misc
        return s
    def __cmp__(self,other):
        return self.value - other.value
    def __lt__(self, other):
        return self.value - other.value
s = "1000616411022086|1000616412607106|4.49|\|'qmviz\\\|s|2014-02-04|credit|\"rnu	'ycsp"
def parseLine(s):
    result = s.split("|",maxsplit=3)
    user_id = result[0]
    account_id = result[1]
    amount = result[2]
    todo1 = result[3]

    calRe = re.search("\d+-\d+-\d+",todo1)
    cal = todo1[calRe.start():calRe.end()]

    desc = todo1[0:calRe.start()-1]

    tranRe = re.search("debit",todo1)
    if tranRe is None:
        tranRe = re.search("credit",todo1)
        tran = todo1[tranRe.start() : tranRe.end()]
    else:
        tran = todo1[tranRe.start() : tranRe.end()]
    misc = todo1[tranRe.end():]
    
    date = cal
#     print(date)
#     print(date[0:4])
#     print(date[5:7])
    
    value = int(date[0:4])*365 + int(date[5:7])*30 + int(date[8:])
    return (user_id,account_id,amount,desc,cal,tran,misc,value)


# In[99]:


d = {}
# df = pd.DataFrame(columns=["user_id","account_id","amount","desc","date","type","misc"])
def createUpdate(fn,d):
    with gzip.open(fn, 'rt') as f:
        line = f.readline()
        line = f.readline()
        while line:
            temp = parseLine(line)
            t = Transaction(temp[2],temp[3],temp[4],temp[5],temp[6])
            if temp[0] in d:
                d[temp[0]].put((t.value,t))
            else:
                d[temp[0]] = Q.PriorityQueue()
                d[temp[0]].put((t.value,t))
            line = f.readline()
for fn in fnArr:
    createUpdate(fn,d)


# In[100]:


def findStats(q):
    numTran = 0
    totalSum = 0
    minBal = 0
    maxBal = 0
    
    prevValue = 0
    
    while not q.empty():
        t = q.get()[1]  
        numTran += 1
        if t.value != prevValue:
            minBal = min(minBal,totalSum)
            maxBal = max(maxBal,totalSum)
        if t.typeTrans == "credit":
            totalSum += round(float(t.amount),2)
        else:
            totalSum -= round(float(t.amount),2)
        prevValue = t.value
        
    return (numTran,totalSum,minBal,maxBal)


# In[101]:


for k in d:
    r = findStats(d[k])
    print("%s,%d,%.2f,%.2f,%.2f" % (k,r[0],r[1],r[2],r[3]))

