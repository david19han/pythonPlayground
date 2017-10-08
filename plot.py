#!/usr/bin/env python
 
"""
Generate a bandlimited function in Python.
"""
 
import numpy as np
 
dx = 0.05
x_max = 6.0
x = np.arange(0.0, x_max, dx)
 
N = 4
a_max = 5
b_max = 7
a = np.random.randint(0, a_max, N)
b = np.random.randint(0, b_max, N)
 
y = np.zeros_like(x)
for i in xrange(0, N):
    y += a[i]*np.cos((i+1)*x)+b[i]*np.sin((i+1)*x)
 
# Optional: if you want to plot the function, set MAKE_PLOT to
# True:
MAKE_PLOT = True
if MAKE_PLOT:
    import matplotlib as mpl
    mpl.use('agg')
    import matplotlib.pyplot as plt
 
    plt.gcf()
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_xlim((min(x), max(x)))
    plt.savefig('plot.png')
 
 
