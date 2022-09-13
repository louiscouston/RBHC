
######################
# Note from 10/05/2022
# python3 freezing.py
# used to obtain the linear approximation for Tf(p)
######################

import numpy as np
from gsw import freezing as freezing
import matplotlib.pyplot as plt

p = np.linspace(0,5000,1000) # in dbar
sa = 0*p
air = 0
T = freezing.t_freezing(sa,p,air)

g = 9.81
rhoi = 917
h = p*1e4/rhoi/g
print(h.min(),h.max())

polcoef = np.polyfit(p,T,1) # linear fit
print(polcoef)
approx1 = -0.00082057*p + 0.06671865# np.poly1d(polcoef)(p)
approx2 = 4.7184e-3-7.4584e-4*p-1.4999e-8*p**2

fig, ax = plt.subplots()
ax.plot(T,p,'-k',linewidth=1.5,label=r'exact')
ax.plot(approx1,p,'--r',linewidth=1.5,label=r'linear')
ax.plot(approx2,p,'--b',linewidth=1.5,label=r'quad')
ax.invert_yaxis()
ax.legend()
fig.savefig('freezing.png',dpi=300)

alpha = np.array([0.003,0.03])

