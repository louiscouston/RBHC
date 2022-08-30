import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
#import nfft #https://github.com/jakevdp/nfft
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
from hc_support import *
import os
from pathlib import Path

foldername = './'
print(foldername)
regex = re.compile(r'\d+')

if os.path.exists(foldername+'figures') != 1: os.mkdir(foldername+'figures')

#########################
####### BULK DATA #######
#########################
bulk = np.load(foldername+'bulk.npz')
time = bulk['time']; print(len(time))
x = bulk['x']
z = bulk['z']
u = bulk['u'] 
w = bulk['w']
T = bulk['T']

c = np.sqrt(u**2+w**2)
zv = z[0,:]
dz = np.gradient(zv)
psi = np.cumsum(u*dz.reshape(1,1,len(zv)),axis=-1)

## FIGURES
fig, ax = plt.subplots(figsize=(8,5),nrows=3)
[ax[i].set_aspect(1) for i in range(len(ax))]
[ax[i].set_xlabel('x') for i in range(len(ax))]
[ax[i].set_ylabel('z') for i in range(len(ax))]
ax[0].set_title('c')
ax[1].set_title('T')
ax[2].set_title(r'$\psi$')
j = 0
for i in range(0,len(time),1):
  if i==0:
    imc = ax[0].pcolormesh(x,z,c[i],vmin=0,vmax=c.max()/2,shading='auto')
    cbar = colorbar(imc)
    imT = ax[1].pcolormesh(x,z,T[i],shading='auto')
    cbar = colorbar(imT)
    imPsi = ax[2].pcolormesh(x,z,psi[i],vmin=psi.min()/2,vmax=psi.max()/2,shading='auto')
    cbar = colorbar(imPsi)
    plt.tight_layout(pad=1)
  else:
    imc.set_array(c[i].flatten())
    imT.set_array(T[i].flatten())
    imPsi.set_array(psi[i].flatten())
  fig.suptitle(r't=%1.3f'%time[i],x=0.6)
  fig.savefig(foldername+'figures/snap%i.png'%j)
  j += 1
  
  
  

