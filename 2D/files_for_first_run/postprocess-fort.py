import numpy as np
import matplotlib.pyplot as plt
import os, os.path, fnmatch, h5py, re

data = np.loadtxt('fort.51')
time = data[:,0]
T = data[:,1]
KE = data[:,3]
  
np.savez('fort.npz', time=time, T=T, KE=KE)  

## FIGURES
fig, ax = plt.subplots(figsize=(9,2.7),ncols=2)

ax[0].plot(time,T,'-b')
ax[1].plot(time,KE,'-r')

ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$\langle T \rangle$')
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$\mathcal{KE}$')
ax[1].set_yscale('symlog',linthresh=1)
#ax[1].legend(bbox_to_anchor=(1.05, 0.5),loc='center left',fontsize='small',ncol=2)
plt.tight_layout(pad=1.2)
fig.savefig('fort.png')
plt.close()
