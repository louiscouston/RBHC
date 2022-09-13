import numpy as np
from hc_support import *
import matplotlib.pyplot as plt

t = np.linspace(0,20*np.pi,200)
f = np.tile(np.sin(t).reshape(1,len(t),1),(3,1,3))
print(f.shape)

ffull = autocor(f,t,mode='full')
fsame = autocor(f,t,mode='same')
#fvalid = autocor(f,t,mode='valid')
fmy, denom, lagnumber = myautocor(f,t)

print(fsame[0,:,0])
print(fsame.min(),fsame.max())

fig, ax = plt.subplots(figsize=(12,4),ncols=4)
ax[0].plot(t,ffull[0,:,0],'-k')
ax[0].plot(t,fmy[0,:,0],'--r')
ax[0].plot(t,fsame[0,:,0],':b')
ax[1].plot(t,lagnumber)
ax[2].plot(t,denom[0,:,0])
ax[3].plot(t,fmy[0,:,0]/denom[0,:,0],'--r')
plt.tight_layout()
fig.savefig('check_autocor.png',dpi=300)
