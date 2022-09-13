"""
Usage:
    all-fit.py
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from hc_support import *

data = np.load('data_for_fit.npz')
RaF = data['RaF'] 	# increases along 0th dim
alpha = data['alpha'] 	# increases along 1st dim
RaL = data['RaL']
NuRB = data['NuRB']
#NuHC = data['NuHC']
NuHC = data['NuHCsurf']
Re = data['Re']
ReKE = data['Remean']
Rerms = data['Resummed']
ReUKE = data['ReUmean']
ReUrms = data['ReUsummed']
ReWKE = data['ReWmean']
ReWrms = data['ReWsummed']



#
def powerlaw_fit(x, *p):
  pref, expo = p
  return np.log(pref)+expo*x

print(RaF[:,0])
print(alpha[0,:])

print(RaL[:-1,-1])
print(alpha[:-1,-1])

Refit = ReKE

fig, ax = plt.subplots(figsize=(8,3),nrows=2,ncols=3)
#
im = ax[0,0].pcolormesh(RaF[:,:-1],alpha[:,:-1],ReKE[:,:-1],shading='gouraud',norm=colors.LogNorm(vmin=ReKE.max()/1e3, vmax=ReKE.max()), cmap=cm.RdYlBu_r)
colorbar(im); ax[0,0].set_title('ReKE')
#
im = ax[0,1].pcolormesh(RaF[:,:-1],alpha[:,:-1],Rerms[:,:-1]/ReKE[:,:-1],shading='gouraud')
colorbar(im); ax[0,1].set_title('Rerms/ReKE')
#
im = ax[0,2].pcolormesh(RaF[:,:-1],alpha[:,:-1],Re[:,:-1]/ReKE[:,:-1],shading='gouraud')
colorbar(im); ax[0,2].set_title('Re/ReKE')
#
im = ax[1,0].pcolormesh(RaF[:,:-1],alpha[:,:-1],ReUKE[:,:-1]/ReWKE[:,:-1],shading='gouraud')
colorbar(im); ax[1,0].set_title('ReUKE/ReWKE')
#
im = ax[1,1].pcolormesh(RaF[:,:-1],alpha[:,:-1],ReUrms[:,:-1]/ReWrms[:,:-1],shading='gouraud')
colorbar(im); ax[1,1].set_title('ReUrms/ReWrms')
#
[[ax[i,j].set_xlim((10**6,10**9)) for i in range(2)] for j in range(3)]
[[ax[i,j].set_xlabel(r'$Ra_F$') for i in range(2)] for j in range(3)]
[[ax[i,j].set_ylabel(r'$\alpha$') for i in range(2)] for j in range(3)]
[[ax[i,j].set_xscale('log') for i in range(2)] for j in range(3)]
[[ax[i,j].set_yscale('symlog',linthresh=1e-3,linscale=1) for i in range(2)] for j in range(3)]
#
fig.tight_layout(pad=0.05)
fig.savefig('reynolds.png',dpi=300)
plt.close()


### Nusselt scalings
# [:,0] -> pure RB results
NuRBcoeffs, var_matrix = curve_fit(powerlaw_fit, np.log(RaF[:,0]), np.log(NuRB[:,0]), p0=[0.1, 2/7], maxfev=10000)
aRB = NuRBcoeffs[0]; bRB = NuRBcoeffs[1]
NuRBrelabserror = np.max(np.abs(NuRB[:,0]-aRB*RaF[:,0]**bRB)/(aRB*RaF[:,0]**bRB))
# [:-1,-1] -> pure HC results [last RaL truncated because of missing simulation results]
NuHCcoeffs, var_matrix = curve_fit(powerlaw_fit, np.log(RaL[:-1,-1]), np.log(NuHC[:-1,-1]), p0=[1, 1/5], maxfev=10000)
aHC = NuHCcoeffs[0]; bHC = NuHCcoeffs[1]
NuHCrelabserror = np.max(np.abs(NuHC[:-1,-1]-aHC*RaL[:-1,-1]**bHC)/(aHC*RaL[:-1,-1]**bHC))

### Reynolds scalings
# [:,0] -> pure RB results
ReRBcoeffs, var_matrix = curve_fit(powerlaw_fit, np.log(RaF[:,0]), np.log(Refit[:,0]), p0=[0.01, 0.5], maxfev=10000)
cRB = ReRBcoeffs[0]; dRB = ReRBcoeffs[1]
ReRBrelabserror = np.max(np.abs(Refit[:,0]-cRB*RaF[:,0]**dRB)/(cRB*RaF[:,0]**dRB))
# [:-1,-1] -> pure HC results [last RaL truncated because of missing simulation results]
ReHCcoeffs, var_matrix = curve_fit(powerlaw_fit, np.log(RaL[:-1,-1]), np.log(Refit[:-1,-1]), p0=[1, 1/5], maxfev=10000)
cHC = ReHCcoeffs[0]; dHC = ReHCcoeffs[1]
ReHCrelabserror = np.max(np.abs(Refit[:-1,-1]-cHC*RaL[:-1,-1]**dHC)/(cHC*RaL[:-1,-1]**dHC))

### Figure
fig, ax = plt.subplots(figsize=(8,3),nrows=1,ncols=4)
#
ax[0].loglog(RaF[:,0],aRB*RaF[:,0]**bRB,'-k',label=r'%1.3e $Ra_F^{%1.3f}$'%(aRB,bRB))
ax[0].loglog(RaF[:,0],NuRB[:,0],'or')
ax[0].set_xlabel(r'$Ra_F$')
ax[0].set_ylabel(r'$Nu_{RB}$')
ax[0].legend()
#
ax[1].loglog(RaL[:-1,-1],aHC*RaL[:-1,-1]**bHC,'-k',label=r'%1.3e $Ra_L^{%1.3f}$'%(aHC,bHC))
ax[1].loglog(RaL[:-1,-1],NuHC[:-1,-1],'ob')
ax[1].set_xlabel(r'$Ra_L$')
ax[1].set_ylabel(r'$Nu_{HC}$')
ax[1].legend()
#
ax[2].loglog(RaF[:,0],cRB*RaF[:,0]**dRB,'-k',label=r'%1.3e $Ra_F^{%1.3f}$'%(cRB,dRB))
ax[2].loglog(RaF[:,0],Refit[:,0],'or',label=r'ReKE')
ax[2].loglog(RaF[:,0],Rerms[:,0],'db',label=r'Rerms')
ax[2].loglog(RaF[:,0],Rerms[:,0],'sg',label=r'Rec')
ax[2].set_xlabel(r'$Ra_F$')
ax[2].set_ylabel(r'$Re$')
ax[2].legend()
#
ax[3].loglog(RaL[:-1,-1],cHC*RaL[:-1,-1]**dHC,'-k',label=r'%1.3e $Ra_L^{%1.3f}$'%(cHC,dHC))
ax[3].loglog(RaL[:-1,-1],Refit[:-1,-1],'ob')
ax[3].loglog(RaL[:-1,-1],Refit[:-1,-1],'or',label=r'ReKE')
ax[3].loglog(RaL[:-1,-1],Rerms[:-1,-1],'db',label=r'Rerms')
ax[3].loglog(RaL[:-1,-1],Rerms[:-1,-1],'sg',label=r'Rec')
ax[3].set_xlabel(r'$Ra_L$')
ax[3].set_ylabel(r'$Re$')
ax[3].legend()
#
fig.tight_layout(pad=0.05)
fig.savefig('fit.png',dpi=300)
plt.close()


