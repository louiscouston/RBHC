"""
Usage:
    all-snapshots.py <files> ...
"""
from docopt import docopt
args = docopt(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import re, fnmatch
#import seaborn as sns
from hc_support import *

### NB 16/05/2022
# run as: python3 all-snapshots-may2022.py data/processed-data-restart_* 
# for other aspect ratio, run one set by one set (fixed aspect ratio)
dpi = 150

all_files = sorted(args['<files>'],key=len)
print(all_files)
RaFeall = np.array([6,7,8,9])
alpha1000all = np.array([0,1,10,100,1000,9999])
Gamma = 8

instantaneousT = [[],[],[],[]]
for j in range(len(instantaneousT)): instantaneousT[j] = plt.subplots(figsize=(8,8),ncols=1, nrows=6)

instantaneousc = [[],[],[],[]]
for j in range(len(instantaneousc)): instantaneousc[j] = plt.subplots(figsize=(8,8),ncols=1, nrows=6)

averages = [[],[],[],[]]
for j in range(len(averages)): averages[j] = plt.subplots(figsize=(16,8),ncols=2, nrows=6)

for i in range(len(all_files)):
  regex = re.compile(r'\d+')
  if fnmatch.fnmatch(all_files[i], '*aspect*')==1:
    Gamma, RaFe, alpha1000 = [int(x) for x in regex.findall(all_files[i])]
  else:
    RaFe, alpha1000 = [int(x) for x in regex.findall(all_files[i])]
    Gamma = 8
  #
  try:
    output = np.where(alpha1000all==alpha1000)
    output = alpha1000all.tolist().index(alpha1000)
    print(output)
  except ValueError:
    print("Not in list")
  else:  
    #
    data = np.load(all_files[i])
    RaFi = np.squeeze(np.where(RaFeall==RaFe))
    ali = np.squeeze(np.where(alpha1000all==alpha1000))
    #
    print(all_files[i],RaFi,ali)
    #
    x = data['x']
    z = data['z']
    u = data['ulast']
    w = data['wlast']
    T = data['Tlast']
    
    
    c = np.sqrt(u**2+w**2)
    uave = data['u_t_ave']
    wave = data['w_t_ave']
    Tave = data['T_t_ave']
    cave = data['c_t_ave']
    #
      
    ## FIGURES
    #
    im = instantaneousT[RaFi][1][ali].pcolormesh(x.T,z.T,T.T,shading='gouraud',cmap=cm.Spectral_r)
    instantaneousT[RaFi][1][ali].set_aspect(1)
    colorbar(im,size="3%")
    im = instantaneousc[RaFi][1][ali].pcolormesh(x.T,z.T,c.T,shading='gouraud',cmap=cm.OrRd)
    instantaneousc[RaFi][1][ali].set_aspect(1)
    colorbar(im,size="3%")
    instantaneousc[RaFi][1][ali].quiver(x.T[::4,::4],z.T[::4,::4],u.T[::4,::4], w.T[::4,::4],width=0.0015)
    instantaneousc[RaFi][1][ali].axhline(0.9,linestyle='--',color='tab:blue')
    #
    im = averages[RaFi][1][ali,0].pcolormesh(x.T,z.T,T.T,shading='gouraud',cmap=cm.nipy_spectral)#Spectral_r
    averages[RaFi][1][ali,0].set_aspect(1)
    colorbar(im,size="3%")
    im = averages[RaFi][1][ali,1].pcolormesh(x.T,z.T,c.T,shading='gouraud',cmap=cm.OrRd)
    averages[RaFi][1][ali,1].set_aspect(1)
    colorbar(im,size="3%")
    averages[RaFi][1][ali,1].quiver(x.T[::4,::4],z.T[::4,::4],u.T[::4,::4], w.T[::4,::4],width=0.0015)


    #### REVISION UPDATED 16-05-22
    zv = z[0,:]
    dz = np.gradient(zv)
    psi = np.cumsum(u*dz.reshape(1,len(zv)),axis=-1)
    instantaneousT[RaFi][1][ali].contour(x.T,z.T,psi.T,8,colors='k',linewidths=0.5,alpha=0.5)#,linestyles='-')


###
[[instantaneousT[i][1][j].set_ylabel(r'$z$',fontsize='large',rotation=0,labelpad=8) for i in range(len(instantaneousT))] for j in range(instantaneousT[0][1].shape[0])]
[instantaneousT[i][1][-1].set_xlabel(r'$x$',fontsize='large') for i in range(len(instantaneousT))]
[[instantaneousT[i][1][j].set_xticklabels([]) for i in range(len(instantaneousT))] for j in range(0,instantaneousT[0][1].shape[0]-1)]

[instantaneousT[i][1][0].set_title(r'$T$',fontsize='large',x=1.02) for i in range(len(instantaneousT))]
[instantaneousT[i][0].tight_layout(w_pad=0.1,h_pad=-6) for i in range(len(instantaneousT))]
[instantaneousT[i][0].savefig('instantaneousT_%i_%i.png'%(RaFeall[i],Gamma),dpi=dpi,bbox_inches='tight',pad_inches=0.02) for i in range(len(instantaneousT))]
if Gamma==8: instantaneousT[2][0].savefig('Figure3.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
plt.close()


[[instantaneousc[i][1][j].set_ylabel(r'$z$',fontsize='large',rotation=0,labelpad=8) for i in range(len(instantaneousc))] for j in range(instantaneousc[0][1].shape[0])]
[instantaneousc[i][1][-1].set_xlabel(r'$x$',fontsize='large') for i in range(len(instantaneousc))]
[[instantaneousc[i][1][j].set_xticklabels([]) for i in range(len(instantaneousc))] for j in range(0,instantaneousT[0][1].shape[0]-1)]
[instantaneousc[i][1][0].set_title(r'$\sqrt{u^2+w^2}$',fontsize='large',x=1.02) for i in range(len(instantaneousc))]
[instantaneousc[i][0].tight_layout(w_pad=0.1,h_pad=-6) for i in range(len(instantaneousc))]
[instantaneousc[i][0].savefig('instantaneousc_%i_%i.png'%(RaFeall[i],Gamma),dpi=dpi,bbox_inches='tight',pad_inches=0.02) for i in range(len(instantaneousc))]
if Gamma==8: instantaneousc[2][0].savefig('Figure2.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02) 
plt.close()

###
[[averages[i][1][j,0].set_ylabel(r'$z$',fontsize='large',rotation=0) for i in range(len(averages))] for j in range(averages[0][1].shape[0])]
[[averages[i][1][j,1].set_yticklabels([]) for i in range(len(averages))] for j in range(averages[0][1].shape[0])]
[[averages[i][1][j,1].set_xticklabels([]) for i in range(len(averages))] for j in range(1,averages[0][1].shape[0])]
[[averages[i][1][-1,j].set_xlabel(r'$x$',fontsize='large') for i in range(len(averages))] for j in range(2)]
[averages[i][1][0,0].set_title(r'$T$',fontsize='large',pad=-20) for i in range(len(averages))]
[averages[i][1][0,1].set_title(r'$\sqrt{u^2+w^2}$',fontsize='large',pad=-20) for i in range(len(averages))]
[averages[i][0].tight_layout(pad=1.2) for i in range(len(averages))]
[averages[i][0].savefig('averages_%i_%i.png'%(RaFeall[i],Gamma),dpi=dpi,pad=0) for i in range(len(averages))]
plt.close()

