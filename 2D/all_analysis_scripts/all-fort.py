"""
Usage:
    all-fort.py <files> ...
"""
######################
# Note from 10/06/2022
# python3 all-fort-may2022.py data/processed-fort-*
# we want them all i.e. transient, restart and aspect
######################

from docopt import docopt
args = docopt(__doc__)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import re, fnmatch
from hc_support import *

all_files = sorted(args['<files>'],key=len)
print(all_files)
RaFeall = np.array([6,7,8,9])
alpha1000all = np.array([0,1,10,100,1000,9999])
alpha1000allmaxi = np.array([0,1,3,10,30,100,300,1000,9999])

fig, ax = plt.subplots(figsize=(9,4),ncols=2)
[ax[i].axvline(x=1,linestyle='--',color='k',linewidth=0.5) for i in range(2)]

figpub1, axpub1 = plt.subplots(figsize=(5,2.3),ncols=2)
[axpub1[i].axvline(x=1,linestyle='--',color='k',linewidth=0.5) for i in range(2)]
axpub1[0].axhline(y=0,linestyle='--',color='k',linewidth=0.5)


fig4rev, ax4rev = plt.subplots(figsize=(6,3),ncols=2,nrows=2)
[[ax4rev[i,j].axvline(x=1,linestyle='--',color='k',linewidth=0.5) for i in range(2)] for j in range(2)]
[ax4rev[0,j].axhline(y=0,linestyle='--',color='k',linewidth=0.5) for j in range(2)]

Reall = []
figre, axre = plt.subplots(figsize=(6,4),ncols=1)

for i in range(len(all_files)):
  data = np.load(all_files[i])
  regex = re.compile(r'\d+')
  
  if fnmatch.fnmatch(all_files[i], '*aspect*')==1:
    Gamma, RaFe, alpha1000 = [int(x) for x in regex.findall(all_files[i])]
  else:
    RaFe, alpha1000 = [int(x) for x in regex.findall(all_files[i])]
    Gamma = 8
  width = Gamma/8 # linewidth

  RaFi = np.squeeze(np.where(RaFeall==RaFe))
  ali = np.squeeze(np.where(alpha1000allmaxi==alpha1000))
  print(Gamma,RaFi,ali,all_files[i])
  time = data['time']
  T = data['T']
  KE = data['KE']
  
  if time[-1]>1.3:
    time = time-0.2
  
  Reall.append(np.sqrt(2*KE))
  
  if alpha1000!=30 and alpha1000!=300:  
  
    ## FIGURES
    if fnmatch.fnmatch(all_files[i], '*restart*')==0: 
      label = r'Rae=%s, $\alpha$=%1.3f'%(RaFe,alpha1000/1000)
      ax[0].plot(time,T,'-',color=my_colors[RaFi][ali])
      ax[1].plot(time,KE,'-',label=label,color=my_colors[RaFi][ali])
    else: 
      ax[0].plot(time,T,'-',color=my_colors[RaFi][ali])
      ax[1].plot(time,KE,'-',color=my_colors[RaFi][ali])
  
    if fnmatch.fnmatch(all_files[i], '*restart*') or fnmatch.fnmatch(all_files[i], '*aspect*'):   
      tsss = 1.0
    else: 
      tsss = 0.6 # (assumed) steady state start time
    itsss = np.argmin(np.abs(time-tsss)); print('tsss is %1.2f'%tsss,'isss is %i'%itsss)

    axre.loglog(10**RaFe,t_average(np.sqrt(2*KE[itsss:]),time[itsss:]),'o')
    print(10**RaFe,t_average(np.sqrt(2*KE[itsss:]),time[itsss:]))
  
  ############## FIGURE FOR PUBLICATION  
    axpub1[0].plot(time,T,'-',color=my_colors[RaFi][ali],linewidth=width)
    axpub1[1].plot(time,np.sqrt(2*KE),'-',color=my_colors[RaFi][ali],linewidth=width)
    # -- updated 10/05/2022
    if T[-1]>0:
      ax4rev[0,0].plot(time,T,'-',color=my_colors[RaFi][ali],linewidth=width)
    else: 
      ax4rev[0,1].plot(time,T,'-',color=my_colors[RaFi][ali],linewidth=width)
    if RaFe<8:
      ax4rev[1,0].plot(time,np.sqrt(2*KE),'-',color=my_colors[RaFi][ali],linewidth=width)
    else:
      ax4rev[1,1].plot(time,np.sqrt(2*KE),'-',color=my_colors[RaFi][ali],linewidth=width)
  ############## FIGURE FOR PUBLICATION  
  
  else:
    print('NOT DRAWN',all_files[i])

axre.set_xlabel('Ra_F')
axre.set_ylabel('Re')
plt.tight_layout(pad=0.05)
figre.savefig('figre.png',dpi=300)
plt.close()

############## FIGURE FOR PUBLICATION 
axpub1[0].set_xlabel(r'$t$',fontsize='large')
axpub1[0].set_ylabel(r'$\langle T \rangle$',fontsize='large')
axpub1[1].set_xlabel(r'$t$',fontsize='large')
axpub1[1].set_ylabel(r'$\widehat{Re}$',fontsize='large')
axpub1[0].set_yscale('symlog',linthresh=1e-1,linscale=1)
axpub1[1].set_yscale('log')
axpub1[1].set_ylim((6e1,6e3))
plt.tight_layout(pad=0.05)
figpub1.savefig('publication1.png',dpi=300)
figpub1.savefig('final4.png',dpi=300,bbox_inches='tight',pad_inches=0.02)
plt.close()
# -- updated 10/05/2022
[ax4rev[1,j].set_xlabel(r'$t$',fontsize='large') for j in range(2)]
[ax4rev[0,j].set_xticklabels([]) for j in range(2)]
ax4rev[0,0].set_ylabel(r'$\langle T \rangle$',fontsize='large')
ax4rev[1,0].set_ylabel(r'$\widehat{Re}$',fontsize='large')
[ax4rev[0,j].set_yscale('symlog',linthresh=1e-3,linscale=0.1) for j in range(2)]
[ax4rev[1,j].set_yscale('log') for j in range(2)]
ax4rev[0,0].set_ylim((1e-2,1e-1))
#
ax4rev[0,0].set_title(r'$\langle T \rangle>0$',fontsize='medium')
ax4rev[0,1].set_title(r'$\langle T \rangle<0$',fontsize='medium')
ax4rev[1,0].set_title(r'$Ra_F\leq 10^7$',fontsize='medium')
ax4rev[1,1].set_title(r'$Ra_F \geq 10^8$',fontsize='medium')
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9), numticks=12)
ax4rev[0,0].yaxis.set_minor_locator(locmin)
ax4rev[0,0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
#
ax4rev[0,1].set_ylim((-10,-1e-2))
ax4rev[1,0].set_ylim((6e1,1e3))
ax4rev[1,1].set_ylim((6e2,3e3))
#ax4rev[1,1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
#[[ax4rev[i,j].tick_params(axis='both', labelsize='large') for i in range(2)] for j in range(2)] #, which='both', width=2, length=6
#plt.tight_layout(pad=2)# , w_pad=0.2
plt.tight_layout(pad=2)
for i in range(2):
  pos1 = ax4rev[i,1].get_position() # get the original position 
  pos2 = [pos1.x0 + 0.05, pos1.y0,  pos1.width, pos1.height] 
  ax4rev[i,1].set_position(pos2) # set a new position
for i in range(2):
  pos1 = ax4rev[0,i].get_position() # get the original position 
  pos2 = [pos1.x0, pos1.y0 + 0.03,  pos1.width, pos1.height] 
  ax4rev[0,i].set_position(pos2) # set a new position
fig4rev.savefig('Figure4revision.png',dpi=300,bbox_inches='tight',pad_inches=0.05) #0.02 ,bbox_inches='tight',pad_inches=0.05
fig4rev.savefig('Figure4revision.eps',bbox_inches='tight',pad_inches=0.05)
plt.close()
############## FIGURE FOR PUBLICATION 


ax[0].set_xlabel(r'$t$',fontsize='large')
ax[0].set_ylabel(r'$\langle T \rangle$',fontsize='large')
ax[1].set_xlabel(r'$t$',fontsize='large')
ax[1].set_ylabel(r'$\mathcal{KE}$',fontsize='large')
ax[0].set_yscale('symlog',linthresh=1e-1,linscale=1)
ax[1].set_yscale('log')
ax[1].set_ylim((1e3,1e7))
ax[1].legend(bbox_to_anchor=(1.05, 0.5),loc='center left',fontsize='small',ncol=2)
plt.tight_layout(pad=1.2)
fig.savefig('all_time_fort.png')
plt.close()




