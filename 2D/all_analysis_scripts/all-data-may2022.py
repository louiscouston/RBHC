"""
Usage:
    all-data-may2022.py <files> ...
"""
######################
# Note from 10/06/2022
# python3 all-data-may2022.py data/processed-data-restart* data/processed-data-aspect*
# we want only restart and aspect (no transient)
# includes publication figures 1b, 5, 6, 7, 8, 9, 10, 11
# NB: here alpha is Lambda ...
dpi = 150
######################

from docopt import docopt
args = docopt(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import re, fnmatch
#import seaborn as sns
from hc_support import *

all_files = sorted(args['<files>'],key=len)
print(all_files)
RaFeall = np.array([6,7,8,9])
alpha1000all = np.array([0,1,10,100,1000,9999])

############## FIGURE FOR PUBLICATION  
figpub1, axpub1 = plt.subplots(figsize=(3,2.3),nrows=1,ncols=1)
#
figfin4, axfin4 = plt.subplots(figsize=(6,4.6),nrows=2,ncols=2)
axfin4[1,1].axhline(y=1,linestyle='--',color='k')
#
figpat6master = [[],[],[],[]]
for i in range(len(RaFeall)):
    figpat6master[i] = plt.subplots(figsize=(5*5/4,3.6),nrows=2,ncols=1)
#
figpat6bmaster = [[],[],[],[]]
for i in range(len(RaFeall)):
    figpat6bmaster[i] = plt.subplots(figsize=(5*5/4,1.8),nrows=1,ncols=1)
#
figfin6, axfin6 = plt.subplots(figsize=(6,4.6),nrows=2,ncols=2)
axfin6[0,1].axhline(y=1,linestyle='--',color='k')
axfin6[1,1].axhline(y=1,linestyle='--',color='k')
axfin6[1,0].plot([3e6,3e8],[1e5,1e1],linestyle=':',color='k')
#
figfin6b, axfin6b = plt.subplots(figsize=(9,2.3),nrows=1,ncols=3)
#
figfin7, axfin7 = plt.subplots(figsize=(3.3,2.3),nrows=1,ncols=1)
figfin7b, axfin7b = plt.subplots(figsize=(6,2.3),nrows=1,ncols=2)
#
figpub8master = [[],[],[],[]]
for i in range(len(RaFeall)):
    figpub8master[i] = plt.subplots(figsize=(7,4.6),nrows=2,ncols=3)
#
fig7a, ax7a = plt.subplots(figsize=(6,2.3),nrows=1,ncols=2)
ax7a[1].axhline(y=1,linestyle='--',color='k')
fig7b, ax7b = plt.subplots(figsize=(9*4/5,3*4/5),nrows=1,ncols=3)
ax7b[1].plot([1e-1,1e0],[3e-3,3e-1],linestyle='-',color='k')
ax7b[2].axhline(y=1,linestyle='--',color='k')
#
figTb, axTb = plt.subplots(figsize=(10,10),nrows=3,ncols=4)
fighalf, axhalf = plt.subplots(figsize=(16,10),nrows=4,ncols=5)
#
figappE, axappE = plt.subplots(figsize=(9*4/5,4.6*4/5),nrows=2,ncols=3)
#
figfin9, axfin9 = plt.subplots(figsize=(3,2.3),nrows=1,ncols=1)
axfin9.axhline(y=4,linestyle='--',color='k')
axfin9.axhline(y=8,linestyle='--',color='k')
axfin9.axhline(y=16,linestyle='--',color='k')
axfin9.axhline(y=1,linestyle=':',color='k')
#
figpsi, axpsi = plt.subplots(figsize=(3,2.3),nrows=1,ncols=1) 
############## FIGURE FOR PUBLICATION  
  
linestyles = ['solid','dashed','dotted','dashdot']
  
aRB=3.938e-1
bRB=0.191
cRB=2.326e-1
dRB=0.441
aHC=3.181e-1 #1.419e-1
bHC=0.226 #0.219
cHC=1.839e-2
dHC=0.396

NuRBall = np.zeros((4,6))
NuHCall = np.zeros((4,6))
NuHCsurfall = np.zeros((4,6))
Reall = np.zeros((4,6))
Remeanall = np.zeros((4,6))
Resummedall = np.zeros((4,6))
ReUmeanall = np.zeros((4,6))
ReUsummedall = np.zeros((4,6))
ReWmeanall = np.zeros((4,6))
ReWsummedall = np.zeros((4,6))
Tmeanall = np.zeros((4,6))
RaLall = np.zeros((4,6))
alphaall = np.zeros((4,6))
RaFall = np.zeros((4,6))
Rediffall = np.zeros((7,4,6))

calesc = []

for i in range(len(all_files)):
  print('LOC FILE IS ',all_files[i])
  data = np.load(all_files[i],'r')
  regex = re.compile(r'\d+')
  #
  if fnmatch.fnmatch(all_files[i], '*aspect*')==1:
    Gamma, RaFe, alpha1000 = [int(x) for x in regex.findall(all_files[i])]
    alpha1000all = np.array([0,1,3,10,30,100,300,1000,9999])
  else:
    RaFe, alpha1000 = [int(x) for x in regex.findall(all_files[i])]
    Gamma = 8
    alpha1000all = np.array([0,1,10,100,1000,9999])
  #
  if Gamma==4: symbsize=4 
  elif Gamma==8: symbsize=6 #6.5  
  elif Gamma==16: symbsize=6.5
  #
  alpha1000allmaxi = np.array([0,1,3,10,30,100,300,1000,9999])
  #
  RaFi = np.squeeze(np.where(RaFeall==RaFe))
  ali = np.squeeze(np.where(alpha1000allmaxi==alpha1000))
  alismall = np.squeeze(np.where(alpha1000all==alpha1000))
  print(RaFi,ali)
  if fnmatch.fnmatch(all_files[i], '*aspect*')==1: 
    if alpha1000==9999:
      print('IMPOSSIBLE!',a) 
    elif fnmatch.fnmatch(all_files[i], '*_4_*')==1: symb = 'd'
    else: symb = 's'
  elif fnmatch.fnmatch(all_files[i], '*restart*')==0: 
    if alpha1000==9999: alpha1000=1001; symb = '+'
    else: symb = 'x'
  else: 
    if alpha1000==9999: alpha1000=1001; symb = '*' # 1001 to distinguish pure HC from RBH at 1000
    else:  symb = 'o'
  time = data['time']
  if time[-1]>1.3:
    time = time-0.2
  if fnmatch.fnmatch(all_files[i], '*restart*') or fnmatch.fnmatch(all_files[i], '*aspect*'): tsss = 1.0
  else: tsss = 0.6 # (assumed) steady state start time
  itsss = np.argmin(np.abs(time-tsss)); print('tsss is %1.2f'%tsss,'isss is %i'%itsss)
  if itsss==len(time)-1: itsss=len(time)-2
  Ret = data['Re'][itsss:]
  utophalft = data['utophalf']
  Re = t_average(Ret,time[itsss:])
  Tb = data['Tb']
  Tt = data['Tt']
  dzTt = data['dzTt']
  dzTb = data['dzTb']
  topflux = -t_average(dzTt[itsss:],time[itsss:])
  dTdz_t_ave = data['dTdz_t_ave']
  dTdx_t_ave = data['dTdx_t_ave']
  psi_max = t_average(data['psi_max'][itsss:],time[itsss:])
  pos_max = t_average(data['pos_max'][itsss:],time[itsss:])
  psi_t_ave = data['psi_t_ave']
  psi_t_ave_max = data['psi_t_ave_max']
  pos_t_ave_max = data['pos_t_ave_max']
  u_x_ave = data['u_x_ave']
  w_z_ave = data['w_z_ave']
  xv = data['x'][:,0]
  zv = data['z'][0,:]
  xb = data['xb']
  xt = data['xt']
  zl = data['zl']
  zr = data['zr']
  Tl = data['Tl']
  Tr = data['Tr']
  u2summed = np.sum(data['u2summed'][itsss:])/np.sum(time[itsss:])
  w2summed = np.sum(data['w2summed'][itsss:])/np.sum(time[itsss:])
  u2mean = t_average(data['u2mean'][itsss:],time[itsss:])
  w2mean = t_average(data['w2mean'][itsss:],time[itsss:])
  ReUsummed = np.sqrt(u2summed)
  ReWsummed = np.sqrt(w2summed)
  Resummed = np.sqrt(u2summed+w2summed)
  ReUmean = np.sqrt(u2mean)
  ReWmean = np.sqrt(w2mean)
  Remean = np.sqrt(u2mean+w2mean)
  Remeant = np.sqrt(data['u2mean'][itsss:]+data['w2mean'][itsss:])
  print(ReUsummed,ReWsummed,ReUmean,ReWmean,Resummed,Remean,Re)
#
  strat = (tz_average(Tr,time,zv)-tz_average(Tl,time,zv))
  print(alpha1000,(Tt.max()-Tt.min()),strat)  
#
  alpha = alpha1000/1000
  gradT = xz_average(np.sqrt(dTdz_t_ave**2+dTdx_t_ave**2),xv,zv)
  Qu = tz_average(np.abs(u_x_ave)[itsss:],time[itsss:],zv)
  Qw = tz_average(np.abs(w_z_ave)[itsss:],time[itsss:],xv)
  T_xz_ave = data['T_xz_ave']
  print('SIZE ',T_xz_ave.shape,np.std(T_xz_ave))
  T_xzt_ave = t_average(T_xz_ave[itsss:],time[itsss:])
  utophalf = np.abs(t_average(data['utophalf'][itsss:],time[itsss:]))
  u09t = x_average(data['u09'][itsss:],xv)
  u09 = t_average(u09t,time[itsss:])
  Tb_x_ave = x_average(Tb,xb)
  Tb_t_ave = t_average(Tb[itsss:],time[itsss:])
  Tt_t_ave = t_average(Tt[itsss:],time[itsss:])
  T_min = -alpha1000/1000*Gamma/2
  #
  ###  Nut = x_average(np.abs(dzTt),xt)[itsss:]/alpha ### BEFORE PROPER NORMALIZATION
  AbsFluxt = x_average(np.abs(dzTt),xt)[itsss:]
  AbsFlux = t_average(AbsFluxt,time[itsss:])
  ### Nu_norm = -1+alpha*np.tanh(np.pi/Gamma) ### only valid for large alpha and small Gamma
  #Nu_norm = alpha*np.tanh(np.pi/Gamma) ### assumes no geothermal flux
  dzTdiff_z1 = alpha*np.pi/2*np.tanh(np.pi/Gamma)*np.sin(np.pi*xv/Gamma)-1*(alpha<=1) # no geothermal heating for pure HC
  Nu_norm_noGH = x_average(np.abs(alpha*np.pi/2*np.tanh(np.pi/Gamma)*np.sin(np.pi*xv/Gamma)),xv)
  Nu_norm = x_average(np.abs(dzTdiff_z1),xv)
  Nut = AbsFluxt/Nu_norm
  Nu = AbsFlux/Nu_norm
  TopCorrelationt = x_average(dzTt[itsss:]*Tt[itsss:]-dzTb[itsss:]*Tb[itsss:],xv) # eq3.8 Rocha2020
  TopCorrelation = t_average(TopCorrelationt,time[itsss:])
  
  ##### ATTENTION #### MISTAKE IN ORIGINAL SUBMISSION ### +1 due to geothermal heating added -- update 12/05/2022
  Nusurf_norm = np.pi*alpha**2*Gamma*np.tanh(np.pi/Gamma)/8 + 1*(alpha<=1)
  Nusurft = TopCorrelationt/Nusurf_norm
  Nusurf = TopCorrelation/Nusurf_norm
  print('DIFFERENCE IS ',alpha/Nu_norm)
  
  
  Fabst = x_average(np.abs(dzTt),xt)[itsss:]
  Fabs = t_average(Nut,time[itsss:])
  x0 = np.argmin(np.abs(xv))
  print(xv.shape,xt.shape,xb.shape)
  Ftophalft = x_average(-dzTt[itsss:,:x0],xv[:x0])
  Ftophalf = t_average(Ftophalft,time[itsss:])
  Nucold_norm = 1*(alpha<=1)+alpha*np.tanh(np.pi/Gamma)
  Nucold = Ftophalf/Nucold_norm
  
  #gradT2 = data['gradT2']
  #NugradT2 = t_average(gradT2[itsss:],time[itsss:]) # eq3.7 Rocha2020
  #NugradT2_norm = 
  #NugradT2 = NugradT2/NugradT2_norm
  #
  RaL = 10**RaFe*alpha1000/1000*Gamma**4
  
  NuRBt = Tb_x_ave[itsss:]-T_min
  
  #Tdiff_z1 = alpha*Gamma/2*np.sin(np.pi*xb/Gamma)
  T_z0_z1 = x_average(Tb-Tt,xv)[itsss:] # the second term gives 0 actually
  print(alpha,'MEAN Tt TEMPERATURE IS : ',t_average(x_average(Tt,xv)[itsss:],time[itsss:])) 
  print(alpha,'MEAN Tb-Tt TEMPERATURE IS : ',t_average(T_z0_z1,time[itsss:])) 
  
  
  NuaveRBt = 1/T_z0_z1
  NuaveRB = 1/t_average(T_z0_z1,time[itsss:]) 
  
  NuRBstd = 1/(Tb_x_ave[itsss:]-T_min)
  NuRB = 1/t_average(NuRBt,time[itsss:]) # maybe we should compute the exact diffusive flux from a bottom uniform T boundary to a sinusoidally-varying T on top boundary? 
  psi_tx_ave = x_average(psi_t_ave,xv)
  psi_tx_ave_max = psi_tx_ave[np.argmax(np.abs(psi_tx_ave))]
  
  autocor_u09 = data['autocor_u09'][itsss:]
  autocor_u09 = autocor_u09/(autocor_u09[:,0].reshape(autocor_u09.shape[0],1))
  autocor_u09_t_ave = t_average(autocor_u09,time[itsss:])
  
  autocor_u09_my = data['autocor_u09_my'][itsss:]
  denom09_my = data['denom09_my'][itsss:]
  lagnumber_my = data['lagnumber_my']
  autocor_u09_my_adjusted = autocor_u09_my/denom09_my
  autocor_u09_my = autocor_u09_my/(autocor_u09_my[:,0].reshape(autocor_u09_my.shape[0],1))
  autocor_u09_my_t_ave = t_average(autocor_u09_my,time[itsss:])
  autocor_u09_my_adjusted_t_ave = t_average(autocor_u09_my_adjusted,time[itsss:])

  iRuu09min = np.argmin(autocor_u09_t_ave)
  xRuu09min = xv[iRuu09min]-xv[0]
  if iRuu09min==len(autocor_u09_t_ave)-1:
    xRuu09max = Gamma
  else:
    iRuu09max = np.argmax(autocor_u09_t_ave[iRuu09min:])
    xRuu09max = xv[iRuu09min+iRuu09max]-xv[0]

  calesc.append([all_files[i], xRuu09min, xRuu09max])

  autocor_u09_tx_ave = x_average(autocor_u09_t_ave,xv)
  ilength_u09_t_ave = t_average(data['ilength_u09'],time[itsss:])
  print(autocor_u09_tx_ave,ilength_u09_t_ave,data['ilength_u09'].shape)
  
  if fnmatch.fnmatch(all_files[i], '*restart*')==0 and fnmatch.fnmatch(all_files[i], '*aspect*')==0:
    autocor_u09_t_ave=np.nan
    xRuu09max=np.nan 
    ilength_u09_t_ave=np.nan
  else:
    print('BON!')
  
  if fnmatch.fnmatch(all_files[i], '*restart*')==1:
    NuRBall[RaFi,alismall] = NuRB
    NuHCall[RaFi,alismall] = Nu
    NuHCsurfall[RaFi,alismall] = Nusurf
    RaLall[RaFi,alismall] = RaL
    Remeanall[RaFi,alismall] = Remean
    Resummedall[RaFi,alismall] = Resummed
    ReUmeanall[RaFi,alismall] = ReUmean
    ReUsummedall[RaFi,alismall] = ReUsummed
    ReWmeanall[RaFi,alismall] = ReWmean
    ReWsummedall[RaFi,alismall] = ReWsummed
    Reall[RaFi,alismall] = Re
    Tmeanall[RaFi,alismall] = T_xzt_ave
    alphaall[RaFi,alismall] = alpha1000/1000
    RaFall[RaFi,alismall] = 10**RaFe
    Rediffall[:,RaFi,alismall] = np.array([ReUsummed,ReWsummed,ReUmean,ReWmean,Resummed,Remean,Re])
  
  ############## Tb figure -- updated 11/05/2022
  if fnmatch.fnmatch(all_files[i], '*restart*')==1 or fnmatch.fnmatch(all_files[i], '*aspect*')==1: 
    if RaFe==6: irow = 0; icol = 0 
    elif RaFe==7: irow = 0; icol = 1 
    elif RaFe==8: irow = 1; icol = 0 
    elif RaFe==9: irow = 1; icol = 1 
    if Gamma==4: irow = 2; icol = 0
    elif Gamma==16: irow = 2; icol = 1
    axTb[irow,icol].plot(xb,Tb_t_ave,color=my_colors[RaFi][ali])
    axTb[irow,icol].set_title(r'$Ra_F=10^{%i}$'%RaFe)
    axTb[irow,icol+2].plot(xb,(Tb_t_ave-Tt_t_ave),color=my_colors[RaFi][ali])
    axTb[irow,icol+2].set_title(r'$Ra_F=10^{%i}$'%RaFe)
  
  ############## FIGURE FOR PUBLICATION  
  RaF = 10**RaFe
  ratio = RaL/RaF
  RaLvec = np.logspace(9,12,10)
  RaFvec = np.logspace(6,9,10)
  print(RaLvec)
  ###
  if symb=='*':
    axpub1.plot(1.6,RaF,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
  elif Gamma==4:
    axpub1.plot(alpha,0.6*RaF,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
  elif Gamma==8:
    axpub1.plot(alpha,RaF,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
  elif Gamma==16:
    axpub1.plot(alpha,1.6*RaF,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
  ###
  if fnmatch.fnmatch(all_files[i], '*restart*')==1 or fnmatch.fnmatch(all_files[i], '*aspect*')==1: 
    print(all_files[i],xv.shape,autocor_u09_t_ave.shape)
    # NOTES ON SCALINGS
    ### Nu = 0.1Ra0.3 = RaF/Ra => RaF = 0.1Ra1.3 => Ra = (RaF/0.1)^(1/1.3) ; source: 2D large-aspect box Pr = 4,3 van der Poel+ 2012
    ### Nu = 0.19Ra0.270 = RaF/Ra => RaF = 0.19Ra1.270 => Ra = (RaF/0.19)^(1/1.270) ; source: 2D large-aspect periodic Pr = 10 Wang+ 2021
    ### Nu = 0.18Ra0.280 = RaF/Ra => RaF = 0.18Ra1.28 => Ra = (RaF/0.18)^(1/1.28) ; source: Couston 2021
    ### Re = 0.0085Ra0.588 = 0.0085(RaF/0.19)^(0.588/1.270) ; source: 2D large-aspect periodic Pr = 10 Wang+ 2021
    ### Re = 0.0086Ra0.580 = 0.0086(RaF/0.18)^(0.58/1.28) ; source: Couston 2021
    ### Re = (10**2*(Ra/1e7)**0.62) = 0.0046Ra0.62 = 0.0046(RaF/0.1)^(0.62/1.3); source: Sugiyama+ 2009
  ### HC
  # Nu = aRaB^b = RaB/RaD => RaD = RaB^(1-b)/a => RaB = (aRaD)^(1/(1-b)) => Nu=a(aRaD)^(b/(1-b))
  # Nu = 0.81587*RaB**(1/6) => Nu = 0.81587*(0.81587*RaD)**(1/5) 
  #
    if i==0: 
      axfin4[0,1].plot(RaFvec,cRB*RaFvec**dRB,'-k')
      axfin4[1,0].plot(RaLvec,cHC*RaLvec**dHC,'-k')
    axfin4[0,0].errorbar(alpha,T_xzt_ave,yerr=np.std(T_xz_ave),color=my_colors[RaFi][ali],ecolor=my_colors[RaFi][ali],marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axfin4[0,0].plot(alpha,-alpha*Gamma/2,color='k',marker=symb,markersize=symbsize)
    axfin4[0,1].errorbar(RaF,Remean,yerr=np.std(Remeant),color=my_colors[RaFi][ali],ecolor=my_colors[RaFi][ali],marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axfin4[1,0].errorbar(RaL,Remean,yerr=np.std(Remeant),color=my_colors[RaFi][ali],ecolor=my_colors[RaFi][ali],marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axfin4[1,1].plot(alpha,Remean/(cRB*RaF**dRB),color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
  
    # 
    if fnmatch.fnmatch(all_files[i], '*restart*')==1:
      if alpha==0:
        figpat6master[RaFi][1][0].plot(xv,topflux,'-',color=my_colors[RaFi][ali],linewidth=2,label=r'$\Lambda=0$')
        #figpat6master[RaFi][1][1].plot(xv,Tt_t_ave,'-k',linewidth=2)
        figpat6master[RaFi][1][1].plot(xv,Tb_t_ave-np.mean(Tb_t_ave),'-',color=my_colors[RaFi][ali],linewidth=2)
        figpat6bmaster[RaFi][1].plot(xv,Tb_t_ave-np.mean(Tb_t_ave)-Tt_t_ave,'-',color=my_colors[RaFi][ali],linewidth=2,label=r'$\Lambda=0$')
      #if alpha==0.001:
      #  figpat6master[RaFi][1][0].plot(xv,topflux,linestyle='--',color=my_colors[RaFi][ali],linewidth=2,label=r'$\Lambda=10^{-3}$')
      #  #figpat6master[RaFi][1][1].plot(xv,Tt_t_ave,'-k',linewidth=0.5)
      #  figpat6master[RaFi][1][1].plot(xv,Tb_t_ave-np.mean(Tb_t_ave),'--',color=my_colors[RaFi][ali],linewidth=2)
      #  figpat6bmaster[RaFi][1].plot(xv,Tb_t_ave-np.mean(Tb_t_ave)-Tt_t_ave,'-',color=my_colors[RaFi][ali],linewidth=2,label=r'$\Lambda=10^{-3}$')
      if alpha==0.01:
        figpat6master[RaFi][1][0].plot(xv,topflux,linestyle='--',color=my_colors[RaFi][ali],linewidth=2,label=r'$\Lambda=10^{-2}$')
        figpat6master[RaFi][1][1].plot(xv,Tt_t_ave,'-k',linewidth=0.5,zorder=1)
        figpat6master[RaFi][1][1].plot(xv,(Tb_t_ave-np.mean(Tb_t_ave)),'--',color=my_colors[RaFi][ali],linewidth=2)
        figpat6bmaster[RaFi][1].plot(xv,Tb_t_ave-np.mean(Tb_t_ave)-Tt_t_ave,'--',color=my_colors[RaFi][ali],linewidth=2,label=r'$\Lambda=10^{-2}$')
      if alpha==0.1:
        figpat6master[RaFi][1][0].plot(xv,topflux,linestyle='-.',color=my_colors[RaFi][ali],linewidth=2,label=r'$\Lambda=10^{-1}$')
        #figpat6master[RaFi][1][1].plot(xv,Tt_t_ave,'-k',linewidth=0.5,zorder=10)
        figpat6master[RaFi][1][1].plot(xv,(Tb_t_ave-np.mean(Tb_t_ave)),'-.',color=my_colors[RaFi][ali],linewidth=2)
        figpat6bmaster[RaFi][1].plot(xv,Tb_t_ave-np.mean(Tb_t_ave)-Tt_t_ave,'-.',color=my_colors[RaFi][ali],linewidth=2,label=r'$\Lambda=10^{-1}$')
      if alpha==1.0:
        figpat6master[RaFi][1][0].plot(xv,topflux,':',color=my_colors[RaFi][ali],linewidth=2,label=r'$\Lambda=1$')
        #figpat6master[RaFi][1][1].plot(xv,Tt_t_ave,'-k',linewidth=1,label=r'$T(z=1;\Lambda=1)$',zorder=1)
        figpat6master[RaFi][1][1].plot(xv,Tb_t_ave-np.mean(Tb_t_ave),':',color=my_colors[RaFi][ali],linewidth=2)
        figpat6bmaster[RaFi][1].plot(xv,Tb_t_ave-np.mean(Tb_t_ave)-Tt_t_ave,':',color=my_colors[RaFi][ali],linewidth=2,label=r'$\Lambda=1$')
    #
    print('std',NuRB,np.std(NuRBstd)/NuRB)
    #
    
    #### NEW FIGURE IN APPE FOR NUHC --- updated 13/05/2022
    print('The norms are !!!',alpha,Nu_norm,Nucold_norm,Nusurf_norm)
    if i==0:
      [axappE[0,i].plot(RaLvec,aHC*RaLvec**bHC,'-k') for i in range(3)]
      #[axappE[0,i].plot(RaLvec,aHC*RaLvec**bHC/20,'--k') for i in range(3)]
      [axappE[1,i].axhline(1,linestyle='--',color='k') for i in range(3)]
    axappE[0,0].plot(RaL,Nu,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axappE[0,1].plot(RaL,Nucold,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axappE[0,2].plot(RaL,Nusurf,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axappE[1,0].plot(alpha,Nu/(aHC*RaL**bHC)*Nu_norm/Nu_norm_noGH,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axappE[1,1].plot(alpha,Nucold/(aHC*RaL**bHC)*Nucold_norm/(Nucold_norm-1*(alpha<=1)),color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axappE[1,2].plot(alpha,Nusurf/(aHC*RaL**bHC)*Nusurf_norm/(Nusurf_norm-1*(alpha<=1)),color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    
    
    #### BIG NEW FIGURE FOR ANALYSIS (NOT PUBLICATION) --- updated 11/05/2022
    if i==0: 
      axhalf[3,0].plot(RaFvec,aRB*RaFvec**bRB,'-k')
      axhalf[3,1].plot(RaFvec,aRB*RaFvec**bRB,'-k')
    axhalf[3,0].errorbar(RaF, NuRB, yerr=np.std(NuRBt),color=my_colors[RaFi][ali], ecolor=my_colors[RaFi][ali], marker=symb,markersize=symbsize, markeredgewidth=1.5, markerfacecolor='none')
    axhalf[3,1].errorbar(RaF, NuaveRB, yerr=np.std(NuaveRBt),color=my_colors[RaFi][ali], ecolor=my_colors[RaFi][ali], marker=symb,markersize=symbsize, markeredgewidth=1.5, markerfacecolor='none')
    axhalf[3,2].plot(alpha,NuRB,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axhalf[3,3].plot(alpha,NuaveRB,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axhalf[3,4].plot(alpha,NuRB/(aRB*RaF**bRB),color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    #
    axhalf[2,0].errorbar(RaL, AbsFlux, yerr=np.std(AbsFluxt),color=my_colors[RaFi][ali], ecolor=my_colors[RaFi][ali], marker=symb,markersize=symbsize, markeredgewidth=1.5, markerfacecolor='none')
    axhalf[2,1].plot(RaL, Nu, color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axhalf[2,2].plot(alpha,AbsFlux,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axhalf[2,3].plot(alpha,Nu,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axhalf[2,4].plot(alpha,Nu/(aHC*RaL**bHC)*Nu_norm/Nu_norm_noGH*(np.tanh(np.pi/Gamma)/np.tanh(np.pi/8)),color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    #
    axhalf[1,0].errorbar(RaL, Ftophalf,yerr=np.std(Ftophalft),color=my_colors[RaFi][ali], ecolor=my_colors[RaFi][ali], marker=symb,markersize=symbsize, markeredgewidth=1.5, markerfacecolor='none')
    axhalf[1,1].plot(RaL, Nucold,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axhalf[1,2].plot(alpha,Ftophalf,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axhalf[1,3].plot(alpha,Nucold,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axhalf[1,4].plot(alpha,Nucold/(aHC*RaL**bHC)*Nucold_norm/(Nucold_norm-1*(alpha<=1))*(np.tanh(np.pi/Gamma)/np.tanh(np.pi/8)),color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    #
    axhalf[0,0].plot(RaL,TopCorrelation,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')  
    if i==0:
      axhalf[0,1].plot(RaLvec,aHC*RaLvec**bHC,'-k')
    axhalf[0,1].errorbar(RaL,Nusurf, yerr=np.std(Nusurft),color=my_colors[RaFi][ali], ecolor=my_colors[RaFi][ali], marker=symb,markersize=symbsize, markeredgewidth=1.5, markerfacecolor='none')  
    axhalf[0,2].plot(alpha,TopCorrelation,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axhalf[0,3].plot(alpha,Nusurf, color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axhalf[0,4].plot(alpha,Nusurf/(aHC*RaL**bHC)*Nusurf_norm/(Nusurf_norm-1*(alpha<=1))*(np.tanh(np.pi/Gamma)/np.tanh(np.pi/8)),color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    #  
    #
    if Ftophalf<0: print('MASSIVE WARNING !!!!!!!!!!!!')
       
    #### FIG7 SPLIT --- updated 13/05/2022
    if i==0: 
      ax7a[0].plot(RaFvec,aRB*RaFvec**bRB,'-k')
      ax7b[0].plot(RaLvec,aHC*RaLvec**bHC,'-k')
      ax7b[0].plot(RaLvec,0.01*aHC*RaLvec**bHC,'--k')
    ax7a[0].errorbar(RaF,NuRB,yerr=np.std(NuRBstd),color=my_colors[RaFi][ali],ecolor=my_colors[RaFi][ali],marker=symb,markersize=symbsize, markeredgewidth=1.5,markerfacecolor='none')
    ax7a[1].plot(alpha,NuRB/(aRB*RaF**bRB),color=my_colors[RaFi][ali], marker=symb,markersize=symbsize, markeredgewidth=1.5,markerfacecolor='none')
    ax7b[0].errorbar(RaL,Nusurf,yerr=np.std(Nusurft),color=my_colors[RaFi][ali], ecolor=my_colors[RaFi][ali], marker=symb,markersize=symbsize, markeredgewidth=1.5, markerfacecolor='none')
    ax7b[1].plot(alpha,Nusurf/(aHC*RaL**bHC),color=my_colors[RaFi][ali], marker=symb,markersize=symbsize, markeredgewidth=1.5,markerfacecolor='none')
    ax7b[2].plot(alpha,Nusurf/(aHC*RaL**bHC)*Nusurf_norm/(Nusurf_norm-1*(alpha<=1))*(np.tanh(np.pi/Gamma)/np.tanh(np.pi/8)),color=my_colors[RaFi][ali], marker=symb,markersize=symbsize, markeredgewidth=1.5,markerfacecolor='none')
       
    
    #### PREVIOUS FIGURES FOR PUBLICATION (KEPT IN REVISION); Note that axfin6 was split in two
    #
    if i==0: 
      axfin6[0,0].plot(RaFvec,aRB*RaFvec**bRB,'-k')
      axfin6[1,0].plot(RaLvec,aHC*RaLvec**bHC,'-k')
    axfin6[0,0].errorbar(RaF,NuRB,yerr=np.std(NuRBstd),color=my_colors[RaFi][ali],ecolor=my_colors[RaFi][ali],marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axfin6[1,0].errorbar(RaL,Nusurf,yerr=np.std(Nusurft),color=my_colors[RaFi][ali], ecolor=my_colors[RaFi][ali], marker=symb,markersize=symbsize, markeredgewidth=1.5, markerfacecolor='none')
    axfin6[0,1].plot(alpha,NuRB/(aRB*RaF**bRB),color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axfin6[1,1].plot(alpha,Nusurf/(aHC*RaL**bHC)*np.tanh(np.pi/Gamma)/np.tanh(np.pi/8),color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    ##############################
    axfin6b[0].plot(alpha,Nusurf,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axfin6b[0].plot(alpha,Nu,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=0.5,markerfacecolor='none')
    axfin6b[1].plot(RaL,Nusurf,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axfin6b[1].plot(RaL,Nu,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=0.5,markerfacecolor='none')
    axfin6b[2].plot(alpha,np.abs(Nusurf-Nu)/(Nusurf),color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    if alpha==1.001:
      axfin6b[0].plot(alpha,Nu,color='k', marker=symb,markersize=symbsize, markeredgewidth=0.5,markerfacecolor='none',zorder=-1000)
      axfin6b[1].plot(RaL,Nu,color='k', marker=symb,markersize=symbsize, markeredgewidth=0.5,markerfacecolor='none',zorder=-1000)
      
    # NB: we don't show the std for u09 as it is much larger than the mean for small RaL/RaF
    axfin7.plot(alpha,-u09/Re,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    #
    axfin7b[0].plot(alpha,-u09,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    axfin7b[1].plot(RaL,-u09,color=my_colors[RaFi][ali], marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    #
    print('std',-u09,np.std(u09t))
    # 
    if fnmatch.fnmatch(all_files[i], '*restart*')==1:
      if alismall==1 or alismall==len(alpha1000all)-2: # alpha=1 or alpha=1000 
        index = 1*(alismall==len(alpha1000all)-2)
        xx, tt = np.meshgrid(xv,time[itsss:])
        im=figpub8master[RaFi][1][index][0].pcolormesh(xx,tt,data['u09'][itsss:],shading='gouraud', norm=colors.SymLogNorm(linthresh=-data['u09'][itsss:].min()/10, linscale=1, vmin=data['u09'][itsss:].min()*2, vmax=-data['u09'][itsss:].min()*2), cmap=cm.RdYlBu_r)
        colorbar(im)
        if RaFi==2:
          im=figpub8master[RaFi][1][index][1].pcolormesh(xv-xv.min(),tt,autocor_u09,shading='gouraud',cmap=cm.PRGn,vmin=-1,vmax=1)
          colorbar(im)
          figpub8master[RaFi][1][index][2].axvline(xRuu09min,linestyle='--',color='k')
          figpub8master[RaFi][1][index][2].plot(xv-xv.min(),autocor_u09_t_ave,color='tab:green')
        
        else:
          im=figpub8master[RaFi][1][index][1].pcolormesh(xv-xv.min(),tt,autocor_u09_my_adjusted,shading='gouraud',cmap=cm.PRGn,vmin=-1,vmax=1)
          colorbar(im)
          figpub8master[RaFi][1][index][2].axvline(xRuu09min,linestyle='--',color='k')
          figpub8master[RaFi][1][index][2].axvline(xRuu09max,linestyle='--',color='k')
          figpub8master[RaFi][1][index][2].plot(xv-xv.min(),autocor_u09_my_adjusted_t_ave,color='tab:green')
          figpub8master[RaFi][1][index][2].plot(xv-xv.min(),autocor_u09_my_t_ave,'--',color='tab:orange')
          figpub8master[RaFi][1][index][2].plot(xv-xv.min(),autocor_u09_t_ave,':',color='k')
    #
    axfin9.plot(alpha,xRuu09min,color=my_colors[RaFi][ali],marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    #axfin9.plot(alpha,xRuu09max,color=my_colors[RaFi][ali],marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    #
    axpsi.plot(alpha,psi_tx_ave_max,color=my_colors[RaFi][ali],marker=symb,markersize=symbsize,markeredgewidth=1.5,markerfacecolor='none')
    
  ############## FIGURE FOR PUBLICATION  
  

print(NuHCall)


############################
### Re details and save data
# Rediffall.append([ReUsummed,ReWsummed,ReUmean,ReWmean,Resummed,Remean,Re])
#Rediffall = np.concatenate(Rediffall,axis=1)
print(Rediffall.shape)
print(Rediffall)
print('(ReUs-ReUm)/ReUs',(Rediffall[0]-Rediffall[2])/Rediffall[0])
print('(ReWs-ReWm)/ReWs',(Rediffall[1]-Rediffall[3])/Rediffall[1])
print('(Res-Rem)/Res',(Rediffall[4]-Rediffall[5])/Rediffall[4])
print('(Res-Re)/Res',(Rediffall[4]-Rediffall[6])/Rediffall[4])
print('(Rem-Re)/Rem',(Rediffall[5]-Rediffall[6])/Rediffall[5])
print('ReWs/ReUs',Rediffall[1]/Rediffall[0])
print('ReWm/ReUm',Rediffall[3]/Rediffall[2])
#
np.savez('data_for_fit.npz', RaF=RaFall, alpha=alphaall, NuRB=NuRBall, NuHC=NuHCall, NuHCsurf=NuHCsurfall, Re=Reall, Remean=Remeanall, Resummed=Resummedall, ReUmean=ReUmeanall, ReUsummed=ReUsummedall, ReWmean=ReWmeanall, ReWsummed=ReWsummedall, RaL=RaLall) 
#



############## FIGURE FOR PUBLICATION -- updated 10/06/2022
### INCLUDES FIGURES 1b, 5, 6, 7, 8, 9, 10, 11


axpub1.axvspan(-1e-4, 3e-3, facecolor='tab:purple', alpha=0.25, zorder=-100)
axpub1.axvspan(3e-2, 2, facecolor='tab:gray', alpha=0.25, zorder=-100)
axpub1.set_xlim((0,1))
axpub1.set_xscale('symlog',linthresh=1e-3,linscale=1) 
axpub1.set_xlim((-1e-4,2)) 
axpub1.set_yscale('log')
axpub1.set_xlabel(r'$\Lambda$',fontsize='large')
axpub1.set_ylabel(r'$Ra_F$',fontsize='large')
figpub1.tight_layout(pad=0.02)
#figpub1.savefig('final1b.png',dpi=300,bbox_inches='tight',pad_inches=0.02)
#figpub1.savefig('Figure1brevision.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
figpub1.savefig('Figure1b.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
plt.close()
#

axfin4[0,1].set_xscale('log')
axfin4[1,0].set_xscale('symlog',linthresh=1e6,linscale=1)
axfin4[1,0].set_xticks([0,1e6,1e8,1e10,1e12])
axfin4[1,1].set_xscale('symlog',linthresh=1e-3,linscale=1) 
axfin4[0,0].set_xscale('symlog',linthresh=1e-3,linscale=1) 
axfin4[0,0].set_yscale('symlog',linthresh=1e-1,linscale=1)
axfin4[1,0].set_yscale('log')
axfin4[0,1].set_yscale('log')
axfin4[1,1].set_yscale('linear')
axfin4[0,0].set_xlabel(r'$\Lambda$')  
axfin4[1,0].set_xlabel(r'$Ra_L$') 
axfin4[0,1].set_xlabel(r'$Ra_F$')  
axfin4[1,1].set_xlabel(r'$\Lambda$')  
axfin4[0,0].set_ylabel(r'$\overline{\langle T \rangle}$',fontsize='large')
axfin4[1,0].set_ylabel(r'$Re$',fontsize='large')
axfin4[0,1].set_ylabel(r'$Re$',fontsize='large') 
axfin4[1,1].set_ylabel(r'$\frac{Re}{c_{RB}Ra_F^{d_{RB}}}$',fontsize='x-large') 
figfin4.tight_layout(pad=0.05)
#figfin4.savefig('final5.png',dpi=300,bbox_inches='tight',pad_inches=0.02)
#figfin4.savefig('Figure5revision.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
figfin4.savefig('Figure5.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
#figfin4.savefig('Figure5revision.eps',bbox_inches='tight',pad_inches=0.02)
plt.close()

###
for i in range(len(RaFeall)):
  [figpat6master[i][1][1].set_xlabel(r'$x$',fontsize='large') for j in range(2)]
  [figpat6master[i][1][0].set_ylabel(r'$-\overline{\frac{dT}{dz}}\left|_{z=1}\right.$',fontsize='x-large',labelpad=5) for j in range(2)]
  #[figpat6master[i][1][1].set_ylabel(r'$\overline{T(z=0)}-\langle \overline{T(z=0)}\rangle_x$',fontsize='large',labelpad=5) for j in range(2)]
  [figpat6master[i][1][1].set_ylabel(r'$[\overline{T}-\langle \overline{T}\rangle_x](z=0)$',fontsize='large',labelpad=5) for j in range(2)]
  [figpat6master[i][1][0].set_yscale('symlog',linthresh=1,linscale=1) for j in range(2)]
  #[figpat6master[i][1][1].set_yscale('symlog',linthresh=1e-2,linscale=1) for j in range(2)]
  [figpat6master[i][1][0].set_xlim((-4.1,4.1)) for j in range(2)]
  [figpat6master[i][1][1].set_xlim((-4.1,4.1)) for j in range(2)]
  [figpat6master[i][1][0].legend(frameon=False,loc='lower left',fontsize='medium',columnspacing=0.5,borderpad=0.2,handletextpad=0.3,borderaxespad=0.2,ncol=2) for j in range(2)]
  [figpat6master[i][1][1].legend(frameon=False,loc='lower right',fontsize='medium',borderpad=0.2,handletextpad=0.3,borderaxespad=0.2,ncol=1) for j in range(2)]
  [figpat6master[i][1][0].set_xticks([-4,-2,0,2,4]) for j in range(2)]
  [figpat6master[i][1][1].set_xticks([-4,-2,0,2,4]) for j in range(2)]
  figpat6master[i][0].tight_layout(pad=0.05)
  figpat6master[i][0].savefig('publication-topflux-6-%i.png'%i,dpi=dpi)
  if i==len(RaFeall)-2:
    #figpat6master[i][0].savefig('final6.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
    figpat6master[i][0].savefig('Figure6.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
  plt.close()
  

############ FIG 7 SPLIT
ax7a[0].set_xscale('log')
ax7a[1].set_xscale('symlog',linthresh=1e-3,linscale=1) 
ax7a[0].set_yscale('log')
ax7a[1].set_yscale('log')
ax7a[0].set_xlabel(r'$Ra_F$')  
ax7a[1].set_xlabel(r'$\Lambda$')  
ax7a[0].set_ylabel(r'$Nu_{RB}$',fontsize='large')
ax7a[1].set_ylabel(r'$\frac{Nu_{RB}}{a_{RB}Ra_F^{b_{RB}}}$',fontsize='x-large') 
ax7a[1].axvspan(3e-2, 1.2, facecolor='tab:gray', alpha=0.25, zorder=-100)
ax7a[1].axvspan(-1e-4, 3e-3, facecolor='tab:purple', alpha=0.25, zorder=-100)
ax7a[1].set_xlim((-1e-4,1.2)) 
fig7a.tight_layout(pad=0.05)
#fig7a.savefig('Figure7arevision.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
fig7a.savefig('Figure7.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
#fig7a.savefig('Figure7arevision.eps',bbox_inches='tight',pad_inches=0.02)
plt.close()


ax7b[0].set_xscale('symlog',linthresh=1e6,linscale=1)
ax7b[0].set_xticks([0,1e6,1e8,1e10,1e12])
ax7b[0].set_yscale('log')
ax7b[0].set_xlabel(r'$Ra_L$') 
ax7b[0].set_ylabel(r'$Nu^{\chi}_{HC}$',fontsize='large')
ax7b[1].set_xscale('symlog',linthresh=1e-3,linscale=1) 
ax7b[1].set_yscale('log')
ax7b[1].axvspan(3e-2, 1.2, facecolor='tab:gray', alpha=0.25, zorder=-100) 
ax7b[1].axvspan(-1e-4, 3e-3, facecolor='tab:purple', alpha=0.25, zorder=-100) 
ax7b[1].set_xlim((-1e-4,1.2)) 
ax7b[1].set_xlabel(r'$\Lambda$')  
ax7b[1].set_ylabel(r'$\frac{Nu^{\chi}_{HC}}{a_{HC}Ra_L^{b_{HC}}}$',fontsize='x-large') 
ax7b[2].set_xscale('symlog',linthresh=1e-3,linscale=1) 
ax7b[2].set_yscale('log')
ax7b[2].axvspan(3e-2, 1.2, facecolor='tab:gray', alpha=0.25, zorder=-100) 
ax7b[2].axvspan(-1e-4, 3e-3, facecolor='tab:purple', alpha=0.25, zorder=-100) 
ax7b[2].set_xlim((-1e-4,1.2)) 
ax7b[2].set_xlabel(r'$\Lambda$')  
ax7b[2].set_ylabel(r'$\frac{Nu^{\chi}_{HC}}{a_{HC}Ra_L^{b_{HC}}}\times\frac{\chi_{\mathrm{diff}}}{\chi_{\mathrm{dim}}}$',fontsize='x-large') 
fig7b.tight_layout(pad=0.02)
#fig7b.savefig('Figure7brevision.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
fig7b.savefig('Figure8.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
#fig7b.savefig('Figure7brevision.eps',bbox_inches='tight',pad_inches=0.02)
plt.close()
############  
  

#
axfin7.axvspan(3e-2, 1.2, facecolor='tab:gray', alpha=0.25, zorder=-100)
axfin7.axvspan(-1e-4, 3e-3, facecolor='tab:purple', alpha=0.25, zorder=-100)
axfin7.set_xlim((-1e-4,1.2))
axfin7.set_xscale('symlog',linthresh=1e-3,linscale=1) 
axfin7.set_yscale('symlog',linthresh=1e-1,linscale=1)
axfin7.set_ylabel(r'$\frac{\overline{\langle -u(z=0.9) \rangle}}{Re}$',fontsize='large')
axfin7.set_xlabel(r'$\Lambda$')  
figfin7.tight_layout(pad=0.05)
#figfin7.savefig('Figure8revision.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
figfin7.savefig('Figure9.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
#figfin7.savefig('Figure8revision.eps',bbox_inches='tight',pad_inches=0.02)
plt.close()
#
###
for i in range(len(RaFeall)):
  [figpub8master[i][1][j,0].set_ylabel(r'$t$',fontsize='large') for j in range(2)]
  [figpub8master[i][1][j,1].set_ylabel(r'$t$',fontsize='large') for j in range(2)]
  [figpub8master[i][1][j,2].set_ylabel(r'$\overline{\mathcal{R}_{uu}}(z=0.9)$',fontsize='large') for j in range(2)]
  [figpub8master[i][1][j,0].set_xlim((-4,4)) for j in range(2)]
  [figpub8master[i][1][j,1].set_xlim((0,8)) for j in range(2)]
  [figpub8master[i][1][j,2].set_xlim((0,8.1)) for j in range(2)]
  [figpub8master[i][1][j,0].set_xticks([-4,-2,0,2,4]) for j in range(2)]
  [figpub8master[i][1][j,1].set_xticks([0,2,4,6,8]) for j in range(2)]
  [figpub8master[i][1][j,2].set_xticks([0,2,4,6,8]) for j in range(2)]
  [figpub8master[i][1][j,0].set_xlabel(r'$x$',fontsize='large') for j in range(2)]
  [figpub8master[i][1][j,1].set_xlabel(r'$x-$lag',fontsize='large') for j in range(2)]
  [figpub8master[i][1][j,2].set_xlabel(r'$x-$lag',fontsize='large') for j in range(2)]
  [figpub8master[i][1][j,0].set_ylim((1,1.2)) for j in range(2)]
  [figpub8master[i][1][j,1].set_ylim((1,1.2)) for j in range(2)]
  [figpub8master[i][1][j,0].set_title(r'$u(z=0.9)$',fontsize='large') for j in range(2)]
  [figpub8master[i][1][j,1].set_title(r'$\mathcal{R}_{uu}(z=0.9)$',fontsize='large') for j in range(2)]
  figpub8master[i][0].tight_layout(pad=0.05)
  figpub8master[i][0].savefig('publication8-%i.png'%i,dpi=dpi)
  if i==len(RaFeall)-2:
    #figpub8master[i][0].savefig('final9.png',dpi=300,bbox_inches='tight',pad_inches=0.02)
#    figpub8master[i][0].savefig('Figure9revision.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
    figpub8master[i][0].savefig('Figure10.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
  plt.close()
#

axfin9.set_xscale('symlog',linthresh=1e-3,linscale=1) 
axfin9.set_yscale('linear') 
axfin9.set_ylabel(r'$\ell$',fontsize='large')
axfin9.set_xlabel(r'$\Lambda$',fontsize='large')
axfin9.text(0,16.2,r'$16$',fontsize='medium')
axfin9.text(0,8.2,r'$8$',fontsize='medium')
axfin9.text(0,4.2,r'$4$',fontsize='medium')
axfin9.text(0.7,1.2,r'$1$',fontsize='medium')
axfin9.axvspan(3e-2, 1.2, facecolor='tab:gray', alpha=0.25, zorder=-100)
axfin9.axvspan(-1e-4, 3e-3, facecolor='tab:purple', alpha=0.25, zorder=-100)
axfin9.set_xlim((-1e-4,1.2))
axfin9.set_ylim((0,18))
figfin9.tight_layout(pad=0.05)
#figfin9.savefig('final10.png',dpi=300,bbox_inches='tight',pad_inches=0.02)
#figfin9.savefig('Figure10revision.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
figfin9.savefig('Figure11.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
plt.close()


#
############## appE FIGURE -- updated 13/05/2022
[axappE[0,i].set_xlabel(r'$Ra_L$',fontsize='large') for i in range(3)] 
[axappE[1,i].set_xlabel(r'$\Lambda$',fontsize='large') for i in range(3)] 
axappE[0,0].set_ylabel(r'$Nu^{abs}_{HC}$',fontsize='large') 
axappE[0,1].set_ylabel(r'$Nu^{half}_{HC}$',fontsize='large')
axappE[0,2].set_ylabel(r'$Nu^{\chi}_{HC}$',fontsize='large')
axappE[1,0].set_ylabel(r'$\frac{Nu^{abs}_{HC}(F=0)}{a_{HC}Ra_L^{b_{HC}}}$',fontsize='x-large') 
axappE[1,1].set_ylabel(r'$\frac{Nu^{half}_{HC}(F=0)}{a_{HC}Ra_L^{b_{HC}}}$',fontsize='x-large')
axappE[1,2].set_ylabel(r'$\frac{Nu^{\chi}_{HC}(F=0)}{a_{HC}Ra_L^{b_{HC}}}$',fontsize='x-large')
[axappE[0,i].set_xscale('log') for i in range(3)]
[axappE[1,i].set_xscale('symlog',linthresh=1e-3,linscale=1) for i in range(3)]
[[axappE[j,i].set_yscale('log') for i in range(3)] for j in range(2)]
[axappE[1,i].set_ylim((5e-1,5e3)) for i in range(3)]
[axappE[1,j].axvspan(3e-2, 1.2, facecolor='tab:gray', alpha=0.25, zorder=-100) for j in range(3)]
[axappE[1,j].axvspan(-1e-4, 3e-3, facecolor='tab:purple', alpha=0.25, zorder=-100) for j in range(3)]
[axappE[1,j].set_xlim((-1e-4,1.2)) for j in range(3)]
figappE.tight_layout(pad=0.02)
#figappE.savefig('FigureappErevision.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
figappE.savefig('Figure15.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
plt.close()

#
############## OTHER FIGURES NOT FOR PUBLICATION 

#
axfin7b[0].axvspan(3e-2, 1.2, facecolor='tab:gray', alpha=0.25, zorder=-100)
axfin7b[0].axvspan(-1e-4, 3e-3, facecolor='tab:purple', alpha=0.25, zorder=-100)
axfin7b[0].set_xlim((-1e-4,1.2))
axfin7b[0].set_xscale('symlog',linthresh=1e-3,linscale=1) 
#axfin7b[0].set_yscale('symlog',linthresh=1e-1,linscale=1)
[axfin7b[i].set_ylabel(r'$\overline{\langle -u(z=0.9) \rangle}$',fontsize='large') for i in range(2)]
axfin7b[0].set_xlabel(r'$\Lambda$')  
axfin7b[1].set_xscale('symlog',linthresh=1e6,linscale=1)
axfin7b[1].set_xlabel(r'$Ra_L$')  
figfin7b.tight_layout(pad=0.05)
figfin7b.savefig('final8explore.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
plt.close()


### UPDATED 24/05/2022
for i in range(len(RaFeall)):
  [figpat6bmaster[i][1].set_xlabel(r'$x$',fontsize='large') for j in range(2)]
  [figpat6bmaster[i][1].set_ylabel(r'$\overline{T(z=0)}-\langle \overline{T(z=0)}\rangle_x-T_t$',fontsize='large',labelpad=5) for j in range(2)]
  [figpat6bmaster[i][1].set_yscale('symlog',linthresh=1e-2,linscale=1) for j in range(2)]
  [figpat6bmaster[i][1].legend(frameon=False,fontsize='medium',borderpad=0.2,handletextpad=0.4,borderaxespad=0.2) for j in range(2)]
  [figpat6bmaster[i][1].set_xticks([-4,-2,0,2,4]) for j in range(2)]
  figpat6bmaster[i][0].tight_layout(pad=0.05)
  figpat6bmaster[i][0].savefig('publication-hortemp-6-%i.png'%i,dpi=dpi)
  if i==len(RaFeall)-2:
    figpat6bmaster[i][0].savefig('final6b.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
  plt.close()


axfin6[0,0].set_xscale('log')
axfin6[1,0].set_xscale('symlog',linthresh=1e6,linscale=1)
axfin6[1,0].set_xticks([0,1e6,1e8,1e10,1e12])
axfin6[0,1].set_xscale('symlog',linthresh=1e-3,linscale=1) 
axfin6[1,1].set_xscale('symlog',linthresh=1e-3,linscale=1) 
axfin6[0,0].set_yscale('log')
axfin6[1,0].set_yscale('log')
axfin6[0,1].set_yscale('log')
axfin6[1,1].set_yscale('log')
[axfin6[i,1].axvspan(3e-2, 1.2, facecolor='tab:gray', alpha=0.25, zorder=-100) for i in range(2)]
[axfin6[i,1].axvspan(-1e-4, 3e-3, facecolor='tab:purple', alpha=0.25, zorder=-100) for i in range(2)]
[axfin6[i,1].set_xlim((-1e-4,1.2)) for i in range(2)]
axfin6[0,0].set_xlabel(r'$Ra_F$')  
axfin6[1,0].set_xlabel(r'$Ra_L$') 
axfin6[0,1].set_xlabel(r'$\Lambda$')  
axfin6[1,1].set_xlabel(r'$\Lambda$')  
axfin6[0,0].set_ylabel(r'$Nu_{RB}$',fontsize='large')
axfin6[1,0].set_ylabel(r'$Nu_{HC}$',fontsize='large')
axfin6[0,1].set_ylabel(r'$\frac{Nu_{RB}}{a_{RB}Ra_F^{b_{RB}}}$',fontsize='x-large') 
axfin6[1,1].set_ylabel(r'$\frac{Nu_{HC}}{a_{HC}Ra_L^{b_{HC}}}\times\frac{\tanh(\pi/\Gamma)}{\tanh(\pi/8)}$',fontsize='x-large') 
figfin6.tight_layout(pad=0.05)
figfin6.savefig('Figure7revision.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
figfin6.savefig('Figure7revision.eps',bbox_inches='tight',pad_inches=0.02)
plt.close()
#

axfin6b[0].set_yscale('log')
axfin6b[1].set_yscale('log')
axfin6b[2].set_yscale('symlog',linthresh=1e-3,linscale=1)
axfin6b[0].set_xscale('symlog',linthresh=1e-3,linscale=1) 
axfin6b[1].set_xscale('log') 
axfin6b[2].set_xscale('symlog',linthresh=1e-3,linscale=1) 
axfin6b[0].set_xlim((0,1))
axfin6b[2].set_xlim((0,1))
axfin6b[0].set_xlabel(r'$\Lambda$') 
axfin6b[1].set_xlabel(r'$Ra_L$') 
axfin6b[2].set_xlabel(r'$\Lambda$') 
axfin6b[2].set_ylabel(r'rel. diff.') 
figfin6b.tight_layout(pad=0.05)
figfin6b.savefig('final7bis.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
plt.close()


############## Tb FIGURE -- updated 11/05/2022
[[axTb[2,j].set_xlabel(r'$x$',fontsize='large') for i in range(3)] for j in range(4)]
[[axTb[i,j].set_xticklabels([]) for i in range(2)] for j in range(4)]
[[axTb[i,0].set_ylabel(r'$\overline{T}_b$',fontsize='large') for i in range(3)] for j in range(2)]
[[axTb[i,2].set_ylabel(r'$(\overline{T}_b-T_t)/\Lambda$',fontsize='large') for i in range(3)] for j in range(2)]
plt.tight_layout(pad=2)
figTb.savefig('Tbrevision.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
plt.close()

############## Ftophalf FIGURE -- updated 12/05/2022
[axhalf[3,i].set_xlabel(r'$Ra_L$',fontsize='large') for i in range(2)] 
axhalf[3,2].set_xlabel(r'$\Lambda$',fontsize='large')
axhalf[3,3].set_xlabel(r'$\Lambda$',fontsize='large')
axhalf[3,4].set_xlabel(r'$\Lambda$',fontsize='large')
axhalf[1,0].set_ylabel(r'$F^{x<0}(z=1)$',fontsize='large')
axhalf[1,1].set_ylabel(r'$(F^{x<0}/F_{diff}^{x<0})(z=1)$',fontsize='large')
axhalf[1,3].set_ylabel(r'$(F^{x<0}/F_{diff}^{x<0})(z=1)$',fontsize='large')
axhalf[1,2].set_ylabel(r'$F^{x<0}(z=1)$',fontsize='large')
axhalf[2,0].set_ylabel(r'$|F(z=1)|$',fontsize='large')
axhalf[2,1].set_ylabel(r'$|F(z=1)|/|F_{diff}(z=1)|$',fontsize='large')
axhalf[2,2].set_ylabel(r'$|F(z=1)|$',fontsize='large')
axhalf[2,3].set_ylabel(r'$|F(z=1)|/|F_{diff}(z=1)|$',fontsize='large')
[[axhalf[i,j].set_xscale('log') for i in range(4)] for j in range(2)]
[[axhalf[i,j].set_yscale('log') for i in range(4)] for j in range(5)]
[axhalf[i,2].set_xscale('symlog',linthresh=1e-3,linscale=1) for i in range(4)]
[axhalf[i,3].set_xscale('symlog',linthresh=1e-3,linscale=1) for i in range(4)]
[axhalf[i,4].set_xscale('symlog',linthresh=1e-3,linscale=1) for i in range(4)]
axhalf[0,0].set_ylabel(r'$\chi$',fontsize='large')
axhalf[0,2].set_ylabel(r'$\chi$',fontsize='large')
axhalf[0,1].set_ylabel(r'$\chi/\chi_{diff}$',fontsize='large')
axhalf[0,3].set_ylabel(r'$\chi/\chi_{diff}$',fontsize='large')
axhalf[3,0].set_ylabel(r'$Nu_{RB}$',fontsize='large')
axhalf[3,1].set_ylabel(r'$Nu^{ave}_{RB}$',fontsize='large')
axhalf[3,2].set_ylabel(r'$Nu_{RB}$',fontsize='large')
axhalf[3,3].set_ylabel(r'$Nu^{ave}_{RB}$',fontsize='large')
axhalf[3,4].set_ylabel(r'$Nu_{RB}/scaling$',fontsize='large')
[[axhalf[i,j].axvspan(3e-2, 1.2, facecolor='tab:gray', alpha=0.25, zorder=-100) for i in range(4)] for j in [2, 3, 4]]
[[axhalf[i,j].axvspan(-1e-4, 3e-3, facecolor='tab:purple', alpha=0.25, zorder=-100) for i in range(4)] for j in [2, 3, 4]]
[[axhalf[i,j].set_xlim((-1e-4,1.2)) for i in range(4)] for j in [2, 3, 4]]
fighalf.tight_layout(pad=0.02)
fighalf.savefig('Halfrevision.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
plt.close()



### FIGURE FOR MAX OF STREAMFUNCTION -- updated 13/09/2022
axpsi.set_xscale('symlog',linthresh=1e-3,linscale=1) 
axpsi.set_yscale('linear') 
axpsi.set_ylabel(r'$\psi_{max}$',fontsize='large')
axpsi.set_xlabel(r'$\Lambda$',fontsize='large')
axpsi.axvspan(3e-2, 1.2, facecolor='tab:gray', alpha=0.25, zorder=-100)
axpsi.axvspan(-1e-4, 3e-3, facecolor='tab:purple', alpha=0.25, zorder=-100)
axpsi.set_xlim((-1e-4,1.2))
axpsi.set_ylim((0,18))
figpsi.tight_layout(pad=0.05)
figpsi.savefig('FigurePSI.png',dpi=dpi,bbox_inches='tight',pad_inches=0.02)
plt.close()






#calesc = np.array(calesc)
#print('HERE IS MY: ',calesc.shape)
#print('HERE IS MY: ',calesc[:,0])
#print('HERE IS MY: ',calesc[:,1])
#print(calesc[:,1].min())
#calesc[calesc[:,1]>2,1]=0
#print(calesc[:,1].max())





