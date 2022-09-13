"""
Usage:
    postprocess-data.py <files>
"""
from docopt import docopt
args = docopt(__doc__)
import numpy as np
import os, os.path, fnmatch, h5py, re
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from hc_support import *

foldername = args['<files>']
regex = re.compile(r'\d+')

if fnmatch.fnmatch(foldername, '*aspectratio*'):
  _, aspect, RaFe, alpha1000 = [int(x) for x in regex.findall(foldername)]
else:
  _, RaFe, alpha1000 = [int(x) for x in regex.findall(foldername)]
  aspect = 8

print(foldername,aspect,RaFe,alpha1000)

# User parameter
if fnmatch.fnmatch(foldername, '*restart*'): tsss = 1.0
elif fnmatch.fnmatch(foldername, '*aspectratio*'): tsss = 1.0
else: tsss = 0.6 # (assumed) steady state start time

#########################
####### BULK DATA #######
#########################
bulk = np.load(foldername+'bulk.npz')
time = bulk['time']
x = bulk['x']
z = bulk['z']
u = bulk['u'] 
w = bulk['w']
T = bulk['T']
print(time.shape,x.shape,z.shape,T.shape)

xv = x[:,0]
zv = z[0,:]

itsss = np.argmin(np.abs(time-tsss)); print('tsss is %1.2f'%tsss,'isss is %i'%itsss)
if itsss==len(time)-1: itsss=len(time)-2

#### ATTENTION: faire c2 puis Re = sqrt(average())
c = np.sqrt(u**2+w**2)
Re = xz_average(c,xv,zv)
T_xz_ave = xz_average(T,xv,zv)

dz = np.gradient(zv)
dx = np.gradient(xv)
iz05 = np.argmin(np.abs(zv-0.5))
iz09 = np.argmin(np.abs(zv-0.9))
utophalf = xz_average(u[:,:,iz05:],xv,zv[iz05:])

dTdz = np.gradient(T,axis=-1)/dz.reshape(1,1,len(zv))
dTdz_x_ave = x_average(dTdz,xv)
dTdz_t_ave = t_average(dTdz[itsss:],time[itsss:])
dTdx = np.gradient(T,axis=1)/dx.reshape(1,len(xv),1)
dTdx_z_ave = z_average(dTdx,zv)
dTdx_t_ave = t_average(dTdx[itsss:],time[itsss:])

gradT2 = dTdz**2+dTdx**2
gradT2_xz_ave = xz_average(gradT2,xv,zv)
fluxx = u*T-dTdx 
Tfluxx = tz_average(fluxx[itsss:],time[itsss:],zv)

u_x_ave = x_average(u,xv)
w_z_ave = z_average(w,zv)
psi = np.cumsum(u*dz.reshape(1,1,len(zv)),axis=-1)
psi_x_ave = x_average(psi,xv)
psi_z_ave = z_average(psi,zv)
psi_t_ave = t_average(psi[itsss:],time[itsss:])
ind_max = []; psi_max = []; pos_max = []
for i in range(len(time)):
  ind_max.append(np.unravel_index(np.argmax(np.squeeze(psi[i]), axis=None), np.squeeze(psi[i]).shape))
  psi_max.append(psi[i,ind_max[i][0],ind_max[i][1]])
  pos_max.append([xv[ind_max[i][0]],zv[ind_max[i][1]]])
ind_max = np.array(ind_max)
pos_max = np.array(pos_max)
psi_max = np.array(psi_max)
print('shape of maxs is : ',ind_max.shape,pos_max.shape,psi_max.shape)   
ind_t_ave_max = np.unravel_index(np.argmax(np.squeeze(psi_t_ave), axis=None), psi_t_ave.shape)
psi_t_ave_max = psi_t_ave[ind_t_ave_max[0],ind_t_ave_max[1]]
pos_t_ave_max = np.array([xv[ind_t_ave_max[0]],zv[ind_t_ave_max[1]]])
print('pos max is : ',pos_t_ave_max)

c05 = c[:,:,iz05]
u09 = u[:,:,iz09]
w05 = w[:,:,iz05]
T05 = T[:,:,iz05]
u05 = u[:,:,iz05]

u2summed = np.sum(u**2,axis=(1,2))/u.shape[1]/u.shape[2]
w2summed = np.sum(w**2,axis=(1,2))/w.shape[1]/w.shape[2]

u2mean = xz_average(u**2,xv,zv)
w2mean = xz_average(w**2,xv,zv)

u_t_ave = t_average(u[itsss:],time[itsss:])
w_t_ave = t_average(w[itsss:],time[itsss:])
T_t_ave = t_average(T[itsss:],time[itsss:])
c_t_ave = t_average(c[itsss:],time[itsss:])
u_z_ave = z_average(u,zv)

autocor_u = autocor(u,xv,mode="full")
autocor_u_same = autocor(u,xv,mode="same")
autocor_u_my, denom_my, lagnumber_my = myautocor(u,xv)
autocor_u05 = autocor_u[:,:,iz05] 
autocor_u09 = autocor_u[:,:,iz09] 
autocor_u09_same = autocor_u_same[:,:,iz09] 
autocor_u09_my = autocor_u_my[:,:,iz09] 
denom09_my = denom_my[:,:,iz09] 
autocor_u05_t_ave = t_average(autocor_u05[itsss:],time[itsss:])
autocor_u09_t_ave = t_average(autocor_u09[itsss:],time[itsss:])
autocor_u09_same_t_ave = t_average(autocor_u09_same[itsss:],time[itsss:])
autocor_u09_my_t_ave = t_average(autocor_u09_my[itsss:],time[itsss:])
autocor_u_z_ave = z_average(autocor_u,zv)
autocor_u_t_ave = t_average(autocor_u[itsss:],time[itsss:])
autocor_u_tz_ave = z_average(t_average(autocor_u[itsss:],time[itsss:]),zv)

iRuu09min = np.argmin(autocor_u09_t_ave)
if iRuu09min==len(autocor_u09_t_ave)-1:
  xRuu09max = 8
else:
  iRuu09max = np.argmax(autocor_u09_t_ave[iRuu09min:])
  xRuu09max = xv[iRuu09min+iRuu09max]-xv[0]

fx, psd_u = psdfun(u,xv,axis=1,window=True) 
fx, psd_w = psdfun(w,xv,axis=1,window=True)
psd_ec = 0.5*(psd_u+psd_w)
print(psd_u.shape,len(zv),len(zv)//2,len(xv)) 
Parseval_spec = np.squeeze(np.sum(psd_u,axis=1))
Parseval_phy = np.squeeze(np.mean(u**2,axis=1))
Parseval_diff = np.mean(np.abs(Parseval_spec-Parseval_phy))
print(Parseval_spec.shape,Parseval_phy.shape) 
print('Parseval : ',Parseval_diff,Parseval_spec[itsss,iz05],Parseval_phy[itsss,iz05])
fx = fx.reshape(1,len(fx),1) 
#print(fx.T)
#print(fx[:,1:].T)
#print(np.sum(psd_u[itsss:,1:],axis=1))
ilength_u = np.sum(psd_u[itsss:,1:,1:-1]/fx[:,1:],axis=1)/np.sum(psd_u[itsss:,1:,1:-1],axis=1)
ilength_ec = np.sum(psd_ec[itsss:,1:,1:-1]/fx[:,1:],axis=1)/np.sum(psd_ec[itsss:,1:,1:-1],axis=1)
ilength_u09 = ilength_u[:,iz09-1]
ilength_ec09 = ilength_ec[:,iz09-1]
ilength_u_z_ave = z_average(ilength_u,zv[1:-1])
ilength_u_t_ave = t_average(ilength_u,time[itsss:])
ilength_u09_t_ave = t_average(ilength_u09,time[itsss:])
ilength_ec_z_ave = z_average(ilength_ec,zv[1:-1])
ilength_ec_t_ave = t_average(ilength_ec,time[itsss:])
ilength_ec09_t_ave = t_average(ilength_ec09,time[itsss:])
ilength_u_tz_ave = tz_average(ilength_u,time[itsss:],zv[1:-1])
ilength_ec_tz_ave = tz_average(ilength_ec,time[itsss:],zv[1:-1])

###
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(411)
xx, zz = np.meshgrid(xv, zv)
im=ax.pcolormesh(xx,zz,dTdz_t_ave.T,shading='auto')
colorbar(im)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_title(r'$\langle \partial_z T\rangle_t$')
ax = fig.add_subplot(412)
xx, zz = np.meshgrid(xv, zv)
im=ax.pcolormesh(xx,zz,dTdx_t_ave.T,shading='auto')
colorbar(im)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_title(r'$\langle \partial_x T\rangle_t$')
ax = fig.add_subplot(413)
xx, zz = np.meshgrid(xv, zv)
dx2 = dx.reshape(len(x),1)
im=ax.pcolormesh(xx,zz,(np.cumsum(dTdx_t_ave*dx2,axis=0)/(np.cumsum(dx)).reshape(len(x),1)).T,shading='auto')
colorbar(im)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_title(r'$\langle T\rangle_t-T(x=-4)$')
ax = fig.add_subplot(414)
ax.plot(xv,z_average(dTdx_t_ave,zv))
ax.set_xlabel('x')
ax.set_ylabel(r'$\langle \partial_x T\rangle_{tz}$')
#
if fnmatch.fnmatch(foldername, '*restart*'): 
  figname = 'stratification-restart_%i_%i.png'%(RaFe,alpha1000)
elif fnmatch.fnmatch(foldername, '*aspectratio*'): 
  figname = 'stratification-aspect_%i_%i_%i.png'%(aspect,RaFe,alpha1000)
else: 
  figname = 'stratification-transient_%i_%i.png'%(RaFe,alpha1000)
plt.tight_layout(pad=1.2)
fig.savefig(figname)
plt.close()

###
cmap = sns.cubehelix_palette(as_cmap=True)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(311)
xx, zz = np.meshgrid(xv, zv)
im=ax.pcolormesh(xx,zz,psi_t_ave.T,shading='auto')
points = ax.scatter(pos_max[:,0],pos_max[:,1], c=time, s=50, cmap=cmap)
ax.plot(pos_t_ave_max[0],pos_t_ave_max[1],'*r',markersize=15)
colorbar(im)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_title(r'$\langle\psi\rangle_t$')
ax = fig.add_subplot(312)
tt, zz = np.meshgrid(time, zv)
im=ax.pcolormesh(tt,zz,psi_x_ave.T,shading='auto')
colorbar(im)
ax.set_xlabel('t')
ax.set_ylabel('z')
ax.set_title(r'$\langle\psi\rangle_x$')
ax = fig.add_subplot(313)
xx, tt = np.meshgrid(xv, time)
im=ax.pcolormesh(xx,tt,psi_z_ave,shading='auto')
colorbar(im)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_title(r'$\langle\psi\rangle_z$')
#
if fnmatch.fnmatch(foldername, '*restart*'): 
  figname = 'streamfunction-restart_%i_%i.png'%(RaFe,alpha1000)
elif fnmatch.fnmatch(foldername, '*aspectratio*'): 
  figname = 'streamfunction-aspect_%i_%i_%i.png'%(aspect,RaFe,alpha1000)
else: 
  figname = 'streamfunction-transient_%i_%i.png'%(RaFe,alpha1000)
plt.tight_layout(pad=1.2)
fig.savefig(figname)
plt.close()

###
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(331)
xx, tt = np.meshgrid(xv-xv.min(), time)
im=ax.pcolormesh(xx,tt,u05,shading='auto',vmin=u05.min(),vmax=-u05.min())
colorbar(im)
ax.set_xlabel('x')
ax.set_ylabel('time')
ax.set_title(r'$u(z=0.5)$')
ax.set_xlim((0,8))
ax = fig.add_subplot(332)
xx, tt = np.meshgrid(xv-xv.min(), time)
im=ax.pcolormesh(xx,tt,autocor_u05/(autocor_u05[:,0]).reshape(len(time),1),shading='auto')
colorbar(im)
ax.set_xlabel('x lag')
ax.set_ylabel('time')
ax.set_title(r'${R}_{uu}(z=0.5)$')
ax.set_xlim((0,8))
ax = fig.add_subplot(333)
im=ax.pcolormesh(xx,tt,autocor_u_z_ave/(autocor_u_z_ave[:,0]).reshape(len(time),1),shading='auto')
colorbar(im)
ax.set_ylabel('time')
ax.set_xlabel('x lag')
ax.set_title(r'$\langle{R}_{uu}\rangle_z$')
ax.set_xlim((0,8))
#
ax = fig.add_subplot(334)
xx, tt = np.meshgrid(xv-xv.min(), time)
im=ax.pcolormesh(xx,tt,u09,shading='auto',vmin=u09.min(),vmax=-u09.min())
colorbar(im)
ax.set_xlabel('x')
ax.set_ylabel('time')
ax.set_title(r'$u(z=0.9)$')
ax.set_xlim((0,8))
ax = fig.add_subplot(335)
xx, tt = np.meshgrid(xv-xv.min(), time)
im=ax.pcolormesh(xx,tt,autocor_u09/autocor_u09_t_ave[0],shading='auto')
colorbar(im)
ax.set_xlabel('x lag')
ax.set_ylabel('time')
ax.set_title(r'${R}_{uu}(z=0.9)$')
ax.set_xlim((0,8))
ax = fig.add_subplot(336)
ax.plot(xv-xv.min(),autocor_u05_t_ave/autocor_u05_t_ave[0],'--k',label=r'$\overline{R}_{cc}(z=0.5)$')
ax.plot(xv-xv.min(),autocor_u09_t_ave/autocor_u09_t_ave[0],'-k',label=r'$\overline{R}_{uu}(z=0.9)$')
ax.plot(xv-xv.min(),autocor_u09_same_t_ave/autocor_u09_same_t_ave[0],':g',label=r'same')
ax.plot(xv-xv.min(),autocor_u09_my_t_ave/autocor_u09_my_t_ave[0],':b',label=r'my')
ax.plot(xv-xv.min(),autocor_u_tz_ave/autocor_u_tz_ave[0],'-r',label=r'$\langle\overline{R}_{uu}\rangle_z$')
ax.axvline(xRuu09max)
ax.set_xlabel('time')
#ax.set_xlabel('x lag')
ax.legend()
ax.set_xlim((0,8))
#
ax = fig.add_subplot(337)
tt, zz = np.meshgrid(time[itsss:],zv[1:-1])
im=ax.pcolormesh(tt,zz,ilength_u.T,shading='auto')
colorbar(im)
ax.set_ylabel('z')
ax.set_xlabel('time')
ax.set_title(r'ilength_u=%1.2f'%ilength_u_tz_ave)
#
ax = fig.add_subplot(338)
tt, zz = np.meshgrid(time[itsss:],zv[1:-1])
im=ax.pcolormesh(tt,zz,ilength_ec.T,shading='auto')
colorbar(im)
ax.set_ylabel('z')
ax.set_xlabel('time')
ax.set_title(r'ilength_ec=%1.2f'%ilength_ec_tz_ave)
#
ax = fig.add_subplot(339)
ax.plot(t_average(ilength_u,time[itsss:]),zv[1:-1],'-k',label='ilength_u')
ax.plot(t_average(ilength_ec,time[itsss:]),zv[1:-1],'--b',label='ilength_ec')
ax.set_ylabel('z')
ax.set_xlabel('ilength')
ax.legend()
#
if fnmatch.fnmatch(foldername, '*restart*'): 
  figname = 'autocorrelation-restart_%i_%i.png'%(RaFe,alpha1000)
elif fnmatch.fnmatch(foldername, '*aspectratio*'): 
  figname = 'autocorrelation-aspect_%i_%i_%i.png'%(aspect,RaFe,alpha1000)
else: 
  figname = 'autocorrelation-transient_%i_%i.png'%(RaFe,alpha1000)
plt.tight_layout(pad=1.2)
fig.savefig(figname)
plt.close()
###

ulast = u[-1]; wlast = w[-1]; Tlast = T[-1]
u = []; w = []; T = []
bulk.close()

#########################
####### SIDE DATA #######
#########################
side = np.load(foldername+'side.npz')
time = side['time']
x3b = side['x3b']; xb = x3b[:,0]
x3t = side['x3t']; xt = x3t[:,0]
x3l = side['x3l']; xl = x3l[:,0]
x3r = side['x3r']; xr = x3r[:,0]
z3b = side['z3b']; zb = z3b[:,0]
z3t = side['z3t']; zt = z3t[:,0]
z3l = side['z3l']; zl = z3l[:,0]
z3r = side['z3r']; zr = z3r[:,0]
# temperature data 
T3b = side['T3b']
T3t = side['T3t']
T3l = side['T3l']
T3r = side['T3r']
# horizontal velocity data
u3b = side['u3b']
u3t = side['u3t']
u3l = side['u3l']
u3r = side['u3r']
# vertical velocity data
w3b = side['w3b']
w3t = side['w3t']
w3l = side['w3l']
w3r = side['w3r']

# Compute dz on top and bottom boundaries
dzt = z3t[0,0] - z3t[0,1]
dzb = z3b[0,1] - z3b[0,0]
# Compute dx on left and right  boundaries
dxl = x3l[0,1]-x3l[0,0]
dxr = x3r[0,0]-x3r[0,1]

# Computing dz(T) on the top, bottom, left and right boundaries
# check formulas here: https://lcn.people.uic.edu/classes/che205s17/docs/che205s17_reading_01e.pdf
dzTt = ( 3*T3t[:,:,0] - 4*T3t[:,:,1] + T3t[:,:,2]) / (2*dzt) # (backward) 2nd order accurate
dzTb = (-3*T3b[:,:,0] + 4*T3b[:,:,1] - T3b[:,:,2]) / (2*dzb) # (forward)
dzTl = (-3*T3l[:,:,0] + 4*T3l[:,:,1] - T3l[:,:,2]) / (2*dxl) # (forward) 2nd order accurate
#dTr = (T3r[:,:,0]-T3r[:,:,1]) / dxr # Try using 1st order
dzTr = ( 3*T3r[:,:,0] - 4*T3r[:,:,1] + T3r[:,:,2]) / (2*dxr) # (backward) 2nd order accurate
# Computing horizontal velocity gradients dz(v)
dzut = ( 3*u3t[:,:,0] - 4*u3t[:,:,1] + u3t[:,:,2]) / (2*dzt) # (backward) 2nd order accurate
dzub = (-3*u3b[:,:,0] + 4*u3b[:,:,1] - u3b[:,:,2]) / (2*dzb) # (forward)
dzul = (-3*u3l[:,:,0] + 4*u3l[:,:,1] - u3l[:,:,2]) / (2*dxl) # (forward) 2nd order accurate
dzur = ( 3*u3r[:,:,0] - 4*u3r[:,:,1] + u3r[:,:,2]) / (2*dxr) # (backward) 2nd order accurate
# Computing vertical velocity gradients dz(w)
dzwt = ( 3*w3t[:,:,0] - 4*w3t[:,:,1] + w3t[:,:,2]) / (2*dzt) # (backward) 2nd order accurate
dzwb = (-3*w3b[:,:,0] + 4*w3b[:,:,1] - w3b[:,:,2]) / (2*dzb) # (forward)
dzwl = (-3*w3l[:,:,0] + 4*w3l[:,:,1] - w3l[:,:,2]) / (2*dxl) # (forward) 2nd order accurate
dzwr = ( 3*w3r[:,:,0] - 4*w3r[:,:,1] + w3r[:,:,2]) / (2*dxr) # (backward) 2nd order accurate

side.close()

############################
####### HISTORY DATA #######
############################

if fnmatch.fnmatch(foldername, '*restart*'): 
  filename = 'processed-data-restart_%i_%i.npz'%(RaFe,alpha1000)
elif fnmatch.fnmatch(foldername, '*aspectratio*'): 
  filename = 'processed-data-aspect_%i_%i_%i.npz'%(aspect,RaFe,alpha1000)
else: 
  filename = 'processed-data-transient_%i_%i.npz'%(RaFe,alpha1000)
  
np.savez(filename, RaFe=RaFe, alpha1000=alpha1000, time=time, x=x, z=z, w2summed=w2summed, u2summed=u2summed, u2mean=u2mean, w2mean=w2mean, ulast=ulast, wlast=wlast, Tlast=Tlast, u_t_ave=u_t_ave, w_t_ave=w_t_ave, c_t_ave=c_t_ave, T_t_ave=T_t_ave, u_x_ave=u_x_ave, w_z_ave=w_z_ave, dTdz_x_ave=dTdz_x_ave, dTdx_z_ave=dTdx_z_ave, dTdz_t_ave=dTdz_t_ave, dTdx_t_ave=dTdx_t_ave, gradT2_xz_ave=gradT2_xz_ave, Tfluxx=Tfluxx, ind_max=ind_max, pos_max=pos_max, psi_max=psi_max, pos_t_ave_max=pos_t_ave_max, psi_t_ave_max=psi_t_ave_max, psi_x_ave=psi_x_ave, psi_z_ave=psi_z_ave, psi_t_ave=psi_t_ave, T_xz_ave=T_xz_ave, Re=Re, utophalf=utophalf, c05=c05, autocor_u05=autocor_u05, autocor_u09=autocor_u09, autocor_u09_same=autocor_u09_same, autocor_u09_my=autocor_u09_my, denom09_my=denom09_my, lagnumber_my=lagnumber_my, autocor_u_z_ave=autocor_u_z_ave, autocor_u_t_ave=autocor_u_t_ave, ilength_u09=ilength_u09, ilength_ec09=ilength_ec09, ilength_u09_t_ave=ilength_u09_t_ave, ilength_ec09_t_ave=ilength_ec09_t_ave, ilength_u_z_ave=ilength_u_z_ave, ilength_ec_z_ave=ilength_ec_z_ave, ilength_u_t_ave=ilength_u_t_ave, ilength_ec_t_ave=ilength_ec_t_ave, u_z_ave=u_z_ave, u05=u05, w05=w05, T05=T05, u09=u09, xb=xb, xt=xt, Tt=T3t[:,:,0], Tb=T3b[:,:,0], dzTt=dzTt, dzTb=dzTb, zl=zl, zr=zr, Tl=T3l[:,:,0], Tr=T3r[:,:,0]) 





