import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Functions for post-processing

colvec = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

def createmap(n,cmap):
  mapvec = []
  for j in range(0,n):
    idd = np.rint(j*256/(n-1))
    idd = idd.astype(int)
    mapvec.append(cmap(idd))
  return mapvec

def colorbar(mappable,size="5%"):
  ax = mappable.axes
  fig = ax.figure
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size=size, pad=0.05)
  return fig.colorbar(mappable, cax=cax)

def myautocor(var,x,axis=1): # assumes 3d array
  autocor = np.zeros_like(var)
  denominator = np.zeros_like(var)
  lagnumber = np.zeros(var.shape[1])
  for lag in range(var.shape[1]):
    for k in range(var.shape[1]-lag):
      autocor[:,lag,:] += var[:,k+lag,:]*var[:,k,:]
      denominator[:,lag,:] += var[:,k,:]*var[:,k,:]
      lagnumber[lag] += 1
  return autocor, denominator, lagnumber

def autocor(var,x,axis=1,mode="full"): # assumes 3d array, nb: valid can't work with same size array
  autocor = np.zeros_like(var)
  for i in range(var.shape[0]):
    for j in range(var.shape[-1]):
      autocor[i,:,j] = np.correlate(var[i,:,j],var[i,:,j],mode=mode)[var.shape[1]-1:]
  return autocor

def psdfun(var,x,axis=1,window=True): # assumes array is 3d
  windowf = 0.5 - 0.5*np.cos(2*np.pi*x/(x[-1]-x[0]))	
  if var.ndim==2 and axis==1: 
    windowf=windowf.reshape(1,len(x))
  elif var.ndim==2 and axis==0: 
    windowf=windowf.reshape(len(x),1)
  elif var.ndim==3: 
    windowf = windowf.reshape(1,len(x),1)   
  if window==True: spec = np.fft.rfft(var*windowf,axis=axis)/len(x)
  else: spec = np.fft.rfft(var,axis=axis)/len(x)
  psd = np.abs(spec)**2
  if x.size % 2 == 1:
    if axis==1: psd[:,1:] *= 2
    elif axis==0: psd[1:] *= 2
  else: 
    if axis==1: psd[:,1:-1] *= 2
    elif axis==0: psd[1:-1] *= 2
  fx = np.fft.rfftfreq(x.size,x[1]-x[0])
  return fx, psd

def t_average(var,t): 
  dt = np.gradient(t)
  if var.ndim==2: dt=dt.reshape(len(t),1)
  elif var.ndim==3: dt=dt.reshape(len(t),1,1)
  var_tave = np.sum(var*dt,axis=0)/np.sum(dt,axis=0)
  return var_tave
  
def x_average(var,x): 
  dx = np.gradient(x)
  if var.ndim==1:
    var_xave = np.sum(var*dx)/np.sum(dx)
  else:
    if var.ndim==3: 
      dx=dx.reshape(1,len(x),1)
      var_xave = np.sum(var*dx,axis=1)/np.sum(dx,axis=1)
    elif var.ndim==2:
      if len(x)==var.shape[1]: # (t,x)
        dx=dx.reshape(1,len(x))
        var_xave = np.sum(var*dx,axis=1)/np.sum(dx,axis=1)
      else:  # (x,z)
        dx=dx.reshape(len(x),1)
        var_xave = np.sum(var*dx,axis=0)/np.sum(dx,axis=0)
  return var_xave

def z_average(var,z): 
  dz = np.gradient(z)
  if var.ndim==2: dz=dz.reshape(1,len(z))
  elif var.ndim==3: dz=dz.reshape(1,1,len(z))
  var_zave = np.sum(var*dz,axis=-1)/np.sum(dz,axis=-1)
  return var_zave
  
def xz_average(var,x,z): # assumes 2d array
  dx = np.gradient(x)
  dz = np.gradient(z)
  if var.ndim==2:
    ds = dx.reshape(len(x),1)*dz.reshape(1,len(z))
    var_xzave = np.sum(var*ds)/np.sum(ds)
  else:
    ds = dx.reshape(1,len(x),1)*dz.reshape(1,1,len(z))
    var_xzave = np.sum(var*ds,axis=(1,2))/np.sum(ds,axis=(1,2))
  return var_xzave
      
def tz_average(var,t,z): # assumes 2d array
  dt = np.gradient(t)
  dz = np.gradient(z)
  if var.ndim==2:
    ds = dt.reshape(len(t),1)*dz.reshape(1,len(z))
    var_tzave = np.sum(var*ds)/np.sum(ds)
  else:
    ds = dt.reshape(len(t),1,1)*dz.reshape(1,1,len(z))
    var_tzave = np.sum(var*ds,axis=(0,2))/np.sum(ds,axis=(0,2))
  return var_tzave
  
def tx_average(var,t,x): # assumes 2d array
  dt = np.gradient(t)
  dx = np.gradient(x)
  if var.ndim==2:
    ds = dt.reshape(len(t),1)*dx.reshape(1,len(x))
    var_txave = np.sum(var*ds)/np.sum(ds)
  else:
    ds = dt.reshape(len(t),1,1)*dx.reshape(1,len(x),1)
    var_txave = np.sum(var*ds,axis=(0,1))/np.sum(ds,axis=(0,1))
  return var_txave

lc = 12  
my_colors = [[],[],[],[],[]]
my_colors[0] = createmap(lc,cm.Blues_r)
my_colors[1] = createmap(lc,cm.Oranges_r)
my_colors[2] = createmap(lc,cm.Greens_r)
my_colors[3] = createmap(lc,cm.Reds_r)
my_colors[4] = createmap(lc,cm.Purples_r)

