
import numpy as np
import matplotlib.pyplot as plt
import fnmatch

foldername = './'
print(foldername)

def t_average(var,time):
   dt = np.gradient(time).reshape(len(time),1,1)
   var_tave = np.sum(var*dt,axis=0)/np.sum(dt,axis=0)
   return var_tave 

# User parameter
tsss = 0.05 # (assumed) steady state start time (set to 0.05 instead of 0.5 to accomodate for short run)

#########################
####### SIDE DATA #######
#########################

# NOTE: organized by sweeping in x to the right and moving away from bot, then top boundary, and then sweeping vertically upward and moving away from left and then right boundary)

# Import boundary data
pside = np.loadtxt(foldername+'probes_side.dat')
dside = np.loadtxt(foldername+'data_side.dat')

xside = pside[:,0]; numprobes = xside.size # total number of probes in SIDE data
zside = pside[:,1]
time = dside[:,0]; time = np.unique(time); nt = len(time)
#
itsss = np.argmin(np.abs(time-tsss)); print('tsss is %1.2f'%tsss,'isss is %i'%itsss)
if itsss==len(time)-1: itsss=len(time)-2
#
uside = dside[:,1]; fullnt = np.floor(len(uside)/numprobes)
print(nt,numprobes*nt,len(uside),len(dside[:numprobes*nt,1]),len(uside)/numprobes,nt)
flag = 0
if fullnt != nt: 
  print('final time step not fully saved') 
  time = time[:-1]; nt = nt-1
  itsss -= 1 
  flag = 1
uside = dside[:numprobes*nt,1] 
wside = dside[:numprobes*nt,2]
Tside = dside[:numprobes*nt,4]
#print(len(time),nt,uside.size,uside.size/numprobes) 
#plt.plot(xside,zside,'.');plt.show()

# Separate data block on horizontal lines from block on vertical lines, reshape and split on all boundaries 
diff = xside[1:]-xside[:-1] # this will be positive as long as we consider data on horizontal lines
isplit = np.argmin(diff>=0)+1 # caution that diff is a reduced array skipping the first entry
#print(diff); print(isplit)
#
xhor = xside[:isplit].reshape((xside[:isplit].size//6),6) # first block corresponds to horizontal lines, then reshape
zhor = zside[:isplit].reshape((xside[:isplit].size//6),6)
xver = xside[isplit:].reshape((xside[isplit:].size//6),6) # second block corresponds to vertical lines, then reshape
zver = zside[isplit:].reshape((xside[isplit:].size//6),6)
#
x3b = xhor[:,:3] # bottom
z3b = zhor[:,:3]
x3t = xhor[:,3:] # top
z3t = zhor[:,3:]
#
x3l = xver[:,:3] # left
z3l = zver[:,:3]
x3r = xver[:,3:] # right
z3r = zver[:,3:]
#
#print(xside);print(xhor);print(zhor);print(xver);print(zver)

# Reshape and separate horizontal from vertical blocks of variables, reshape and split on all boundaries 
uuside = uside.reshape(nt,numprobes)
wwside = wside.reshape(nt,numprobes)
TTside = Tside.reshape(nt,numprobes)
#
uuhor = uuside[:,:isplit].reshape(nt,xhor.shape[0],6)
wwhor = wwside[:,:isplit].reshape(nt,xhor.shape[0],6)
TThor = TTside[:,:isplit].reshape(nt,xhor.shape[0],6)
#
uuver = uuside[:,isplit:].reshape(nt,xver.shape[0],6)
wwver = wwside[:,isplit:].reshape(nt,xver.shape[0],6)
TTver = TTside[:,isplit:].reshape(nt,xver.shape[0],6)
#
u3b = uuhor[:,:,:3] 
w3b = wwhor[:,:,:3]
T3b = TThor[:,:,:3]
#
u3t = uuhor[:,:,3:] 
w3t = wwhor[:,:,3:]
T3t = TThor[:,:,3:]
#
u3l = uuver[:,:,:3] 
w3l = wwver[:,:,:3]
T3l = TTver[:,:,:3]
#
u3r = uuver[:,:,3:] 
w3r = wwver[:,:,3:]
T3r = TTver[:,:,3:]
#
#print(x3b,z3b)
# SAVE DATA
np.savez(foldername+'side.npz', time=time, x3b=x3b,z3b=z3b,u3b=u3b,w3b=w3b,T3b=T3b,
			   	 x3t=x3t,z3t=z3t,u3t=u3t,w3t=w3t,T3t=T3t,
			   	 x3l=x3l,z3l=z3l,u3l=u3l,w3l=w3l,T3l=T3l,
			   	 x3r=x3r,z3r=z3r,u3r=u3r,w3r=w3r,T3r=T3r)
				 
# Time averages
u3bta = t_average(u3b[itsss:],time[itsss:])  
w3bta = t_average(w3b[itsss:],time[itsss:]) 
T3bta = t_average(T3b[itsss:],time[itsss:]) 
#
u3tta = t_average(u3t[itsss:],time[itsss:]) 
w3tta = t_average(w3t[itsss:],time[itsss:]) 
T3tta = t_average(T3t[itsss:],time[itsss:]) 
#
u3lta = t_average(u3l[itsss:],time[itsss:]) 
w3lta = t_average(w3l[itsss:],time[itsss:]) 
T3lta = t_average(T3l[itsss:],time[itsss:]) 
#
u3rta = t_average(u3r[itsss:],time[itsss:]) 
w3rta = t_average(w3r[itsss:],time[itsss:]) 
T3rta = t_average(T3r[itsss:],time[itsss:]) 

np.savez(foldername+'side_t_ave.npz', time=time, x3b=x3b,z3b=z3b,u3b=u3bta,w3b=w3bta,T3b=T3bta,
			         	x3t=x3t,z3t=z3t,u3t=u3tta,w3t=w3tta,T3t=T3tta,
			         	x3l=x3l,z3l=z3l,u3l=u3lta,w3l=w3lta,T3l=T3lta,
			         	x3r=x3r,z3r=z3r,u3r=u3rta,w3r=w3rta,T3r=T3rta)

#########################
####### BULK DATA #######
#########################

# Import bulk data
pbulk = np.loadtxt(foldername+'probes_bulk.dat')
dbulk = np.loadtxt(foldername+'data_bulk.dat')
time = np.loadtxt(foldername+'time.dat'); nt = len(time); print(nt)

# Data in the bulk
xbulk = pbulk[:,0]; nxbulk = len(np.unique(xbulk))
zbulk = pbulk[:,1]; nzbulk = len(np.unique(zbulk))
ubulk = dbulk[:,1]
wbulk = dbulk[:,2]
Tbulk = dbulk[:,4]
#print(ubulk.shape,nt,nxbulk,nzbulk,nt*nxbulk*nzbulk)
x = xbulk.reshape((nxbulk,nzbulk))
z = zbulk.reshape((nxbulk,nzbulk))
u = ubulk.reshape((nt,nxbulk,nzbulk))
w = wbulk.reshape((nt,nxbulk,nzbulk))
T = Tbulk.reshape((nt,nxbulk,nzbulk))

if flag==1:
  time = time[:-1]
  u = u[:-1]
  w = w[:-1]
  T = T[:-1]
  itsss -= 1

np.savez(foldername+'bulk.npz', time=time, x=x, z=z, u=u, w=w, T=T)

# Time averages
itsss = np.argmin(np.abs(time-tsss)); print('tsss is %1.2f'%tsss,'isss is %i'%itsss)
if itsss==len(time)-1: itsss=len(time)-2
uta = t_average(u[itsss:],time[itsss:]) # could use, alternatively, np.mean() if uniform saving time steps
wta = t_average(w[itsss:],time[itsss:])
Tta = t_average(T[itsss:],time[itsss:])

np.savez(foldername+'bulk_t_ave.npz', time=time, x=x, z=z, u=uta, w=wta, T=Tta)












