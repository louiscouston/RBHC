# Instructions

## Running your first simulation

The files available in this folder currently allow reproducing the run Ra_F=1e6, Lambda=0.1 from the paper, albeit with a shorter simulation time, such that the run shouldn't take longer than a few minutes with the pre-set choice of 8 cores (last checked: 30/08/2022 on PSMN by Louis). 

If you're running on ENS de Lyon supercomputer (PSMN), make sure that you first installed Nek5000 and loaded the required libraries. While on a compute node, I do this running the alias "nek5000", which is set in my .bashrc as "alias nek5000="module load GCC/7.2.0/OpenMPI/3.0.0 GCC/7.2.0"". Note that you may have to go to $home/nek5000/tools/ and run “maketools all”.

1. Generate hc.his, i.e., the Eulerian grid (fixed probes) that will be used to evaluate and save dependent variables (t,u,w,p,T)
- taylor genhpts.f90 if you must (pick nxhis, nzhis; nhis=nxhis.nzhis+(nxhis+nzhis).3.2; .=times)  
- run "gfortran genhpts.f90"  
- run "./a.out"  
- run "mv fort.66 hc.his"  

2. Generate hc.rea and hc.map
- taylor hc.box if you must (pick nx, nz)  
- run “genbox”, enter “hc.box” when prompted  
- rename box.rea as hc.rea  
- run “genmap”, enter “hc” when prompted, press “enter” when prompted for the mesh tolerance.

3. Taylor SIZE based on the choices made above
- pick the spectral order lx1, lxd (=3/2lx1)
- pick the number of processors lp (the code will also work with more cores but not fewer)
- change total number of elements lelg, number elements per core lelt (=lelg/lp), number of probes per core lhis (=nhis/lp) (always using upper integer numbers)

4. Taylor hc.usr, where the physics happen  
- modify the control parameters if you must, i.e. rayl (Ra_F in the paper), hg (Lambda), A (Gamma)  
- modify the routine userchk if you must, which sets variables stored in fort.51  
- modify the block "if(mod(istep,1000) .eq. 0) then; call hpts(); endif" if you want to adjust the number of iterations between two different writes of primitive variables in hc.his  
- modify the routine userbc if you don't like our sinusoidal temperature forcing along the top boundary  
- modify useric if you want to change the initial condition

5. Compile your code and run!  
- run “makenek hc”  
- run “nekmpi hc X” with X=#cores  
- run “visnek hc” to prepare hc.fldxxxx files for visualization in VisIt (frequence set by iotime in hc.rea)  
- open VisIt and load the hc.nek5000 file

6. Analyze the results  
- run "python3 postprocess-fort.py", which post-processes the volume-averaged quantities computed on the fly via the hc.usr routine (creating figure fort.png)  
- run "./clean.sh", which breaks down hc.his in reduced .dat files  
- run "python3 generate-data.py", which post-processes the .dat files into .npz files (including reduced files with time averages beyond tsss only)  
- run "python3 plot-figures.py", which creates figures (snapshots) from bulk.npz  
- fancier analysis (as reported in the paper) can be run adjusting the python scripts stored in the folder all_analysis_scripts (nb: they only work using the folder organization I used personally... and some variables have been renamed... so use with caution!)  


## Changing a handful of parameters after having run your first simulation

- If you want to change the number of elements in hc.box (besides other things), then you have to follow all steps listed above
- If you are happy with the number of elements

Running a simulation using pre-compiled files  
Use hc.rea as input in hc.box.  
Option 1: You only change hc.rea and/or hc.usr. Do step 1, then go directly to step 8.  
Option 2: You also change #cores and/or spectral resolution. Do step 1, then go directly to step 4.  
Option 3: You also change #elements in hc.box. Do all steps again.
	
