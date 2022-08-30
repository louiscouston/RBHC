# Running a simulation from scratch

The files available in this folder currently allow reproducing the run Ra_F=1e6, Lambda=0.1 from the paper, albeit with a shorter simulation time, such that the run shouldn't take longer than a few minutes with the pre-set choice of 8 cores (last checked: 30/08/2022 on PSMN). 

If you're running on ENS de Lyon supercomputer (PSMN), make sure that you first installed Nek5000 and loaded the required libraries. While on a compute node, I do this running the alias "nek5000", which is set in my .bashrc as "alias nek5000="module load GCC/7.2.0/OpenMPI/3.0.0 GCC/7.2.0"". Note that you may have to go to $home/nek5000/tools/ and run “maketools all”.

1. Generate hc.his, i.e., the Eulerian grid (fixed probes) that will be used to evaluate and save dependent variables (t,u,w,p,T)   
-> taylor genhpts.f90 if you must (pick nxhis, nzhis; nhis=nxhis.nzhis+(nxhis+nzhis).3.2; .=times)
-> run "gfortran genhpts.f90"
-> run "./a.out"
-> run "mv fort.66 hc.his"

2. Generate hc.rea and hc.map
-> taylor hc.box if you must (pick nx, nz)
-> run “genbox”, enter “hc.box” when prompted
-> rename box.rea as hc.rea
-> run “genmap”, enter “hc” when prompted, press “enter” when prompted for the mesh tolerance.

3. Taylor SIZE based on the choices made above
-> pick the spectral order lx1, lxd (=3/2lx1)
-> pick the number of processors lp (the code will also work with more cores but not fewer)
-> change total number of elements lelg, number elements per core lelt (=lelg/lp), number of probes per core lhis (=nhis/lp) (always using upper integer numbers)

4. Taylor hc.usr, where the physics happen
-> modify the control parameters if you must, i.e. rayl (Ra_F in the paper), hg (Lambda), A (Gamma)
-> modify the routine userchk if you must, which sets variables stored in fort.51
-> modify the block "if(mod(istep,1000) .eq. 0) then; call hpts(); endif" if you want to adjust the number of iterations between two different writes of primitive variables in hc.his
-> modify the routine userbc if you don't like our sinusoidal temperature forcing along the top boundary
-> modify useric if you want to change the initial condition

5. Compile your code and run!
-> run “makenek hc”
-> run “nekmpi hc X” with X=#cores
-> run “visnek hc” to prepare hc.fldxxxx files for visualization in VisIt

6. Analyze the results
-> run "fort.py", which post-process the volume-averaged quantities computed on the fly via the hc.usr routine 



Running a simulation using pre-compiled files
Use hc.rea as input in hc.box.
Option 1: You only change hc.rea and/or hc.usr. Do step 1, then go directly to step 8.
Option 2: You also change #cores and/or spectral resolution. Do step 1, then go directly to step 4.
Option 3: You also change #elements in hc.box. Do all steps again.
	
Output
Option 1: Userchck routine in hc.usr file defines volume averaged variables that are computed and stored at every time step in successive columns in fort.51. The first column shows time.
Option 2: Variables are saved on the collocation grid used for computations in hc.fld00 files, which are written every time the simulation advances by iotime, which is a parameter that must be set in hc.rea. These files can be used for visualization in VisIt. To do so, run “visnek hc”, open VisIt and load the hc.nek5000 file.
Option 3: Primitive variables (u,w,T) can be saved on a custom grid as defined in genhpts.f90. To generate the grid, compile genhpts.f90 using “gfortran genhpts.f90” and run the obtained a.out executable using “./a.out”. This will create file fort.66, which should be renamed as hc.his. As the simulation proceeds, the data is ordered as (t,x,z,u,w,T) and appended to hc.his every time the simulation advances by iohis.

Analysis
Currently, the script fort.py can be used to process the fort.51 data; the script clean.sh can be used to split the hc.his data into bulk and side data by running “./clean.sh”; the data_bulk.dat and data_side.dat can then be further analysed using genDAT.py and sideANALYSIS.py.
