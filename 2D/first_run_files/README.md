# Running a simulation from scratch

Running a simulation from scratch
0. Go to $home/nek5000/tools/ and run “maketools all”
1. Taylor initial.rea and hc.usr
2. Taylor hc.box (using initial.rea as input)
3. Choose a number of #cores to run on
4. Taylor SIZE, i.e., set spectral resolution (lx1 and lxd=3/2*lx1; typically 8 and 12), set lelg, and ensure that #elements/core lelt = lelg/lp as small as possible, i.e., with lelg = #elements and lp = #cores (min considered); set lhis as small as possible but such that lhis*#cores>#pts in hc.his
5. Run “genbox”, enter “hc.box” when prompted
6. Rename box.rea as hc.rea
7. Run “genmap”, enter “hc” when prompted, press “enter” when prompted for the mesh tolerance. This will generate hc.map, which is used by the parallelization algorithm
8. Run “makenek hc”
9. Run “nekmpi hc X” with X=#cores
10. Run “visnek hc” to prepare hc.fld files for visualization in VisIt

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
