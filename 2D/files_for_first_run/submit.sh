#!/bin/bash
### SGE variables
### job shell
#$ -S /bin/bash
### job name (your choice)
#$ -N e6a01
### queue name
### -q E5-2697Av4deb256 
### -q E5-2667v2deb128
#$ -q E5-2670deb128F
### -q h48-E5-2670deb128 
### -q h48-CLG6226Rdeb192 
### parallel environment & nslots (use correct -pe -q pair)
#$ -pe mpi16_debian 16
### load user env onto SGE
#$ -cwd
### export env variables to all compute nodes
#$ -V
### mails at begining and end of job
### -m b
### -m e

### Other queues
### -q E5_test (use with -pe test_debian X) ### Max time = 5mins
### -q h48-E5-2670deb128 (use with -pe mpi16_debian X) ### Max time = 48h
### -q h48-CLG6226Rdeb192 (use with -pe mpi32_debian (ALSO mpi16) X) - Lake ### Max time = 48h ### BEST
### -q h6-E5-2667v4deb128 (use with -pe mpi16_debian X) # Max time = 6h
### -q CLG5218deb192D (use with -pe  mpi16) X) - Lake ### Max time = 168h  
### -q h48-CLG6226Rdeb192

### go to local dir; otherwise SGE goes to ~/
echo "${SGE_O_WORKDIR}"
cd "${SGE_O_WORKDIR}" || { echo "cannot cd to ${SGE_O_WORKDIR}"; exit 1; }

# init env (should be in ~/.profile)
source /usr/share/lmod/lmod/init/bash

### environment modules
module load GCC/7.2.0
module load GCC/7.2.0/OpenMPI/3.0.0
module load python/3.8.3

### below is required for E5_test or when using multiple nodes
export HOSTFILE=${TMPDIR}/machines

### program run
makenek hc
nekmpi hc 16
visnek hc
