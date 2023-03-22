To compile to program:

1. Load module PAPI with
module load papi

2. Make with
make

==========================================================================================
To run it (example job192xy):
sbatch job192xy.scp

==========================================================================================

Example jobscript to run the code on batch node on 4 nodes 192 cores, with 12 processes in x direction, 16 in y direction (also included as job192xy.scp):

#!/bin/bash

#SBATCH -J heat

#SBATCH -o job192xy.out
#SBATCH -e job192xy.out

#SBATCH --time=00:01:00
#SBATCH --account=h039v
#SBATCH --partition=test
#Number of nodes and MPI tasks per node:
#SBATCH --nodes=4
#SBATCH --ntasks=192
#SBATCH --ntasks-per-node=48
 
#Important
module load papi
module load slurm_setup
 
#Run the program:
mpiexec -n $SLURM_NTASKS ./heat test.dat -x 12 -y 16

============================================================================================
To Run on local machine (untested):
mpiexec -n 6 ./heat test.dat -x 3 -y 2

============================================================================================
To Run with arbitrary resolution and iteration please edit the test.dat before compiling
