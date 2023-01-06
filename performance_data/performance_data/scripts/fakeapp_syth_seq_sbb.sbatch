#!/bin/bash

#SBATCH -J fakeapp_synth_sbb_seq
##SBATCH --time=#TIME#
#SBATCH -N 1 
#SBATCH --partition=equipment_compute_diskfull
##SBATCH --exclusive
##SBATCH --tasks-per-node=32
#SBATCH --ioi=yes 
#SBB Flavor=Huge Targets=/fsiof/phamtt/tmp/ datanodes=1 
export IOI_PROTECTED=1

hostname

PATH1=/fsiof/phamtt/tmp
PATH2=/fsiof/phamtt/tmp/write

srun -n 1 ./seq_job1.sh $PATH1 $PATH2

wait
