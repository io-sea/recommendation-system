#!/bin/bash

#SBATCH -J fakeapp_synth_nfs_stride
##SBATCH --time=#TIME#
#SBATCH -N 1 
#SBATCH --partition=equipment_compute_diskfull
##SBATCH --exclusive
##SBATCH --tasks-per-node=32
#SBATCH --ioi=yes 

export IOI_PROTECTED=1

hostname

PATH1=/scratch/phamtt/tmp
PATH2=/scratch/phamtt/tmp/write
#PATH1=/dev/zero
#PATH2=/dev/null

srun -n 1 ./stride_job1.sh $PATH1 $PATH2

wait
