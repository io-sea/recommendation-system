#!/bin/bash

#SBATCH -J fakeapp_synth_nfs_seq
##SBATCH --time=#TIME#
#SBATCH -N 2
#SBATCH --reservation=phamtt #on kea cluster
##SBATCH --partition=equipment_compute_diskfull #on kiwi cluster
##SBATCH --ioi=yes
#export IOI_PROTECTED=1

srun -N $NODES ./run_fakeapp.sh $TARGET $LEAD $SIZE $OPS $MODE
