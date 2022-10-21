#!/bin/bash

#SBATCH -J fakeapp_gen
#SBATCH -N 4
##SBATCH --reservation=phamtt #enable only on kea cluster
#SBATCH --partition=equipment_compute_diskfull #enable only on kiwi cluster

############# ENABLE TO USE IOI ###############
##SBATCH --ioi=$IOI
#export IOI_PROTECTED=1

time(srun -N $NODES ./run_fakeapp.sh $TARGET $LEAD $SIZE $OPS $MODE)
