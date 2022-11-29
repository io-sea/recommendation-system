#!/bin/bash

#SBATCH -J fakeapp_gen
#SBATCH -N 1
##SBATCH --reservation=salim #enable only on kea cluster
#SBATCH --partition=equipment_compute_diskfull #enable only on kiwi cluster
#$ACC Flavor=Medium Targets=$TARGET

############# ENABLE TO USE IOI ###############
#export IOI_PROTECTED=1

time(srun --exclusive --io-drop-cache=yes -N $NODES ./run_fakeapp.sh $TARGET $LEAD $SIZE $SCATTER $OPS $MODE)
