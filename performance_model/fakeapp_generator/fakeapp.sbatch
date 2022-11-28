#!/bin/bash

#SBATCH -J fakeapp_gen
#SBATCH -N 4
##SBATCH --reservation=phamtt #enable only on kea cluster
#SBATCH --partition=equipment_compute_diskfull #enable only on kiwi cluster
#$ACC Flavor=Medium Targets=$TARGET

############# ENABLE TO USE IOI SHORT JOB PROTECTION ###############
##export IOI_PROTECTED=1

time(srun --io-drop-cache=yes -N $NODES ./run_fakeapp.sh $TARGET $LEAD $SIZE $SCATTER $OPS $MODE)
