#!/bin/bash

#SBATCH -J fakeapp_gen
#SBATCH -N 1
##SBATCH --reservation=salim #enable only on kea cluster
#SBATCH --partition=equipment_compute_diskfull #enable only on kiwi cluster
#SBB Flavor=Medium Targets=/fsiof/mimounis/tmp

############# ENABLE TO USE IOI ###############
#export IOI_PROTECTED=1

time(srun --exclusive --io-drop-cache=yes -N 1 ./run_fakeapp.sh /fsiof/mimounis/tmp 1 10485760.0 0 1095 )
