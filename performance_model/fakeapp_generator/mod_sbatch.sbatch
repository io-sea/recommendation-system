#!/bin/bash

#SBATCH -J fakeapp_gen
#SBATCH -N 4
##SBATCH --reservation=phamtt #enable only on kea cluster
#SBATCH --partition=equipment_compute_diskfull #enable only on kiwi cluster
#$ACC Flavor=Medium Targets=/fsiof/phamtt/tmp

############# ENABLE TO USE IOI ###############
#export IOI_PROTECTED=1

time(srun --io-drop-cache=yes -N 1 ./run_fakeapp.sh /fsiof/phamtt/tmp 1 10000 1000000 100000 )
