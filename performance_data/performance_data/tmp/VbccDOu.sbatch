#!/bin/bash

#SBATCH -J fakeapp_gen
#SBATCH -N 1
##SBATCH --reservation=salim #enable only on kea cluster
#SBATCH --partition=equipment_compute_diskfull #enable only on kiwi cluster
#$ACC Flavor=Medium Targets=/scratch/mimounis/tmp

############# ENABLE TO USE IOI ###############
#export IOI_PROTECTED=1

time(srun --exclusive --io-drop-cache=yes -N 1 /home_nfs/mimounis/iosea-wp3-recommandation-system/performance_data/performance_data/defaults/./run_fakeapp.sh /scratch/mimounis/tmp 1 1044400.0876494024 0 139 -R)
