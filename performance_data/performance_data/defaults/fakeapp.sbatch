#!/bin/bash

#SBATCH -J fakeapp_gen
#SBATCH -N 1
##SBATCH --reservation=salim #enable only on kea cluster
#SBATCH --partition=equipment_compute_diskfull #enable only on kiwi cluster
#$ACC Flavor=Medium Targets=$TARGET

############# ENABLE TO PROTECT SHOR JOB IN IOI ###############
#export IOI_PROTECTED=1

#use to run fakeapp in sequential
run_fakeapp ()
{
    srun --io-drop-cache=yes -N $NODES /home_nfs/phamtt/IO-SEA/iosea-wp3-recommandation-system/performance_data/performance_data/defaults/./run_fakeapp.sh $TARGET $LEAD_R $SIZE_R $SCATTER_R $OPS_R -R
    srun --io-drop-cache=yes -N $NODES /home_nfs/phamtt/IO-SEA/iosea-wp3-recommandation-system/performance_data/performance_data/defaults/./run_fakeapp.sh $TARGET $LEAD_W $SIZE_W $SCATTER_W $OPS_W 
}

#use to run fakeapp in parallel
run_fakeapp_parallel ()
{
    srun --io-drop-cache=yes -N $NODES /home_nfs/phamtt/IO-SEA/iosea-wp3-recommandation-system/performance_data/performance_data/defaults/./run_fakeapp.sh $TARGET $LEAD_R $SIZE_R $SCATTER_R $OPS_R -R &
    srun --io-drop-cache=yes -N $NODES /home_nfs/phamtt/IO-SEA/iosea-wp3-recommandation-system/performance_data/performance_data/defaults/./run_fakeapp.sh $TARGET $LEAD_W $SIZE_W $SCATTER_W $OPS_W &
    wait
}

time (run_fakeapp_parallel)
