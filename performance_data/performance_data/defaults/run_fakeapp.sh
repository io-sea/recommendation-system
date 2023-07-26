#!/bin/bash

DIR_FILE=$1 # for read mode
if [ -z "$6" ] # empty string for write mode
then
    DIR_FILE=$1/$(hostname)
    mkdir -p $DIR_FILE
fi
echo $DIR_FILE
#./fakeapp-nompi -s 1 -d $DIR_WRITE -Z 10G -L $LEAD -W $SIZE -w $SIZE -S $SCATER -l 1M -N $OPS $MODE
/p/home/jusers/mimouni1/deep/recsys/iosea-wp3-recommandation-system/performance_data/performance_data/defaults/./fakeapp-nompi-nodata -s 1 -d $DIR_FILE -Z 10G -L $2 -W $3 -w $3 -S $4 -l $3 -N $5 $6
