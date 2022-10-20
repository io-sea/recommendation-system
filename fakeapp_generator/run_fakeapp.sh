#!/bin/bash
DIR_FILE=$1 # for read mode
if [ -z "$5" ] # empty string for write mode
then
    DIR_FILE=$1/$(hostname)
    mkdir -p $DIR_FILE
fi
echo $DIR_FILE
#./fakeapp-nompi -s 1 -d $DIR_WRITE $MODE -Z 10G -L $LEAD -W $SIZE -w $SIZE -S 0 -l 1M -N $OPS
./fakeapp-nompi -s 1 -d $DIR_FILE -Z 10G -L $2 -W $3 -w $3 -S 0 -l 1M -N $4 $5 #> $DIR_WRITE/log