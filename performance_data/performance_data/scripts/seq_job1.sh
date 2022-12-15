#!/bin/bash
sleep_random (){
max=60
seed=$RANDOM
min=10
random_time_to_sleep=$(( seed % max + min ))
sleep $random_time_to_sleep
}
PATH1=$1
PATH2=$2
FA=./fakeapp-nompi-nodata

echo "load params start"
sleep 10
ioi-event -m "start load param1"
$FA -s 1 -d $PATH1 -R -Z 10G -L 1 -W 4K -w 4K -S 0 -l 10K -N 100000
ioi-event -m "end load param1"
sleep 60
ioi-event -m "start load param2"
$FA -s 1 -d $PATH1 -R -Z 10G -L 1 -W 32K -w 32K -S 0 -l 10K -N 100000
ioi-event -m "end param2"
sleep 60
ioi-event -m "start load param3"
$FA -s 1 -d $PATH1 -R -Z 10G -L 1 -W 128K -w 128K -S 0 -l 10K -N 100000
ioi-event -m "end param3"
sleep 60
ioi-event -m "start checkpoint 1"
$FA -s 1 -d $PATH2 -Z 10G -L 1 -W 4K -w 4K -S 0 -l 1M -N 100000
ioi-event -m "end checkpoint 1"
sleep 60
ioi-event -m "start checkpoint 2"
$FA -s 1 -d $PATH2 -Z 10G -L 1 -W 32K -w 32K -S 0 -l 1M -N 100000
ioi-event -m "end checkpoint 2"
sleep 60
ioi-event -m "start checkpoint 3"
$FA -s 1 -d $PATH2 -Z 10G -L 1 -W 128K -w 128K -S 0 -l 1M -N 100000
ioi-event -m "end checkpoint 3"
sleep 10

