#!/bin/bash

io_size=(4 16 32 64 256 1024 4096 16384 65536 262144)

type=$1
once=$2

for i in ${io_size[@]}
do
    echo "io_size: $i"
    /mnt/phxfs/benchmarks/loop_performance $type $i $once
done
