#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec_path=${SCRIPT_DIR}/../build/bin/breakdown
io_size=(4 8 16 32 64 128 256 512 1024 2048 4096)

echo "exec_path: $exec_path"

for i in ${io_size[@]}
do
    echo "io_size: $i"
    ${exec_path} $1 $i
done
