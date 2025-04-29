#!/bin/bash

trace_text=(
    "paper_assit.txt"
    "gsm100.txt"
    "quality.txt"
    "sharegpt-sample-200.txt"
)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
traces=${SCRIPT_DIR}/../benchmarks/kvcache/traces
exec_path=${SCRIPT_DIR}/../build/bin/kvcache
block_file="/mnt/phxfs/kvcache.bin"
block_size=(8192 16384 65536)
test_type=(0 1)

for i in ${!trace_text[@]}; do
    for j in ${!test_type[@]}; do
        for k in ${!block_size[@]}; do
            echo "Running kvcache with ${traces}/${trace_text[$i]} with block size ${block_size[$k]}" >> "kvcache.log"
            ${exec_path} "${traces}/${trace_text[$i]}" "${test_type[$j]}" "${block_size[$k]}" ${block_file} >> "test.log" 2>&1
            echo "cmdline: ${exec_path} ${traces}/${trace_text[$i]} ${test_type[$j]} ${block_size[$k]} ${block_file}"
        done
    done
done