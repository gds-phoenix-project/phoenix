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
block_file="/home/sc25/p5800/dataset/kvcache_tensor.bin"
block_size=(8192 16384 65536)
test_type=("phxfs" "gds")

for i in ${!trace_text[@]}; do
    for j in ${!test_type[@]}; do
        for k in ${!block_size[@]}; do
            echo "Running kvcache with ${traces}/${trace_text[$i]} with block size ${block_size[$k]}" >> "kvcache.log"
            sudo ${exec_path} "${test_type[$j]}" 0 "${traces}/${trace_text[$i]}" "${block_size[$k]}" ${block_file} >> "kvcache.log" 2>&1
            echo "cmdline: sudo ${exec_path} "${test_type[$j]}" 0 "${traces}/${trace_text[$i]}" "${block_size[$k]}" ${block_file}"
        done
    done
done