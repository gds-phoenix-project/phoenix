cmake_minimum_required (VERSION 3.18)
project (libphoenix-benchmarks)
find_package (CUDA 12.4 REQUIRED)

file(GLOB safetensor_src "${benchmarks_root}/safetensor/*.cc")

make_benchmark (safetensor safetensor "${safetensor_src}")
