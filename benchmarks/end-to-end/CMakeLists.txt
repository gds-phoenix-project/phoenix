cmake_minimum_required (VERSION 3.18)
project (libphoenix-benchmarks)
find_package (CUDA 12.4 REQUIRED)


make_benchmark (end-to-end end-to-end "end-to-end.cc")
