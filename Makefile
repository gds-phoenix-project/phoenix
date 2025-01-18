################################################################################
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Linux Platforms)
#
################################################################################

# Common includes and paths for CUDA
ARCH ?= $(shell uname -m)
CUDA_PATH   := /usr/local/cuda-12.4
ifeq ($(ARCH),aarch64)
	CUFILE_PATH ?= $(CUDA_PATH)/targets/sbsa-linux/lib/
	CUFILE_INCLUDE_PATH ?= $(CUDA_PATH)/targets/sbsa-linux/include/
else
	CUFILE_PATH ?= $(CUDA_PATH)/targets/x86_64-linux/lib/
	CUFILE_INCLUDE_PATH ?= $(CUDA_PATH)/targets/x86_64-linux/include/
endif
CXXFLAGS    := -Wall
CXXFLAGS    += -I $(CUDA_PATH)/include/ 
CXXFLAGS    += -I $(CUFILE_INCLUDE_PATH)
CXXFLAGS    += -I ./
ifneq ($(CONFIG_CODE_COVERAGE),)
CXXFLAGS    += -ftest-coverage -fprofile-arcs
endif
CXXFLAGS 	+= -std=c++17
CXXFLAGS 	+= -O3

ifeq ($(GDS),1)
    CXXFLAGS += -DGDS
endif

CUDART_STATIC := -Bstatic -L $(CUDA_PATH)/lib64/ -lcudart_static -lrt -lpthread -ldl
LDFLAGS     :=  $(CUFILE_LIB) $(CUDART_STATIC) -lcrypto -lssl
CUFILE_LIB  := -L $(CUFILE_PATH) -lcufile
CUFILE_LIB_STATIC  := -L $(CUFILE_PATH) -lcufile_static
LDFLAGS     :=  $(CUFILE_LIB) -L $(CUDA_PATH)/lib64/stubs -lcuda $(CUDART_STATIC) -Bdynamic -lrt
LDFLAGS_STATIC     :=  $(CUFILE_LIB_STATIC) -L $(CUDA_PATH)/lib64/stubs -lcuda $(CUDART_STATIC) -Bdynamic -lrt -ldl
INSTALL_GDSSAMPLES_PREFIX = /usr/local/gds/samples
NVCC          :=$(CUDA_PATH)//bin/nvcc
CC:=g++
# Target rules
all: phxfs_bench breakdown loop_performance

phxfs_bench: main.cc gds_core.cc phxfs_core.cc posix_core.cc phxfs.cc $(CUFILE_INCLUDE_PATH)/cufile.h
	$(CC) $(INCLUDES) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -ldl -lpthread -lcufile -lcuda -lcudart -luring -laio

breakdown: breakdown.cc phxfs.cc $(CUFILE_INCLUDE_PATH)/cufile.h
	$(CC) $(INCLUDES) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -ldl -lpthread -lcufile -lcuda -lcudart -luring

loop_performance: loop_performance.cc phxfs.cc $(CUFILE_INCLUDE_PATH)/cufile.h
	$(CC) $(INCLUDES) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -ldl -lpthread -lcufile -lcuda -lcudart -luring

clean:
	find . -type f -executable -delete
	rm -f *.o cufile.log phxfs_bench breakdown loop_performance

.PHONY : build install clean
