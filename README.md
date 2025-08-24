# Phoenix: A Refactored I/O Stack for GPU Direct Storage without Phony Buffers
This is the open-source repository for our paper: **"Phoenix: A Refactored I/O Stack for GPU Direct Storage without Phony Buffers"**, which has been accepted for [SC'25](https://sc25.supercomputing.org/). This documentation explains how to build, configure and use this I/O stack.

## Directory structure
```python
phonix
|--benchmarks   # artifact evaluation files
|--example      # example for how to use phoenix
|--libphoenix   # user library for phoenix
|--module       # kernel module for phoenix
|--scripts      # test scripts
```

## How to Build

### Environment
* OS: Ubuntu 22.04.2 LTS
* kernel: Linux 6.1.0-rc8
* NVIDIA driver: 550.54
* OFED driver: 24.10
* CUDA: 12.4

### 1. NVIDIA GDS

```shell
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo bash cuda_12.4.0_550.54.14_linux.run
# select nvidia-fs option and choose open driver
```

### 2. MLNX_OFED Driver
```shell
sudo ./mlnxofedinstall --with-nvmf --with-nfsrdma --enable-gds --add-kernel-support --dkms --skip-unsupported-devices-check
sudo update-initramfs -u -k `uname -r`
sudo reboot
```
### 3. NVMe-of/NFS
#### 3.1 NVMe-of
```shell
cd scripts
sudo bash nvme_of.sh <target|initiator> <setup|cleanup>
```
#### 3.2 NFS
```shell
cd scripts
sudo bash nfs.sh <server|client>
```
### 4. Phoenix and Benchmarks
```shell
mkdir -p build
cd build && cmake ../
make -j 
```
Note: this will compile all the benchmarks including the kernel module
## How to Use
### 1. Install Kernel Module
```shell
cd build && sudo make insmod
```
Note: must run `nvidia-smi` to `modprobe` nvidia driver before install phoenix kernel module.
### 2. Example for Using libphoenix
We have provided a simple example to illustrate how to program using libphoenix
```shell
# see example/example.cc
```

### 3. Evaluation Procedure
We provide some scripts to execute the evaluation procedure.

Note: make sure to update the paths in the scripts.
### 3.1 Faster Reproduction. 
We have integrated all experiment scripts and provided a Python script. 
Users can run the corresponding experiment by specifying the artifact parameter. This script will also print all the corresponding execution commands.
Before running this Python script, users need to set the variables `file_path`, `nvmeof_file_path`, and `model_dir` to the paths specific to users’ own environment. 
All results will be stored in the `phoenix/sc25/results` directory

```shell
cd phoenix/sc25
# `all` will run table3 and fig 3 ~ 12
sudo python run_all_benchmarks.py --artifact all
```
In addition, we also provide individual scripts for each experiment as follows:
#### 3.2 Breakdown
```shell
cd scripts && sudo bash breakdown.sh
```
#### 3.3 I/O Performance
```shell
cd scripts
# see micro.py for detail
sudo python micro.py <0|1> <0|1|2> 0
```
#### 3.4 End-to-End Performance
```shell
cd build/
sudo bin/end-to-end <file_path> <io_size> <mode>
```
#### 3.5 KVCache Loading
```shell
cd scripts
sudo bash kvcache.sh
```
#### 3.6 Model Loading
```shell
cd scripts
sudo python load_safetensors.py
```
