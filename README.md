# Phoenix: GPU Direct Storage without Phony Buffers
This is the open-source repository for our paper: **Phoenix: GPU Direct Storage without Phony Buffers**.

## Directory structure
```python
phonix
|--benchmarks   # artifact evaluation files
|--example      # example for how to use phoenix
|--module       # kernel module for phoenix
|--scripts      # test scripts
```

## How to build

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

### 2. OFED Driver
```shell
sudo ./mlnxofedinstall --with-nvmf --with-nfsrdma --enable-gds --add-kernel-support --dkms --skip-unsupported-devices-check
sudo update-initramfs -u -k `uname -r`
sudo reboot
```
### 3. NVMe-of/NFS
#### 3.1 NVMe-of
```shell
cd scripts
bash nvme_of.sh <target|initiator> <setup|cleanup>
```
#### 3.2 NFS
* server
```shell
sudo apt-get install nfs-kernel-server
mkfs.ext4 /dev/nvme0n1
mount -o data=ordered /dev/nvme0n1 /mnt/nvme
echo /mnt/nvme *(rw,async,insecure,no_root_squash,no_subtree_check) | sudo tee /etc/exports
service nfs-kernel-server restart
modprobe rpcrdma 
echo rdma 20049 > /proc/fs/nfsd/portlist
```
* clinet
```shell
sudo apt-get install nfs-common
modprobe rpcrdma
mkdir -p /mnt/nfs_rdma_gds
sudo mount -v -o proto=rdma,port=20049,vers=3 192.168.0.206:/mnt/nvme /mnt/nfs_rdma_gds
```
### 4. Phoenix module
```shell
cd module && make
sudo make ins
```
## How to run
Here, we provide the experimental scripts corresponding to the paper.
### 0. Build benchmarks
```shell
cd benchmarks && make
```
### 1. Motivation
```shell
cd benchmarks
# Table 1, 0 -> GDS
sudo ./breakdown 0 10000 
```
### 2. Breakdown of Oprations
```shell
# Tabel 2, 1 -> Phoenix
cd scripts
sudo ./breakdown 1 10000 
```

```shell
# Fig.3
cd scripts
sudo bash get_op_latency.sh <0|1> 0
```
### 3. Comparison with NVIDIA GDS
```shell
cd scripts
# see run_batch.py for detail
sudo python run_batch.py <0|1> <0|1|2> 0
```
### 4. End to End Perfmance
```shell
cd benchmarks
sudo ./loop_performance <0|1> <1048576|2,097152|4194304> 0 # 1G 2G 4G
```