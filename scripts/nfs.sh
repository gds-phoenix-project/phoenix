#!/bin/bash

nvme_device="/dev/nvme1n1"
nvme_mnt_path="/mnt/nvme"
nfs_port=20049

nfs_server_setup(){
    sudo apt-get install nfs-kernel-server
    sudo mkfs.ext4 ${nvme_device}
    sudo mount -o data=ordered ${nvme_device} ${nvme_mnt_path}
    echo ${nvme_mnt_path} *(rw,async,insecure,no_root_squash,no_subtree_check) | sudo tee /etc/exports
    sudo service nfs-kernel-server restart
    sudo modprobe rpcrdma
    echo "rdma ${nfs_port}" | sudo tee /proc/fs/nfsd/portlist
}

nfs_client_setup(){
    sudo apt-get install nfs-common
    sudo modprobe rpcrdma
    sudo mkdir -p ${nvme_mnt_path}
    sudo mount -v -o vers=4,proto=rdma,port=${nfs_port} ${nvme_ip}:${nvme_mnt_path} ${nvme_mnt_path}
}

case "$1" in
    "server")
        nfs_server_setup
        ;;
    "client")
        nfs_client_setup
        ;;
    *)
        echo "Usage: $0 <server|client>"
        exit 1
        ;;
esac
