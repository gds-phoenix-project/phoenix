#!/bin/bash

nvme_subsystem_name="nvme-subsystem-name"
namespaces=10
nvme_device="/dev/nvme1n1"
nvme_port=1
nvme_ip="192.168.0.206"
nvme_port=4420

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <target|initiator> <setup|cleanup>"
    exit 1
fi

mlex_model_probe(){
    modprobe nvmet
    modprobe nvmet-rdma
    modprobe nvme-rdma
}

nvme_of_target_setup(){
    sudo mkdir /sys/kernel/config/nvmet/subsystems/${nvme_subsystem_name}
    cd /sys/kernel/config/nvmet/subsystems/${nvme_subsystem_name}
    echo 1 | sudo tee attr_allow_any_host
    sudo mkdir namespaces/${namespaces}
    cd namespaces/${namespaces}
    echo -n ${nvme_device} | sudo tee device_path
    echo 1 | sudo tee enable
    mkdir /sys/kernel/config/nvmet/ports/${nvme_port}
    cd /sys/kernel/config/nvmet/ports/${nvme_port}
    echo "${nvme_ip}" | sudo tee addr_traddr
    echo rdma | sudo tee addr_trtype
    echo ${nvme_port} | sudo tee addr_trsvcid
    echo ipv4 | sudo tee addr_adrfam
    sudo ln -s /sys/kernel/config/nvmet/subsystems/${nvme_subsystem_name} /sys/kernel/config/nvmet/ports/1/subsystems/${nvme_subsystem_name}
}

nvme_of_initiator_setup(){
    sudo nvme discover -t rdma -q ${nvme_subsystem_name} -a ${nvme_ip} -s ${nvme_port}
    sudo nvme connect -t rdma -q ${nvme_subsystem_name} -n ${nvme_subsystem_name}  -a ${nvme_ip} -s ${nvme_port}
}

nvme_of_initiator_cleanup(){
    sudo nvme disconnect -n ${nvme_subsystem_name}
}

nvme_of_target_cleanup(){
    sudo rmdir /sys/kernel/config/nvmet/ports/${nvme_port}/subsystems/${nvme_subsystem_name}
    sudo rmdir /sys/kernel/config/nvmet/subsystems/${nvme_subsystem_name}
}

check_nvme_cli(){
    if ! command -v nvme &> /dev/null
    then
        echo "nvme command could not be found"
        exit
    fi
}

case "$1" in
    "target")
        check_nvme_cli
        if [ "$2" == "setup" ]; then
            nvme_of_target_setup
        elif [ "$2" == "cleanup" ]; then
            nvme_of_target_cleanup
        else
            echo "Usage: $0 target <setup|cleanup>"
        fi
        ;;
    "initiator")
        check_nvme_cli
        if [ "$2" == "setup" ]; then
            nvme_of_initiator_setup
        elif [ "$2" == "cleanup" ]; then
            nvme_of_initiator_cleanup
        else
            echo "Usage: $0 initiator <setup|cleanup>"
        fi
        ;;
    *)
        echo "Usage: $0 <target|initiator> <setup|cleanup>"
        exit 1
        ;;
esac