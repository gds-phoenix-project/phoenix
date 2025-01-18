import os
import subprocess
import sys
import re
import time
import shlex

from logger import Log

result_path = ""
IS_PERF = False
FILE_PATH = ["/mnt/phxfs/test.data", "/mnt/nvme-of/test.data", "/mnt/nfs/test.data"]
SUBDIR = "phxfs"
# 1M 
io_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
threads = [1, 2, 4, 8, 16, 32, 64, 128]
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

read_write = ["read", "write"]
file_path = os.path.dirname(os.path.realpath(__file__))
phxfs_exec = os.path.join(file_path, "..", "benchmarks", "phxfs_bench")

class test_config:
    def __init__(self):
        self.muti_size = False
        self.muti_thread = False
        self.muti_batch = False
        self.async_mode = 0
        self.xfer_mode = 0
    
    def reset(self):
        self.muti_size = False
        self.muti_thread = False
        self.muti_batch = False

# for get cpu utilization
PERF_CONFIG_PATH = "/mnt/phxfs/perf-6.6.0/tools/perf/perf" 

pattern = r"(?:Average IO bandwidth|Average IO latency|95th percentile latency|99th percentile latency|99.9th percentile latency):\s*([\d.]+)"

def get_perf_cmdline(pid):
    return f"sudo {PERF_CONFIG_PATH} stat -p {pid} -e task-clock -I 500"

def run_perf(pid, file_name):
    file_name = os.path.join(file_path, "results/cpu", file_name)
    with open(file_name, 'a') as f:
        return subprocess.Popen(shlex.split(get_perf_cmdline(pid)),
                                stdout=f,
                                stderr=f)

def get_chile_pid(pid):
    return subprocess.check_output(shlex.split(f"pgrep -P {pid}"))

def run_phxfs_bench(rw="read", io_size=4, thread=1, batch_size=1, file_path=FILE_PATH[0], async_mode=0, xfer_mode=0):
    return f"sudo numactl -N 1 {phxfs_exec} -f {file_path} -l 10G -s {io_size}k -t {thread} -i 16 -m {rw} -a {async_mode} -d 0 -x {xfer_mode}"

def parse_result(result):
    matches = re.findall(pattern, result)
    matches = [float(match) for match in matches]
    return matches[0], matches[1], matches[2], matches[3], matches[4]

def x_thread_y_size_z_batch(config: test_config):
    f = open(result_path, "a+")
    io_size_iter = io_sizes if config.muti_size == True else [4]
    thread_iter = threads if config.muti_thread  == True else [1]
    batch_size_iter = batch_sizes if config.muti_batch == True else [1]
    run_type_name = "phxfs" if config.xfer_mode == 0 else "gds"
    f.write("thread-rw-io_size-batch_size,bandwidth,latency,p95_latency,p99_latency,p999_latency\n")
    f.write(f"async_mode: {config.async_mode}, xfer_mode: {config.xfer_mode}\n")
    for rw in read_write:
        for io_size in io_size_iter:
            for thread in thread_iter:
                for batch_size in batch_size_iter:
                    cmdline = run_phxfs_bench(rw=rw, io_size=io_size, thread=thread, batch_size=batch_size, async_mode=config.async_mode, xfer_mode=config.xfer_mode)
                    Log.info(f"Run {cmdline}")
                    if IS_PERF:
                        io_test = subprocess.Popen(shlex.split(cmdline), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        time.sleep(1)
                        pid = get_chile_pid(io_test.pid).decode()
                        pid = get_chile_pid(pid).decode()
                        perf = run_perf(pid, f"{run_type_name}-{config.async_mode}")
                        perf.wait()
                        io_test.wait()
                        perf.kill()
                    else:
                        result = subprocess.check_output(cmdline, shell=True).decode()
                        Log.info(result)
                    bandwidth, latency, p95_latency, p99_latency, p999_latency = parse_result(result)
                    f.write(f"{thread}-{rw}-{io_size}-{batch_size},{bandwidth},{latency},{p95_latency},{p99_latency},{p999_latency}\n")
                    f.flush()

def run_perf_cpu(pid: int):
    return subprocess.Popen("")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        Log.error("Usage: python run_batch.py <xfer_mode> <mode> <device_type>")
        Log.info("xfer_mode: 0 - phxfs, 1 - gds")
        Log.info("mode: 0 - sync, 1 - async, 2 - batch")
        Log.info("device_type: 0 - nvme, 1 - nvmeof")
        sys.exit(1)
    run_batch_type = int(sys.argv[1])
    run_mode = sys.argv[2]
    run_device_type = sys.argv[3]
    
    print(run_batch_type)
    
    result_dir = os.path.join(file_path, "results", "latency", "phxfs" if run_batch_type == 0 else "gds")
    print(result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
        
    run_device_type = int(run_device_type)
    run_mode = int(run_mode)
    
    name_dict = {0:"sync", 1:"async", 2:"batch"}
    dev_type_dict = {0:"nvme", 1:"nvmeof"}    
    
    result_path = os.path.join(result_dir, name_dict[run_mode] + "_" + dev_type_dict[run_device_type] + ".txt")
    
    config = test_config()
    config.reset()
    config.async_mode = run_mode
    config.xfer_mode = run_batch_type
    
    if run_mode in [0, 1]:
        config.muti_size = True
        x_thread_y_size_z_batch(config)
        config.reset()
    else:
        config.muti_batch = True
        x_thread_y_size_z_batch(config)
        config.reset()
    
    
    
    




    


                    