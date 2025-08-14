import os
import subprocess
import re
import pandas as pd
import argparse

from logger import Log

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
build_root = os.path.join(project_root, "build")
all_size = [4 ,16 ,64 ,256 ,1024, 4096, 16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024]
all_thread = [1, 2, 4, 8, 16, 32, 64, 128]
all_batch = [1, 2, 4, 8, 16, 32, 64, 128, 256]
file_path = "/home/sc25/p5800/dataset/kvcache_tensor.bin"
nvmeof_file_path = "/home/sc25/nvme-of/kvcache_tensor.bin"
model_dir = "/home/sc25/p5800/dataset"

only_get_command = False

def exist_path(path):
    return os.path.exists(path)

def run_cmd(bin, args: list):
    if not os.path.exists(bin):
        Log.error(f"bin {bin} not found")
        raise RuntimeError()
    args = [str(x) for x in args]
    cmdline = "sudo {} {}".format(bin, " ".join(args))
    Log.info(cmdline)
    if only_get_command:
        return ""
    popen = subprocess.check_output(cmdline, shell=True)
    return popen.decode()

def run_fig3():
    block_size = [4 ,8 ,16 ,32 ,64 ,128 ,256 ,512 ,1024 ,2048 ,4096]
    bin_path = os.path.join(build_root, "bin/breakdown")

    result = [[], []]
    for t in [0, 1]:
        for bs in block_size:
            result[t].append(run_cmd(bin_path, [file_path, t, bs, 10]))
    
    if only_get_command:
        return
    
    def parse(data, pattern):
        result = []
        for d in data:
            m = re.search(pattern, d)
            if m:
                result.append(m.group(1))
            else:
                raise RuntimeError
        return result
    

    patterns = [
        {"phxfs_regmem(ns)": r"phxfs_regmem:\s*(\d+)\s*ns",
        "phxfs_deregmem(ns)": r"phxfs_deregmem:\s*(\d+)\s*ns"},
        {"cuFileBufRegister(ns)": r"cuFileBufRegister:\s*(\d+)\s*ns",
        "cuFileBufDeregister(ns)": r"cuFileBufDeregister:\s*(\d+)\s*ns"}]

    df = pd.DataFrame()
    df["block size(KB)"] = block_size
    df.set_index("block size(KB)")
    for t in [0, 1]:
        for k, v in patterns[t].items():
            df[k] = parse(result[t], v)
    df.to_excel("results/fig3.xlsx", index=True, merge_cells=True)
    

def run_table3():
    block_size = [64]
    bin_path = os.path.join(build_root, "bin/breakdown")
    

    result = [[], []]
    for t in [0, 1]:
        for bs in block_size:
            result[t].append(run_cmd(bin_path, [file_path, t, bs, 10]))

    if only_get_command:
        return

    def parse(data, pattern):
        result = []
        for d in data:
            m = re.search(pattern, d)
            if m:
                result.append(m.group(1))
            else:
                raise RuntimeError
        return result
    
    print("\n".join(result[1][0].split("\n")[4:]))
    patterns = [
        {"phxfs_open(ns)": r"phxfs_open:\s*(\d+)\s*ns",
        "phxfs_close(ns)": r"phxfs_close:\s*(\d+)\s*ns"},
        {"cuFileDriverOpen(ns)": r"cuFileDriverOpen:\s*(\d+)\s*ns",
        "cuFileDriverClose(ns)": r"cuFileDriverClose:\s*(\d+)\s*ns"}]

    df = pd.DataFrame()
    for t in [0, 1]:
        for k, v in patterns[t].items():
            df[k] = parse(result[t], v)
    df.to_excel("results/table1.xlsx", index=True, merge_cells=True)

def run_io(args):
    bin_path = os.path.join(build_root, "bin", "microbenchmark")
    return run_cmd(bin_path, args)

def run_fig4():
    result = [pd.DataFrame(columns=["bandwidth(MB/s)", "latency(us)", "P95(us)", "P99(us)", "P99.9(us)"]), 
              pd.DataFrame(columns=["bandwidth(MB/s)", "latency(us)", "P95(us)", "P99(us)", "P99.9(us)"])]

    block_size = all_size
    thread = 1
    rw = "read"
    async_mode = 0
    
    io_depth = 1

    func_args = lambda xfer_mode, bs: [f"-f {file_path}", "-l 10G", f"-s {bs}", f"-t {thread}", f"-i {io_depth}", f"-m {rw}", f"-a {async_mode}", f"-x {xfer_mode}"]
    for t in [0, 1]: 
        for bs in block_size:
            data = run_io(func_args(t, bs * 1024))
            if only_get_command:
                continue
            data = data.split("\n")[-6:-1]
            pattern = r":\s*(\d+?.\d*)\s*"
            tmp = []
            for d in data:
                m = re.search(pattern, d)
                if m:
                    tmp.append(m.group(1))
                else:
                    print("not match")
            result[t].loc[len(result[t])] = tmp
    if only_get_command:
        return
    
    final_result = pd.concat(result, 
    axis=1,
    keys=["Phoenix", "NVIDIA GDS"])
    final_result["block size(KB)"] = block_size
    final_result.set_index("block size(KB)", inplace=True)
    final_result.to_excel("results/fig4.xlsx", index=True, merge_cells=True)


def run_fig5():
    result = [pd.DataFrame(columns=["bandwidth(MB/s)", "latency(us)", "P95(us)", "P99(us)", "P99.9(us)"]), 
              pd.DataFrame(columns=["bandwidth(MB/s)", "latency(us)", "P95(us)", "P99(us)", "P99.9(us)"])]

    block_size = 4096
    threads = all_thread
    rw = "read"
    async_mode = 0
    
    io_depth = 1

    func_args = lambda xfer_mode, thread: [f"-f {file_path}", "-l 10G", f"-s {block_size}", f"-t {thread}", f"-i {io_depth}", f"-m {rw}", f"-a {async_mode}", f"-x {xfer_mode}"]
    for t in [0, 1]: 
        for thread in threads:
            data = run_io(func_args(t, thread))
            if only_get_command:
                continue
            data = data.split("\n")[11:-1]
            pattern = r":\s*(\d+?.\d*)\s*"
            tmp = []
            for d in data:
                m = re.search(pattern, d)
                if m:
                    tmp.append(m.group(1))
                else:
                    print("not match")
            result[t].loc[len(result[t])] = tmp
    
    if only_get_command:
        return
    
    final_result = pd.concat(result, 
    axis=1,
    keys=["Phoenix", "NVIDIA GDS"])
    final_result["threads"] = threads
    final_result.set_index("threads", inplace=True)
    final_result.to_excel("results/fig5.xlsx", index=True, merge_cells=True)

def run_fig6():
    result = [pd.DataFrame(columns=["bandwidth(MB/s)", "latency(us)", "P95(us)", "P99(us)", "P99.9(us)"]), 
              pd.DataFrame(columns=["bandwidth(MB/s)", "latency(us)", "P95(us)", "P99(us)", "P99.9(us)"]),
              pd.DataFrame(columns=["bandwidth(MB/s)", "latency(us)", "P95(us)", "P99(us)", "P99.9(us)"])]

    block_size = all_size
    thread = 1
    rw = "read"
    async_mode = 1
    
    io_depth = 16

    func_args = lambda xfer_mode, bs, mode: [f"-f {file_path}", f"-l 20G", f"-s {bs}", f"-t {thread}", f"-i {io_depth}", f"-m {rw}", f"-a {mode}", f"-x {xfer_mode}"]
    for t in [0, 1, 2]: 
        for bs in block_size:
            if t == 2:
                data = run_io(func_args(0, bs * 1024, 3))
            else:
                data = run_io(func_args(t, bs * 1024, async_mode))
            if only_get_command:
                continue
            data = data.split("\n")[-6:-1]
            pattern = r":\s*(\d+?.\d*)\s*"
            tmp = []
            for d in data:
                m = re.search(pattern, d)
                if m:
                    tmp.append(m.group(1))
                else:
                    print("not match")
            result[t].loc[len(result[t])] = tmp
    
    if only_get_command:
        return

    final_result = pd.concat(result, 
    axis=1,
    keys=["Phoenix", "NVIDIA GDS", "Phoenix with stream"])
    final_result["block size(KB)"] = block_size
    final_result.set_index("block size(KB)", inplace=True)
    final_result.to_excel("results/fig6.xlsx", index=True, merge_cells=True)

def run_fig7():
    result = [pd.DataFrame(columns=["bandwidth(MB/s)", "latency(us)", "P95(us)", "P99(us)", "P99.9(us)"]), 
              pd.DataFrame(columns=["bandwidth(MB/s)", "latency(us)", "P95(us)", "P99(us)", "P99.9(us)"]),]
    block_size = 4096
    thread = 1
    rw = "read"
    async_mode = 2
    

    batch_size = all_batch

    func_args = lambda xfer_mode, batch: [f"-f {file_path}", "-l 10G", f"-s {block_size}", f"-t {thread}", f"-i {batch}", f"-m {rw}", f"-a {async_mode}", f"-x {xfer_mode}"]
    for t in [0, 1]: 
        for batch in batch_size:
            data = run_io(func_args(t, batch))
            if only_get_command:
                continue
            data = data.split("\n")[-6:-1]
            pattern = r":\s*(\d+?.\d*)\s*"
            print(data)
            tmp = []
            for d in data:
                m = re.search(pattern, d)
                if m:
                    tmp.append(m.group(1))
                else:
                    print("not match")
            result[t].loc[len(result[t])] = tmp
    
    if only_get_command:
        return

    final_result = pd.concat(result, 
    axis=1,
    keys=["Phoenix", "NVIDIA GDS"])
    final_result["batch size"] = batch_size
    final_result.set_index("batch size", inplace=True)
    final_result.to_excel("results/fig7.xlsx", index=True, merge_cells=True)

def run_fig8():
    Log.info("Small size result")
    bin_path = os.path.join(build_root, "bin/end-to-end")

    result = [pd.DataFrame(columns=["end to end", "io"]), pd.DataFrame(columns=["end to end", "io"])]
    pattern = r"\s*(\d+?.\d*)\s*us"

    for t in [0, 1]:
        GB = 1024 * 1024 * 1024
        for bs in [GB, 2 * GB, 4 * GB]:
            data = run_cmd(bin_path, [file_path, bs, "phxfs" if t == 0 else "gds"])
            if only_get_command:
                continue
            data = data.split("\n")[-3:-1]
            tmp = []
            for d in data:
                m = re.search(pattern, d)
                if m:
                    tmp.append(m.group(1))
                else:
                    print("not match")
            result[t].loc[len(result[t])] = tmp
    if only_get_command:
        return
    
    final_result = pd.concat(result, 
    axis=1,
    keys=["Phoenix", "NVIDIA GDS"])
    final_result["block size(GB)"] = [1, 2, 4]
    final_result.set_index("block size(GB)", inplace=True)
    final_result.to_excel("results/fig8.xlsx", index=True, merge_cells=True)

def run_fig9():
    return 0

def run_fig10():
    result = [pd.DataFrame(columns=["bandwidth(MB/s)", "latency(us)", "P95(us)", "P99(us)", "P99.9(us)"]), 
              pd.DataFrame(columns=["bandwidth(MB/s)", "latency(us)", "P95(us)", "P99(us)", "P99.9(us)"])]

    block_size = 4096
    threads = all_thread
    rw = "read"
    async_mode = 0
    io_depth = 1

    func_args = lambda xfer_mode, thread: [f"-f {nvmeof_file_path}", "-l 10G", f"-s {block_size}", f"-t {thread}", f"-i {io_depth}", f"-m {rw}", f"-a {async_mode}", f"-x {xfer_mode}"]
    for t in [0, 1]: 
        for thread in threads:
            data = run_io(func_args(t, thread))
            if only_get_command:
                continue
            data = data.split("\n")[11:-1]
            pattern = r":\s*(\d+?.\d*)\s*"
            tmp = []
            for d in data:
                m = re.search(pattern, d)
                if m:
                    tmp.append(m.group(1))
                else:
                    print("not match")
            result[t].loc[len(result[t])] = tmp
    
    if only_get_command:
        return

    final_result = pd.concat(result, 
    axis=1,
    keys=["Phoenix", "NVIDIA GDS"])
    final_result["threads"] = threads
    final_result.set_index("threads", inplace=True)
    final_result.to_excel("results/fig10.xlsx", index=True, merge_cells=True)



def run_fig11():
    trace_text = ["paper_assit.txt", "gsm100.txt", "quality.txt", "sharegpt-sample-200.txt"]
    bin_path = os.path.join(build_root, "bin/kvcache")
    
    traces_path = os.path.join(project_root, "benchmarks/kvcache/traces")
    
    result = [pd.DataFrame(columns=trace_text), pd.DataFrame(columns=trace_text)]

    block_size = [8192, 16384, 65536]
    for trace in trace_text:
        for t in [0, 1]:
            trace_result = []
            for b in block_size:
                data = run_cmd(bin_path, ["phxfs" if t == 0 else "gds", 0, os.path.join(traces_path, trace), b, file_path])
                if only_get_command:
                    continue
                pattern = r"IO Bandwidth:\s*(\d+?.\d*)\s*GB/s"
                m = re.search(pattern, data)
                if m:
                    trace_result.append(m.group(1))
                else:
                    Log.error("not match")
                    exit(-1)
            result[t][trace] = trace_result
    if only_get_command:
        return
    final_result = pd.concat(result, 
    axis=1,
    keys=["Phoenix", "NVIDIA GDS"])
    final_result["block size"] = block_size
    final_result.set_index("block size", inplace=True)
    final_result.to_excel("results/fig11.xlsx", index=True, merge_cells=True)

def run_fig12():
    model_list = {"facebook": [
                     "opt-2.7b_safetensors_0", "opt-6.7b_safetensors_0", "opt-13b_safetensors_0"],
                "llama": [
                        "Meta-Llama-3-8B"],
                "tiiuae": [
                        "falcon-7b", "falcon-11B"],
                "qwen" : [
                    "Qwen2.5-14B", "qwen2-5-7B"
                ]
    }

    benchmark_type = {
        "phxfs": "0",
        "gds": "1",
        "native": "2"
    }
    pattern = r'(?<=Elapsed time: )\d+\.\d+|(?<=Total size: )\d+\.\d+|(?<=Throughput: )\d+\.\d+'
    bin_path = os.path.join(build_root, "bin", "safetensor")

    result = pd.DataFrame()

    for name, benchmark in benchmark_type.items():
        cur_result = []
        for model in model_list:
            for model_name in model_list[model]:
                model_path = os.path.join(model_dir, model, model_name)
                # subprocess.run("echo 3 | sudo tee /proc/sys/vm/drop_caches", shell=True)
                if os.path.exists(bin_path) and os.path.exists(model_path):
                    data = run_cmd(bin_path, [model_path, benchmark, 0])
                    if only_get_command:
                        continue
                    numbers = re.findall(pattern, data)
                    cur_result.append(numbers[0])
                else:
                    print(f"Binary {bin_path} or model path {model_path} does not exist")
        result[name] = cur_result
    if only_get_command:
        return
    result.to_excel("results/fig12.xlsx", index=True, merge_cells=True)

def parser():
    parser = argparse.ArgumentParser()
    artifacts = [f"fig{i}" for i in range(3, 13)]
    artifacts.append("table3")
    artifacts.append("all")
    parser.add_argument("-g", "--get_commands", action="store_true", help="get the artifact command")
    parser.add_argument("-a", "--artifact", help="Artifact to reproduce", required=True, choices=artifacts)
    return parser.parse_args()


if __name__ == "__main__":
    result_dir = os.path.join(project_root, "sc25/results")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    args = parser()
    if args.get_commands:
        only_get_command = True
    
    if args.artifact == "all":
        run_table3()
        for i in range(3, 13):
            eval("run_fig{}()".format(i))
    else:
        eval(f"run_{args.artifact}()")
