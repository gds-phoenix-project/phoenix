import os
import re
import subprocess

import time
import shlex
import json


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

model_prefix = "/mnt/phxfs"
file_path = os.path.abspath(__file__)
# get the parent directory
parent_dir = os.path.join(os.path.dirname(file_path), "../")
benchmark_type = {
    "phxfs": "0",
    "gds": "1",
    "native": "2"
}

pattern = r'(?<=Elapsed time: )\d+\.\d+|(?<=Total size: )\d+\.\d+|(?<=Throughput: )\d+\.\d+'

env = os.environ.copy()

result = {
    "phxfs": {},
    "gds": {},
    "native": {}
}

def write_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

now = time.time()

bin_path = os.path.join(parent_dir, "build", "bin", "safetensor")
device_id = 0

for name, benchmark in benchmark_type.items():
    for model in model_list:
        for model_name in model_list[model]:
            model_path = os.path.join(model_prefix, model, model_name)
            # subprocess.run("echo 3 | sudo tee /proc/sys/vm/drop_caches", shell=True)
            if os.path.exists(bin_path) and os.path.exists(model_path):
                print(f"Running cmdline: {bin_path} {model_path} {benchmark} {device_id}")
                popen = subprocess.Popen(shlex.split(f"{bin_path} {model_path} {benchmark} {device_id}"), stdout=subprocess.PIPE, text=True, env=env)
                popen.wait()
                output = popen.stdout.read()
                numbers = re.findall(pattern, output)
                result[benchmark][model_name] = {
                    "Elapsed time": float(numbers[0]),
                    "Total size": float(numbers[1]),
                    "Throughput": float(numbers[2]),
                }
                print(f"resutl: {result}")
            else:
                print(f"Binary {bin_path} or model path {model_path} does not exist")

write_json(f"result-{now}.json", result)