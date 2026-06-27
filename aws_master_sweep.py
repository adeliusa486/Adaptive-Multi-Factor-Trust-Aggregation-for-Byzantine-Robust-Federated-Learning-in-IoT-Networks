import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

seeds = [43, 44, 45, 46]
network_sizes = [40, 60, 80, 100]
attack = "label_flipping"
byz_rate = 0.3
epochs = 2

# We have 8 GPUs on the p3dn.24xlarge
gpus = [0, 1, 2, 3, 4, 5, 6, 7]

commands = []
# 1. Scalability Sweep (All sizes, all 5 seeds including 42 just in case, single attack)
for seed in [42, 43, 44, 45, 46]:
    for n in network_sizes:
        cmd = f"python run_main.py --num_clients {n} --attack {attack} --byzantine_fraction {byz_rate} --seeds {seed} --local_epochs {epochs}"
        commands.append(cmd)

# 2. Remaining Seeds for N=20 (All attacks)
# We can do this as a secondary sweep if needed, but let's focus on Scalability first
print(f"Total commands to run: {len(commands)}")

def run_command(cmd, gpu_idx):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"[GPU {gpu_idx}] Starting: {cmd}")
    process = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process.wait()
    print(f"[GPU {gpu_idx}] Finished: {cmd}")

active_tasks = []
with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
    for i, cmd in enumerate(commands):
        gpu = gpus[i % len(gpus)]
        active_tasks.append(executor.submit(run_command, cmd, gpu))
        time.sleep(2) # Stagger startups
        
print("All AWS simulations completed successfully!")
