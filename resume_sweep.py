import os
import glob
import subprocess

methods = ["fedavg", "trimmed_mean", "krum", "fltrust", "amfta"]
rates = [0.10, 0.20, 0.30, 0.40]
attacks = ["label_flipping", "gaussian_noise"]
seed = 42

for method in methods:
    for rate in rates:
        for attack in attacks:
            # Check if this combination already has a json file in results/
            pattern1 = f"results/{method}_byz{rate:.1f}_{attack}_seed{seed}_*.json"
            pattern2 = f"results/{method}_byz{rate}_{attack}_seed{seed}_*.json"
            if len(glob.glob(pattern1)) > 0 or len(glob.glob(pattern2)) > 0:
                print(f"Skipping {method} {rate} {attack} - already completed!")
                continue
            
            print(f"Running {method} {rate} {attack}...")
            cmd = f'python experiments/run_main.py --method {method} --byzantine_fraction {rate} --attack {attack} --num_clients 20 --num_rounds 50 --local_epochs 2 --seeds {seed}'
            subprocess.run(cmd, shell=True)
