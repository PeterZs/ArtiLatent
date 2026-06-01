import subprocess
import os

world_size = 2  # Number of parallel processes (e.g., number of GPUs)
ranks = 2
script = "dataset_toolkits/clip_encode_part.py"
dataset_name = "PartMove"  # Add the missing dataset name
log_dir = "clip_encode_part_logs"
os.makedirs(log_dir, exist_ok=True)
processes = []

for rank in range(ranks):
    log_file = os.path.join(log_dir, f"clip_encode_part_rank_{rank}.log")
    with open(log_file, "w") as logf:
        cmd = [
            "python", script, dataset_name,  # Add dataset_name as first argument
            "--rank", str(rank),
            "--world_size", str(world_size),
        ]
        print(f"Launching rank {rank} ...")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)  # Uncomment if using GPUs
        proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, env=env)
        processes.append(proc)

for proc in processes:
    proc.wait()

print("All feature extraction processes finished.")