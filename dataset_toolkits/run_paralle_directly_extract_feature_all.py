import subprocess
import os

world_size = 4  # Number of parallel processes (e.g., number of GPUs)
ranks = 4
script = "dataset_toolkits/extract_feature_state.py"
log_dir = "extract_feature_logs"
os.makedirs(log_dir, exist_ok=True)
processes = []

for rank in range(ranks):
    log_file = os.path.join(log_dir, f"extract_rank_{rank}.log")
    with open(log_file, "w") as logf:
        cmd = [
            "python", script,
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