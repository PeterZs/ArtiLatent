import subprocess
import os

world_size = 2  # Adjust to number of available GPUs
render_script = "dataset_toolkits/directly_render_cond_all.py"
dataset_name = "PartMove"
log_dir = "render_logs"
ranks = 2

os.makedirs(log_dir, exist_ok=True)
processes = []

for rank in range(ranks):
    log_file = os.path.join(log_dir, f"cond_rank_{rank}.log")
    with open(log_file, "w") as logf:
        cmd = [
            "python", render_script, dataset_name,
            "--rank", str(rank),
            "--world_size", str(world_size),
        ]
        print(f"Launching rank {rank} on GPU {rank}...")

        # Set environment variable to select GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)  # Assign rank-th GPU

        proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, env=env)
        processes.append(proc)

for proc in processes:
    proc.wait()

print("All conditional rendering processes finished.")
