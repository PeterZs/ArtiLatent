import subprocess
import os
import sys

# ======== 配置项 ========
num_ranks = 4  # 启动多少个进程（即 rank 数）
render_script = "dataset_toolkits/directly_render_state_all.py"
dataset_name = "PartMove"
num_views = 48
num_states = 8
max_workers = 8
world_size = 2  # Adjust to number of available GPUs

log_dir = "render_logs"

# ======== 创建日志目录 ========
os.makedirs(log_dir, exist_ok=True)

# ======== 启动进程 ========
processes = []

for rank in range(num_ranks):
    log_file = os.path.join(log_dir, f"rank_{rank}.log")
    logf = open(log_file, "w")  # 保持文件打开直到子进程结束

    cmd = [
        sys.executable, render_script, dataset_name,
        "--rank", str(rank),
        "--world_size", str(world_size),
    ]

    print(f"Launching rank {rank}...", flush=True)

    # 你可以根据需要设置环境变量，例如按GPU分配：
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(rank)  # 可选

    proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, env=env)
    processes.append((proc, logf))

# ======== 等待所有进程完成 ========
for rank, (proc, logf) in enumerate(processes):
    ret = proc.wait()
    logf.close()
    if ret != 0:
        print(f"进程 rank {rank} 异常退出，返回码：{ret}", flush=True)

print("所有渲染进程完成。", flush=True)
