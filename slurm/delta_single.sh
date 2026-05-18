#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH --partition=ghx4
#SBATCH --account=bgje-dtai-gh
#SBATCH --gpu-bind=verbose,closest
#SBATCH --out=logs/%j.log
#SBATCH --error=logs/%j.log
#SBATCH --mail-user=michael.ginn@colorado.edu
#SBATCH --mail-type=ALL

cd "$SLURM_SUBMIT_DIR"
uv sync
uv pip install flash-attn setuptools --no-build-isolation

echo "=== CUDA + PyTorch diagnostics ==="
uv run python - <<'PY'
import torch, os
print("CUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch CUDA version:", torch.version.cuda)
print("Torch built with:", torch.__config__.show())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Detected GPUs:", torch.cuda.device_count())
    print("GPU 0:", torch.cuda.get_device_name(0))
PY

uv run python run.py "$1" "${@:2}"
