#!/bin/bash
#SBATCH --gres=gpu:h100_80gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000m
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=rltc
#SBATCH --partition=blanca-clearlab1
#SBATCH --account=blanca-clearlab1
#SBATCH --qos=blanca-clearlab1
#SBATCH --mail-user=michael.ginn@colorado.edu
#SBATCH --mail-type=ALL

# module load cuda/12.1.1
# module load gcc/11.2.0
# module load miniforge
# mamba activate flash_env
module load uv
uv sync
uv pip install flash-attn setuptools --no-build-isolation

export PYTORCH_ALLOC_CON=expandable_segments:True

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
