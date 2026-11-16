#!/bin/bash
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=rltc
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --mail-user=jacob.burton-1@colorado.edu
#SBATCH --mail-type=ALL

# module load cuda/12.1.1
# module load gcc/11.2.0
# module load miniforge
# mamba activate flash_env
module load uv
export HF_HOME=/projects/$USER/.cache/huggingface
export UV_CACHE_DIR=/projects/$USER/.cache/uv
export TF_FORCE_GPU_ALLOW_GROWTH=true
uv sync
# COMET needs transformers<5, which conflicts with pyproject.toml, so install
# the metric packages on top and skip uv run's auto-sync (it would undo this).
uv pip install unbabel-comet bert-score
uv pip install "git+https://github.com/google-research/bleurt.git"

export PYTORCH_ALLOC_CON=expandable_segments:True

echo "=== CUDA + PyTorch diagnostics ==="
uv run --no-sync python - <<'PY'
import torch, os
print("CUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch CUDA version:", torch.version.cuda)
print("Torch built with:", torch.__config__.show())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Detected GPUs:", torch.cuda.device_count())
    print("GPU 0:", torch.cuda.get_device_name(0))
PY

uv run --no-sync python run.py "$1" "${@:2}"
