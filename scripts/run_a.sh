#!/bin/bash
#SBATCH --job-name=one_ttlr_colota
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --account=def-ssanner
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=32G
#SBATCH --mail-user=armin.toroghi@mail.utoronto.ca
#SBATCH --mail-type=ALL

export PYTHONNOUSERSITE=1

# Must load these BEFORE activating venv
module purge
module load gcc
module load arrow
module list

VENV=/home/atoroghi/projects/def-ssanner/atoroghi/project/myenv
source $VENV/bin/activate

echo "Using Python at: $(which python)"
python - << 'EOF'
import pyarrow, sys
print("pyarrow:", pyarrow.__version__)
print("python:", sys.executable)
EOF

cd ~/projects/def-ssanner/atoroghi/project/TTL
export DISABLE_VERSION_CHECK=1
export PYTHONPATH=$PWD/src:$PYTHONPATH
# CUDA_VISIBLE_DEVICES=0 python -m llamafactory.cli train examples/train_lora/offline_ttl_math_qwen_2048.yaml
CUDA_VISIBLE_DEVICES=0 python -m llamafactory.cli train examples/train_lora/offline_ttl_qwen2_cotgen_colota.yaml
# CUDA_VISIBLE_DEVICES=0 python -m llamafactory.cli train examples/train_lora/offline_ttlr_qwen_selected.yaml

# CUDA_VISIBLE_DEVICES=0 python -m llamafactory.cli train examples/train_lora/offline_ttlr_select_mini15.yaml
# CUDA_VISIBLE_DEVICES=0 python -m llamafactory.cli train examples/train_lora/offline_ttlr_10.yaml

# CUDA_VISIBLE_DEVICES=0 python -m llamafactory.cli train examples/train_lora/offline_ttlr_3000.yaml

# CUDA_VISIBLE_DEVICES=0 python -m llamafactory.cli train examples/train_lora/offline_ttlr_4000.yaml

# CUDA_VISIBLE_DEVICES=0 python -m llamafactory.cli train examples/train_lora/offline_ttlr_5000.yaml