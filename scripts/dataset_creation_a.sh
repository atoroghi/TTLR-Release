#!/bin/bash
#SBATCH --job-name=three_dataset_creation_metamath
#SBATCH --nodes=1
#SBATCH --time=03:00:00
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
CUDA_VISIBLE_DEVICES=0 python -m src.llamafactory.train.ttlr.create_augmented_dataset train examples/train_lora/offline_ttlr_qwen.yaml --start_index 4000 --end_index 5000