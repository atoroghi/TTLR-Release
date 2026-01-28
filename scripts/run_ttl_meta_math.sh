#!/bin/bash
#SBATCH --job-name=run_ttl
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --account=aip-ssanner
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=32G
#SBATCH --mail-user=faeze.moradi@mail.utoronto.ca
#SBATCH --mail-type=ALL

export PYTHONNOUSERSITE=1

# Must load these BEFORE activating venv
module purge
module load gcc
module load arrow
module list

VENV=/home/kalarde/projects/aip-ssanner/kalarde/myenv
source $VENV/bin/activate

echo "Using Python at: $(which python)"
python - << 'EOF'
import pyarrow, sys
print("pyarrow:", pyarrow.__version__)
print("python:", sys.executable)
EOF

cd ~/projects/aip-ssanner/kalarde/TTL
export DISABLE_VERSION_CHECK=1
export PYTHONPATH=$PWD/src:$PYTHONPATH




CUDA_VISIBLE_DEVICES=0 python -m llamafactory.cli train examples/train_lora/offline_ttl_llama31_meta_math.yaml
