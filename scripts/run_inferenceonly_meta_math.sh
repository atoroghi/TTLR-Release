#!/bin/bash
#SBATCH --job-name=run_inferenceonly
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --account=def-benliang
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

VENV=/home/kalarde/projects/def-benliang/kalarde/myenv
source $VENV/bin/activate

echo "Using Python at: $(which python)"
python - << 'EOF'
import pyarrow, sys
print("pyarrow:", pyarrow.__version__)
print("python:", sys.executable)
EOF

cd ~/projects/def-benliang/kalarde/TTL
export DISABLE_VERSION_CHECK=1
export PYTHONPATH=$PWD/src:$PYTHONPATH



CUDA_VISIBLE_DEVICES=0 python -m llamafactory.cli train examples/train_lora/inferenceonly_llama31_meta_math.yaml
