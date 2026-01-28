#!/bin/bash
#SBATCH --job-name=one
#SBATCH --nodes=1
#SBATCH --time=00:00:40
#SBATCH --account=def-ssanner
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
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
python scripts/eval/eval_accuracy.py