#!/bin/bash
#SBATCH --job-name=one
#SBATCH --nodes=1
#SBATCH --time=00:00:30
#SBATCH --account=rrg-ssanner
#SBATCH --gpus-per-node=h100:1
echo 'Hello, world!'
sleep 10