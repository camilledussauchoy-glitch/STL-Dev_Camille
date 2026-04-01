#!/bin/bash
#SBATCH --job-name=run_benchmark
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=gpu_ens_vlong

python benchmark_1.py

rm -f slurm-*.out