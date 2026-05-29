#!/bin/bash
#SBATCH --job-name=Toy_Compsep
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=gpu_ens_vlong
#SBATCH --nodelist=node0257
#SBATCH --output=/obs/cdussauchoy/STL-Dev_Camille/Toy_CompSepPatchbis.log
#SBATCH --error=/obs/cdussauchoy/STL-Dev_Camille/Toy_CompSepPatchbis.err

set -euo pipefail

module purge
module load anaconda/3-2023.07
module load cuda/12.6
source /shared/apps/anaconda/3-2023.07/etc/profile.d/conda.sh
conda activate mon_env

cd "/obs/cdussauchoy/STL-Dev_Camille"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# Runtime.
export COMPSEP_DEVICE="cuda:0"
export COMPSEP_DTYPE="float64"
export COMPSEP_SEED="2"
export COMPSEP_OUTDIR="results_toyPatchbis"

# Toy problem size and stochastic optimizer sampling.
export COMPSEP_N_NOISE="500"
export COMPSEP_BATCH="1"
export COMPSEP_ITERS="500"

# Optimizer.
export COMPSEP_OPTIMIZER_LR="1"

# STL/statistic controls.
export COMPSEP_WTYPE="Bump-Steerable"
export COMPSEP_PBC="True"
export COMPSEP_ST_COMPUTE_PS="False"

# Aliases understood by the Python script, kept here for consistency with the
# Planck launcher naming.
export PBC="$COMPSEP_PBC"
export ST_COMPUTE_PS="$COMPSEP_ST_COMPUTE_PS"

# Fourier post-processing. This is applied only after optimization.
export COMPSEP_APPLY_NYQUIST_AFTER="True"


echo "Python utilisé : $HOME/.conda/envs/mon_env/bin/python3"
$HOME/.conda/envs/mon_env/bin/python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
/obs/cdussauchoy/.conda/envs/mon_env/bin/python3 -u toy_compsep.py
