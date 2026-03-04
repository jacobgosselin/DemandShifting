#!/bin/bash
#SBATCH --job-name=demandshift_cal
#SBATCH -A p32386
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cal_%j.out
#SBATCH --error=logs/cal_%j.err

module purge
module load mamba
source activate demandshifting

# Prevent numpy/BLAS from spawning extra threads that would conflict with
# the multiprocessing workers used by differential_evolution(workers=-1)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p logs
cd $SLURM_SUBMIT_DIR
/home/jtg3519/.conda/envs/demandshifting/bin/python3 calibrate_investment_params.py
