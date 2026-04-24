#!/bin/bash
#SBATCH --job-name=demandshift_phi_cal
#SBATCH -A p32386
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phi_cal_%j.out
#SBATCH --error=logs/phi_cal_%j.err

module purge
module load mamba
source activate demandshifting

# Prevent numpy/BLAS from spawning extra threads that would conflict with
# the multiprocessing workers
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p logs
cd $SLURM_SUBMIT_DIR
/home/jtg3519/.conda/envs/demandshifting/bin/python3 ../calibrate_phi_path.py
