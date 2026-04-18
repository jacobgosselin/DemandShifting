#!/bin/bash
#SBATCH --job-name=demandshift_run_alt_paths
#SBATCH -A p32386
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/run_alt_paths_%j.out
#SBATCH --error=logs/run_alt_paths_%j.err

module purge
module load mamba
source activate demandshifting

# Prevent numpy/BLAS/Numba from spawning extra threads that conflict with
# the multiprocessing Pool workers used in solve_phi_path.py
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

mkdir -p logs
mkdir -p solved_eqm
cd $SLURM_SUBMIT_DIR
/home/jtg3519/.conda/envs/demandshifting/bin/python3 -u run_sigma_path.py
/home/jtg3519/.conda/envs/demandshifting/bin/python3 -u run_sigma_eps_path.py

