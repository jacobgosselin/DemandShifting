# Running calibration on Quest

## Contents of this folder
- `calibrate_investment_params.py` — two-phase calibration (DE → least_squares)
- `solve_eqm.py`, `solve_vf.py`, `integrate_dist.py`, `prod_fncts.py` — model solvers
- `structural_parameters.csv`, `sales_elasticity_m_by_year.csv` — input data
- `run_calibration.slurm` — SLURM job script (account p32386, 20 CPUs, 4h)

## Steps

### 1. Upload
```bash
scp -r quest/ <netid>@quest.northwestern.edu:~/demandshift/
```

### 2. Log in and submit
```bash
ssh <netid>@quest.northwestern.edu
cd ~/demandshift
sbatch run_calibration.slurm
```

### 3. Monitor
```bash
squeue -u <netid>          # check job status
tail -f logs/cal_<jobid>.out   # follow live output
```

### 4. Retrieve results
```bash
# from your local machine:
scp <netid>@quest.northwestern.edu:~/demandshift/calibrated_investment_params.csv .
```

## Notes
- The job requests 20 CPUs on 1 node. `differential_evolution(workers=-1)` will use
  all 20 for parallel population evaluation (~45 candidates per generation).
- Wall-time estimate: 2–4 hours for DE (300 generations) + ~10 min for LS refinement.
- If the job needs more than 4 hours, change `--partition=short` to `--partition=normal`
  and increase `--time` (max 48:00:00 on normal).
- Output `calibrated_investment_params.csv` is written to the same directory as the script.
