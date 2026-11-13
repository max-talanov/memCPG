#!/bin/bash -l
#SBATCH --job-name=Neuron_calc
#SBATCH --output=Neuron_calculation.slurmout
#SBATCH --error=Neuron_calculation.slurmerr
set -euo pipefail
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
srun nrniv -mpi -python main_cpg.py

