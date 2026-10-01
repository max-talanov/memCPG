#!/bin/bash
cd "$(dirname "$0")" || exit 1

INJURY=0.99  # calibrated with calibrate_injury.py: <=0.95 leaves RG_E intact

submit() {  # name, mode variables
  sbatch --job-name="cpg_$1" --output="cpg_$1.slurmout" --error="cpg_$1.slurmerr" \
         --export="ALL,CPG_OUT=res_$1,$2" --wrap="srun python3 main_cpg.py"
}

submit intact "CPG_SPEED=100,CPG_BWS=0,CPG_INJURY=0"
submit slow   "CPG_SPEED=125,CPG_BWS=0,CPG_INJURY=$INJURY"
submit medium "CPG_SPEED=100,CPG_BWS=0,CPG_INJURY=$INJURY"
submit fast   "CPG_SPEED=50,CPG_BWS=0,CPG_INJURY=$INJURY"
submit toe    "CPG_SPEED=100,CPG_BWS=0.5,CPG_INJURY=$INJURY"
