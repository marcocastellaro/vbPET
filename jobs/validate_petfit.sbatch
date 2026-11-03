#!/bin/bash
# Regional 2TCM vs petfit on ds004869 sub-01 (12 TACs), CPU
#SBATCH --job-name=vbpet_valpetfit
#SBATCH --partition=allgroups
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/nfsd/biopetmri4/Users/Marco/code/vb_pet/jobs/logs/%x_%j.log
set -euo pipefail
ROOT=/nfsd/biopetmri4/Users/Marco/code/vb_pet
JAX_PLATFORMS=cpu $ROOT/env/vbpet/bin/python $ROOT/tests/validate_petfit.py
