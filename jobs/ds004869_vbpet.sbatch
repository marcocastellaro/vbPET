#!/bin/bash
# ds004869 sub-01 baseline: vbpet BIDS App (regional + voxel 2TCM), GPU
#SBATCH --job-name=vbpet_ds004869
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:rtx:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/nfsd/biopetmri4/Users/Marco/code/vb_pet/jobs/logs/%x_%j.log
set -euo pipefail
ROOT=/nfsd/biopetmri4/Users/Marco/code/vb_pet
DS=$ROOT/testdata/ds004869
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$ROOT/env/vbpet/bin/vbpet $DS $ROOT/testdata/out/ds004869 participant \
  --participant-label 01 --session-label baseline --model ${MODEL:-2TCM} \
  --blood-derivative $DS/derivatives/bloodstream \
  --derivatives $DS/derivatives/vbpet-prep --dseg-desc gtm \
  --device gpu --precision ${PREC:-float64} ${EXTRA:-}
