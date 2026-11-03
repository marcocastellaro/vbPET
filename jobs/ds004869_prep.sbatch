#!/bin/bash
# ds004869 sub-01: gtm dseg (T1w) -> PET grid (FSL flirt), CPU
#SBATCH --job-name=vbpet_prep
#SBATCH --partition=allgroups
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/nfsd/biopetmri4/Users/Marco/code/vb_pet/jobs/logs/%x_%j.log
set -euo pipefail
ROOT=/nfsd/biopetmri4/Users/Marco/code/vb_pet
DS=$ROOT/testdata/ds004869
PP=$DS/derivatives/petprep/sub-01/anat
bash $ROOT/scripts/prep_dseg_to_pet.sh \
  $DS/sub-01/ses-baseline/pet/sub-01_ses-baseline_pet.nii.gz \
  $PP/sub-01_desc-preproc_T1w.nii.gz $PP/sub-01_desc-preproc_seg-gtm_dseg.nii.gz \
  $DS/derivatives/vbpet-prep gtm
