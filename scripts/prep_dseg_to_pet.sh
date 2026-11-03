#!/bin/bash
# Resample an anatomical dseg (T1w space) onto the PET grid, as a BIDS derivative.
#
#   prep_dseg_to_pet.sh <pet.nii.gz> <T1w.nii.gz> <dseg.nii.gz> <out_dir> [seg_label]
#
# Rigid PET->T1w registration (FSL flirt, 6 dof, normalised MI) of the
# duration-weighted mean PET, then nearest-neighbour resampling of the dseg with
# the inverse transform. Writes <prefix>_space-pet_seg-<label>_dseg.nii.gz (+ .tsv
# copied from the source dseg) under <out_dir>/sub-X[/ses-Y]/pet/.
# Not part of the BIDS App: the app expects the dseg already on the PET grid.
set -euo pipefail
PET=$1; T1=$2; DSEG=$3; OUT=$4; SEG=${5:-gtm}
: "${FSLDIR:?set FSLDIR to your FSL installation}"
export FSLDIR FSLOUTPUTTYPE=NIFTI_GZ
PY=${PY:-python}

base=$(basename "$PET"); prefix=${base%_pet.nii*}
sub=$(echo "$prefix" | grep -o 'sub-[a-zA-Z0-9]*'); ses=$(echo "$prefix" | grep -o 'ses-[a-zA-Z0-9]*' || true)
odir=$OUT/$sub${ses:+/$ses}/pet; mkdir -p "$odir"
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT

"$PY" - "$PET" "$work/mean.nii.gz" <<'EOF'
import json, sys, nibabel as nib, numpy as np
pet, out = sys.argv[1], sys.argv[2]
img = nib.load(pet)
meta = json.load(open(pet.split('.nii')[0] + '.json'))
dur = np.asarray(meta['FrameDuration'], float)
data = np.asarray(img.dataobj, np.float32)
mean = (data * dur).sum(-1) / dur.sum()
nib.save(nib.Nifti1Image(mean.astype(np.float32), img.affine), out)
EOF

"$FSLDIR/bin/flirt" -in "$work/mean.nii.gz" -ref "$T1" -dof 6 -cost normmi \
    -searchrx -30 30 -searchry -30 30 -searchrz -30 30 -omat "$odir/${prefix}_from-pet_to-T1w_mode-image_xfm.mat" \
    -out "$work/mean_in_t1.nii.gz"
"$FSLDIR/bin/convert_xfm" -omat "$work/t1_to_pet.mat" -inverse "$odir/${prefix}_from-pet_to-T1w_mode-image_xfm.mat"
dseg_out=$odir/${prefix}_space-pet_seg-${SEG}_dseg.nii.gz
"$FSLDIR/bin/flirt" -in "$DSEG" -ref "$work/mean.nii.gz" -applyxfm -init "$work/t1_to_pet.mat" \
    -interp nearestneighbour -out "$dseg_out"
# flirt writes float; store integer labels on exactly the PET header grid
"$PY" - "$dseg_out" "$PET" <<'EOF'
import sys, nibabel as nib, numpy as np
d, pet = sys.argv[1], sys.argv[2]
ref = nib.load(pet)
lab = np.rint(np.asarray(nib.load(d).dataobj)).astype(np.int16)
nib.save(nib.Nifti1Image(lab, ref.affine), d)
EOF
cp "${DSEG%.nii*}.tsv" "${dseg_out%.nii*}.tsv"
cp "$work/mean.nii.gz" "$odir/${prefix}_desc-mean_pet.nii.gz"
echo "{\"Sources\": [\"$DSEG\", \"$PET\"], \"Registration\": \"FSL flirt 6dof normmi, mean PET -> T1w, NN resampling\"}" \
    > "${dseg_out%.nii*}.json"
echo "wrote $dseg_out"
