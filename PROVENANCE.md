# Provenance — data and code used by this project

Record of every external dataset and piece of code used (read, copied,
executed, ported), so that owners can be informed and acknowledged.
Update this file whenever something new is used.

Rule: folders outside this project are **read only**. What is needed is copied
here (data: one test subject only).

## Code

| What | Origin | Owner / authors | How it is used | Date |
|---|---|---|---|---|
| vbPET MATLAB code (`UTILITY_VB/*.m`, `MAIN_parametricMaps_VB.m`, `README.txt`) | `/nfsd/biopetmri4/Codes/PET/MODELS/VB_package/` (+ `README.txt` from `.../MODELS/UTILITY_VB/`) | Gaia Rizzo (GR), EG; Castellaro et al. 2017 | Copied to `third_party/vbPET_matlab/` (SHA256SUMS there); algorithm ported to Python (`vbpet/`); executed with MATLAB R2026b by `tests/test_matlab_compat.py` as numerical reference | 2026-11-03 |
| `.../MODELS/UTILITY_VB/` (older copy of the same code) | `/nfsd/biopetmri4/Codes/PET/MODELS/UTILITY_VB/` | as above | Read only (compared with VB_package: VB_package is newer, adds 1TCM) | 2026-11-03 |
| fabber `spatialvb` + VB_BBB fitting strategy | `/nfsd/biopetmri4/Users/Marco/Projects/VB_BBB/` (`fabber_dev/inference_spatialvb.cc`, `DESIGN_*.md`, `run_*.sh`, `Real_data/fabber_fit/STRATEGIE_FITTING.md`) | FSL fabber (Chappell, Groves et al.); VB_BBB project (M. Castellaro) | Read only: design of the spatial prior (prior types, two-step warm start) | 2026-11-03 |
| GitHub `marcocastellaro/vbPET` | https://github.com/marcocastellaro/vbPET (commit 4186687) | M. Castellaro | README only (no code in the repo); shallow clone in a temp folder | 2026-11-03 |
| FSL `flirt`/`convert_xfm` | `/nfsd/opt/FSL/fsl607` on DEI | FMRIB, Oxford | Executed by `scripts/prep_dseg_to_pet.sh` (dseg → PET grid) | 2026-11-03 |

## Data

| Dataset | Origin | Licence | What was copied | Where | Date |
|---|---|---|---|---|---|
| OpenNeuro ds004869 v1.4.1 — PET quantification of COX-2 in healthy humans ([11C]MC1) | https://openneuro.org/datasets/ds004869 , doi:10.18112/openneuro.ds004869.v1.4.1 | CC0 | **sub-01 only**: `ses-baseline/pet` (PET, json, manual blood), `anat/T1w`, derivatives for sub-01 ses-baseline: `petprep` (anat dseg/T1w/mask), `bloodstream` (input function), `petfit` (regional kinpar/TACs); top-level json/tsv/README/LICENSE. File list: `testdata/ds004869_files.txt` | `testdata/ds004869/` (~200 MB) | 2026-11-03 |

Authors (ds004869): Xuefeng Yan, Martin Noergaard, Cheryl L. Morse, Jeih-San
Liow, Jinsoo Hong, et al. — cite the dataset DOI in any output.

Derived here from ds004869 (not original data): `testdata/ds004869/derivatives/vbpet-prep/`
(dseg resampled to PET), `testdata/out/` (vbpet outputs).

## Candidates looked at, not downloaded

- OpenNeuro ds001420 ([11C]DASB Cimbi example): file list only.
- Zenodo 16364694 / 16686025 QUADRA_HC (total-body FDG, static only, CC-BY 4.0): metadata only.
- "Multimodal total-body dynamic [18F]FDG PET/CT/MRI dataset of 100 healthy humans" (Sci Data 2026): paper only; repository/access not yet known.
