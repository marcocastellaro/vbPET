# vbPET-py — design document

Python/JAX re-implementation of the vbPET Variational Bayes estimator
(Castellaro et al., NeuroImage 150:136-149, 2017), packaged as a BIDS App, with
a region-based prior pre-step and optional spatial priors. Target: brain **and
whole-body / total-body** dynamic PET.

Status: draft, 2026-11-03.

---

## 1. Sources

| Source | What is taken from it |
|---|---|
| `/nfsd/biopetmri4/Codes/PET/MODELS/VB_package/` (MATLAB, GR/EG 2015-16) | Reference algorithm: `VB_estimation.m` (ROI WNLLS → prior → voxel VB), `vbEstimation.m`, `trialsMethodW.m`, `updateVB.m`, `freeEnergy.m`, `computeModel_*.m`, `Derived_Par_*.m`, `wnllsEstimation.m`, `MAIN_parametricMaps_VB.m` |
| GitHub `marcocastellaro/vbPET` | README only (no code) |
| `Projects/VB_BBB` (fabber `spatialvb`) | Spatial prior design: per-parameter prior types (`M` MRF, `I` image prior, `A` ARD, `N`), two-step fit (VB → spatialVB warm start, a cold start diverges), image priors with high precision to propagate stage-1 values |

### 1.1 MATLAB algorithm (what must be reproduced)

For each ROI (cluster) of an atlas:

1. ROI TAC → WNLLS fit (`lsqnonlin`, bounds `[0, Inf)`), weights
   `w = sqrt(Δt / C_roi)` (0 where C ≤ 0).
2. Prior: `m0 = θ_ROI`, `L0 = diag(1 / (λ·m0)²)`, λ = 0.5.
3. For every voxel of the ROI: VB with
   - noise precision ~ Gamma(s, c), prior s0 = 1e6, c0 = 50;
   - `sd = 1/w`, data ≤ 0 get sd = Inf;
   - update (`updateVB`): `c = (n-1)/2 + c0`,
     `s = 1/(1/s0 + (kᵀΣ⁻¹k + tr(L⁻¹JᵀΣ⁻¹J))/2)`, `L = sc·JᵀΣ⁻¹J + L0`,
     `m = L⁻¹(sc·JᵀΣ⁻¹(k + J m_old) + L0 m0)`, then clip `m ≥ 0`;
   - numerical Jacobian (forward diff, δ = max(1e-6|m|, 1e-12));
   - stop: 100 iterations, |ΔF| < 1e-3, or 10 consecutive non-increasing F;
     return the iterate with **maximum F**;
   - optional ARD on one parameter (prior mean 0, precision re-estimated).
4. Derived parameters (VT, Ki, ...) with delta-method CV.

Models: 1TCM, 2TCM, 2TCM-1K, 3K (irreversible 2TCM), 1K, 5K; all with Vb and
whole-blood Cb; plasma input interpolated on a uniform grid (Δ = t_peak/3),
convolution via `filter`, output sampled at mid-frame (`interp1`).

Why it is slow: serial voxel loop, numerical Jacobian (P+1 model calls per
iteration), `pinv` everywhere, `interp1`/`filter` per call.

---

## 2. Architecture

```
vbpet/
  cli.py              BIDS-App entry point (bids_dir output_dir participant ...)
  io/bids.py          run discovery (pybids), frame timing, blood/IDIF, dseg, labels
  io/derivatives.py   output naming + JSON sidecars + dataset_description
  input.py            input functions: plasma×parent fraction, whole blood, IDIF,
                      reference region TAC; uniform fine grid; delay shift
  models/
    base.py           Model protocol: params, bounds, p0, predict(θ, ctx), derived(θ, Σ)
    plasma.py         1TCM, 2TCM, irr2TCM(3K), 2TCM-1K, 1K, 5K, Patlak  (+Vb, +delay)
    reference.py      SRTM, SRTM2 (k2' fixed from regional fit)
    conv.py           exact exp ⊗ piecewise-linear input (O(N) scan), frame integration
  vb/
    core.py           batched VB (vmap over voxels): updateVB + freeEnergy + trials logic
    spatial.py        spatial (MRF) prior on deviations from the regional prior
    ard.py
  regional.py         pre-step: ROI TACs → WNLLS (batched JAX LM) → prior spec
  voxel.py            stage 2: chunked voxel VB with regional (± spatial) priors
tests/                unit tests + MATLAB comparison fixtures
containers/           Dockerfile (CPU), Dockerfile.cuda, apptainer.def
```

### 2.1 Speed-ups (vs MATLAB)

- `jax.vmap` over voxels + `jax.jit`, chunks of N voxels (`--chunk-size`,
  auto from available GPU/host memory) streamed through GPU/CPU.
- Exact Jacobian with `jax.jacfwd` (one forward pass with P tangents instead of
  P+1 model calls; no step-size issues).
- Convolution with exponentials computed by a recursive scan over a uniform
  grid (exact for piecewise-linear input), shared across voxels; frame means
  from the cumulative integral (option `--frame-sampling mid` for exact MATLAB
  compatibility).
- `pinv` → Cholesky solve with jitter (fallback to `pinv` only for flagged
  voxels).
- Fixed-length `lax.fori_loop` with per-voxel "converged/stalled" masks that
  reproduce the MATLAB stopping rules and the keep-best-F selection.
- Patlak (and other linearisable models) solved in closed form, batched.

Target: whole-brain 2TCM (~2e5 voxels) in minutes on the A2000; total-body
(~1e7 body voxels) on a DEI A40/L40S in < 1 h.

### 2.2 Region-based prior pre-step (`--stage regional`)

Input: dynamic PET + `dseg` **already in PET space** (BIDS derivative, e.g.
PETPrep/PETSurfer, TotalSegmentator/MOOSE resampled to PET) + labels TSV.

1. ROI TACs (mean, n voxels, SD across voxels) → `*_desc-regional_tacs.tsv`.
2. Weights: MATLAB formula (`sqrt(Δt/C)`) by default, `--weights` alternatives
   (uniform, frame duration, decay-corrected counts if available).
3. Batched WNLLS (Levenberg-Marquardt in JAX, multi-start from p0 grid) for all
   ROIs → `*_model-<M>_desc-regional_kinpar.tsv` with estimates, SE, CV,
   derived parameters, WRSS/AIC/BIC.
4. Prior spec → `*_model-<M>_desc-prior_kinpar.tsv/.json`:
   mean = ROI estimate, sd = λ·mean (λ default 0.5, as in MATLAB), optional
   floor on sd, optional user-supplied priors (literature tables) for ROIs where
   the fit fails (CV above threshold).
   This file is the interface to stage 2, so priors can also come from another
   tool or population values.
5. Reference-region models: the reference TAC also comes from the dseg
   (`--ref-label`); SRTM2 takes k2' from the regional SRTM fit of
   high-binding regions (median).

### 2.3 Voxel VB (`--stage voxel`)

Per voxel v, parameter k, the prior mean/precision come from the ROI the voxel
belongs to (image prior, like fabber type `I`). Prior type per parameter
(fabber-like string, `--prior-types`, e.g. `RRRS`):

| Code | Prior | Notes |
|---|---|---|
| `R` | regional: N(μ_ROI, (λ μ_ROI)²) | MATLAB behaviour (default) |
| `S` | regional mean + spatial MRF on deviations δ = θ - μ_ROI | φ_k learned (VB Gamma update), 6/26-neighbourhood, edges not crossing ROI borders optional |
| `N` | broad Gaussian | |
| `A` | ARD | as MATLAB (one parameter) and fabber |

Spatial prior (type `S`): p(δ_k) ∝ exp(-φ_k/2 · δ_kᵀ(Λ + εI)δ_k), Λ the graph
Laplacian of the mask. Mean-field: each voxel sees a Gaussian prior with mean
μ_ROI + mean of its neighbours' deviations and precision φ_k(deg + ε); φ_k gets
a Gamma posterior updated after each sweep. As in VB_BBB, stage 2 starts from
the non-spatial VB solution (two-step), since a cold spatial start diverges.

Outputs: `*_model-<M>_meas-<p>_mimap.nii.gz` (posterior mean),
`*_model-<M>_meas-<p>_desc-sd_mimap.nii.gz`, derived parameters (VT/Ki/BPND)
with sd, `meas-FreeEnergy`, `meas-noiseSD`, optional fitted 4D
(`desc-fit_pet.nii.gz`), JSON sidecars with model, priors and software version.

### 2.4 Whole-body / total-body extensions

- **Input function**: IDIF from a dseg label (aorta / left ventricle) when no
  `_blood.tsv`; IDIF TSV from derivatives if present; optional
  plasma/whole-blood ratio and population parent fraction.
- **Per-voxel delay** `dt` (and optional dispersion) as an inferred parameter:
  essential with a long axial FOV (arrival time differs head-to-feet).
- **Organ priors** from TotalSegmentator/MOOSE dseg (same pre-step).
- **FDG**: irreversible 2TCM (K1, k2, k3, Vb, dt) + Patlak Ki.
- Later: dual-input liver model (hepatic artery + portal vein), reversible
  models for other tracers.
- **Memory**: total-body matrices (~440×440×645×N frames) are processed in
  voxel chunks read from a memory-mapped array; never the full 4D in device
  memory.

### 2.5 BIDS App

```
vbpet <bids_dir> <output_dir> participant
      [--participant-label 01 02] [--session-label baseline] [--task/--trc ...]
      [--stage regional|voxel|all]
      [--model 2TCM|1TCM|irr2TCM|Patlak|SRTM|SRTM2] [--fit-delay] [--fit-vb]
      [--input blood|idif|reference] [--blood-derivative bloodstream]
      [--dseg-derivative petprep] [--dseg-desc ...] [--ref-label ...]
      [--prior-types RRRS] [--prior-lambda 0.5] [--priors-tsv file]
      [--device cpu|gpu] [--chunk-size auto] [--frame-sampling mean|mid]
```

Containers: Docker CPU image, CUDA image; Apptainer for DEI (`--nv`, built
and run inside sbatch: apptainer only exists on DEI compute nodes; local
Docker has no GPU passthrough).

---

## 3. Data

| Purpose | Dataset | Notes |
|---|---|---|
| Brain, plasma input + reference | OpenNeuro **ds004869** (v1.4.1, 27 subj., 54 PET) | raw dynamic PET, `recording-manual_blood.tsv`, `bloodstream` input functions, PETPrep `seg-gtm_dseg` (T1w space), `petfit` regional 2TCM/Logan/MA1 → reference values for the regional pre-step |
| Brain, small | OpenNeuro ds001420 ([11C]DASB, 2 subj.) | no blood; FreeSurfer only; reference-region tests |
| Whole-body dynamic | "Multimodal total-body dynamic [18F]FDG PET/CT/MRI dataset of 100 healthy humans" (Sci Data 2026, Quadra, BIDS, organ TACs + Patlak Ki) | repository / access conditions to be checked |
| Whole-body segmentations | Zenodo 16364694 QUADRA_HC (48 HC, CC-BY, 20 GB) | **static only** (57-62 min) + MOOSE 135-region segs: used to build realistic synthetic dynamic phantoms |
| Synthetic | generated here | known ground truth; brain and whole-body phantoms |

The dseg is in T1w space in ds004869 (no `space-pet` output in the shared
PETPrep), so a small prep script (outside the app) registers the PET mean to
the T1w (FSL/ANTs) and resamples the dseg into PET space, writing a
`space-pet` dseg derivative.

---

## 4. Validation plan

1. Model functions vs MATLAB `computeModel_*` (MATLAB is available locally)
   on identical inputs, `--frame-sampling mid`.
2. Per-voxel VB vs MATLAB `vbEstimation` on a few thousand voxels (same priors,
   numerical Jacobian for comparison): agreement of m, sd, F.
3. Speed benchmark vs MATLAB (same voxels).
4. ds004869: regional 2TCM vs `petfit`; voxel VT vs PETSurfer Logan VT.
5. Synthetic phantoms (brain + whole-body): bias/variance, effect of spatial
   prior (S vs R), delay recovery.

## 5. Milestones

| # | Content |
|---|---|
| M1 | Core: conv, plasma models, batched VB (non-spatial), MATLAB comparison tests |
| M2 | BIDS I/O, regional pre-step, voxel stage, derivatives, CLI |
| M3 | Reference-region models (SRTM, SRTM2) |
| M4 | Spatial prior (S) + two-step |
| M5 | Whole-body: IDIF, delay, chunked memory-mapped streaming, synthetic total-body phantom |
| M6 | Containers (Docker/Apptainer), DEI GPU run, docs |
