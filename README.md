# vbPET
Variational Bayesian estimator for Positron Emission Tomography imaging

This toolbox provides a Variational Bayesian estimator for the quantification of Positron Emission Tomography data. It includes several model used for kinetic modeling. It is based on MATLAB and require the use of Parallel Toolbox to speedup the estimation. Please cite [1] if you use this toolbox in your research.

[1] **M. Castellaro**, G. Rizzo, M. Tonietto, M. Veronese, F.E. Turkheimer, M.A. Chappell and A. Bertoldo “[A Variational Bayesian inference method for parametric imaging of PET data][paper1]”, *NeuroImage, 150:136-149 (2017)*.


## Installation

Matlab:

```
git clone https://github.com/marcocastellaro/vbPET
addpath('vbUtils')
```

## Example

## Demo

## Python / JAX implementation (work in progress, branch `python-port`)

`vbpet/` is a Python re-implementation of the estimator on JAX (batched over
voxels, CPU or GPU), packaged as a BIDS App:

```
pip install -e .            # GPU: pip install -e ".[cuda]"
vbpet <bids_dir> <output_dir> participant --model 2TCM \
      --blood-derivative <bids_dir>/derivatives/bloodstream \
      --derivatives <dseg_derivative_dir> --dseg-desc gtm
```

Discretisation: `--frame-sampling mean|mid` (model averaged over each frame,
default, or taken at mid-frame as in the MATLAB code and kinfitr/petfit),
`--convolution linear|rect`, `--input-step`; `--matlab-compat` sets all three to
the MATLAB behaviour.

Stage 1 fits the regional TACs of a PET-space segmentation (WNLLS) and builds
region-based priors; stage 2 runs voxel-wise VB with those priors. Outputs are
BIDS derivatives (`*_kinpar.tsv`, `*_tacs.tsv`, `*_model-<M>_meas-<p>_mimap.nii.gz`).
See `DESIGN.md` for the design (spatial priors, whole-body PET) and
`PROVENANCE.md` for the code and data used during development.

## Release History

* 0.0.1
    * Work in progress

## Meta

Marco Castellaro – [@marcocaste84](https://twitter.com/marcocaste84) – marco.castellaro@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/marcocastellaro/vbPET](https://github.com/marcocastellaro/vbPET)

## Contributing

1. Fork it (<https://github.com/marcocastellaro/vbPET/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

[//]: # (reference links)

   [paper1]: <https://www.sciencedirect.com/science/article/pii/S1053811917301143>
