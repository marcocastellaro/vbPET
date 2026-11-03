"""BIDS App entry point.

    vbpet <bids_dir> <output_dir> participant [options]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger("vbpet")


def build_parser():
    p = argparse.ArgumentParser(
        prog="vbpet", description="Variational Bayes parametric imaging of dynamic PET (BIDS App)")
    p.add_argument("bids_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("analysis_level", choices=["participant"])
    g = p.add_argument_group("selection")
    g.add_argument("--participant-label", nargs="+")
    g.add_argument("--session-label", nargs="+")
    g.add_argument("--tracer", help="trc- entity filter")
    g = p.add_argument_group("model")
    g.add_argument("--stage", choices=["regional", "voxel", "all"], default="all")
    g.add_argument("--model", default="2TCM")
    g.add_argument("--input", choices=["blood"], default="blood",
                   help="input function source (reference/idif: planned)")
    g.add_argument("--blood-derivative", type=Path,
                   help="derivative folder with *_inputfunction.tsv (e.g. bloodstream)")
    g.add_argument("--derivatives", type=Path, nargs="+", default=[],
                   help="derivative folders searched for the PET-space dseg")
    g.add_argument("--dseg-desc", help="desc-/seg- entity of the dseg to use")
    g.add_argument("--dseg-space", help="space- entity of the dseg (default: any on the PET grid)")
    g.add_argument("--min-roi-voxels", type=int, default=10)
    g.add_argument("--weights", choices=["matlab", "duration", "uniform"], default="matlab")
    g.add_argument("--prior-lambda", type=float, default=0.5)
    g.add_argument("--priors-tsv", type=Path, help="use this prior table instead of stage 1")
    g.add_argument("--frame-sampling", choices=["mean", "mid"],
                   help="model value per frame: mean = average over the frame (default), "
                        "mid = value at mid-frame (MATLAB vbPET, kinfitr/petfit)")
    g.add_argument("--convolution", choices=["linear", "rect"],
                   help="input convolution: linear = exact for a piecewise-linear input "
                        "(default), rect = rectangle rule (MATLAB filter)")
    g.add_argument("--input-step",
                   help="input grid step: auto (t_peak/3 clipped to 1-10 s, default) | "
                        "matlab (t_peak/3) | <minutes>")
    g.add_argument("--matlab-compat", action="store_true",
                   help="defaults as the MATLAB code: --convolution rect --frame-sampling mid "
                        "--input-step matlab (each can still be set explicitly)")
    g = p.add_argument_group("VB")
    g.add_argument("--s0", type=float, default=1e6)
    g.add_argument("--c0", type=float, default=50.0)
    g.add_argument("--max-iter", type=int, default=100)
    g.add_argument("--ard", help="parameter name with ARD prior (e.g. Vb)")
    g = p.add_argument_group("compute")
    g.add_argument("--device", choices=["cpu", "gpu", "auto"], default="auto")
    g.add_argument("--precision", choices=["float32", "float64"], default="float64")
    g.add_argument("--chunk-size", type=int, default=16384)
    g.add_argument("--max-voxels", type=int, help="debug: fit only the first N voxels")
    g.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    import jax
    if args.device != "auto":
        jax.config.update("jax_platforms", args.device if args.device == "cpu" else "cuda")
    log.info("JAX devices: %s", jax.devices())

    from .io.bids import find_pet_runs
    from .io.derivatives import write_dataset_description

    filters = {"trc": args.tracer} if args.tracer else None
    runs = find_pet_runs(args.bids_dir, args.participant_label, args.session_label, filters)
    if not runs:
        log.error("no PET runs found")
        return 1
    write_dataset_description(args.output_dir, args.bids_dir)
    for run in runs:
        log.info("=== %s", run.path.name)
        process_run(run, args)
    return 0


def process_run(run, args):
    import jax.numpy as jnp
    import nibabel as nib
    import numpy as np
    import pandas as pd

    from .input import make_plasma_input
    from .io.bids import find_dseg, load_blood, read_labels
    from .io.derivatives import RunWriter
    from .models.base import Frames
    from .models.plasma import get_model
    from .regional import fit_regions, frame_weights, make_priors, roi_tacs
    from .vb.core import VBOptions
    from .voxel import fit_voxels, to_volume, voxel_priors

    w = RunWriter(args.output_dir, run)
    mc = args.matlab_compat
    conv = args.convolution or ("rect" if mc else "linear")
    sampling = args.frame_sampling or ("mid" if mc else "mean")
    step = args.input_step or ("matlab" if mc else "auto")
    try:
        step = float(step)
    except ValueError:
        pass
    model = get_model(args.model, conv=conv, sampling=sampling)

    # --- data (kBq/mL, minutes)
    img = nib.load(str(run.path))
    data = np.asarray(img.dataobj, dtype=np.float32) * (run.unit_factor / 1e3)
    frames = Frames.from_start_duration(run.frame_start_min, run.frame_duration_min)
    t_end = float(frames.end[-1])
    blood = load_blood(run, args.blood_derivative)
    inp = make_plasma_input(blood, t_end, step=step, scale=1e-3)
    log.info("input: %s (grid %d pts, dt=%.3f min)", blood.source, len(inp.tp),
             float(inp.tp[1] - inp.tp[0]))

    dseg_path = find_dseg(run, args.derivatives, args.dseg_desc, args.dseg_space)
    dseg = np.asarray(nib.load(str(dseg_path)).dataobj).astype(int)
    labels = read_labels(dseg_path)
    log.info("dseg: %s (%d labels)", dseg_path, len(np.unique(dseg)) - 1)
    common = dict(Model=model.name, Convolution=conv, FrameSampling=sampling,
                  InputSource=blood.source, Segmentation=str(dseg_path),
                  InputGridStepMin=float(inp.tp[1] - inp.tp[0]),
                  TimeUnits="min", ActivityUnits="kBq/mL")

    # --- stage 1
    tacs, roi_info = roi_tacs(data, dseg, labels, args.min_roi_voxels)
    dur = np.asarray(frames.end - frames.start)
    weights = pd.DataFrame({c: frame_weights(tacs[c], dur, args.weights) for c in tacs})
    if args.stage in ("regional", "all") or args.priors_tsv is None:
        t0 = time.perf_counter()
        tac_df = pd.concat([pd.DataFrame({"frame_start": np.asarray(frames.start),
                                          "frame_end": np.asarray(frames.end)}), tacs], axis=1)
        w.tsv(tac_df, "desc-dseg", suffix="tacs", meta=common)
        w.tsv(roi_info, "desc-dseg", suffix="morph")
        kin, fits = fit_regions(model, inp, frames, tacs, weights)
        w.tsv(kin, f"model-{model.name}", "desc-regional", suffix="kinpar",
              meta={**common, "Weights": args.weights})
        priors = make_priors(kin, model, lam=args.prior_lambda)
        w.tsv(priors, f"model-{model.name}", "desc-prior", suffix="kinpar",
              meta={**common, "PriorLambda": args.prior_lambda})
        log.info("regional fits: %d ROIs in %.1f s", len(kin), time.perf_counter() - t0)
    if args.priors_tsv is not None:
        priors = pd.read_csv(args.priors_tsv, sep="\t")
    if args.stage == "regional":
        return

    # --- stage 2
    mask = dseg > 0
    idx, m0, sd, names = voxel_priors(dseg, roi_info, priors, model, mask)
    if args.max_voxels:
        idx, m0, sd, names = idx[:args.max_voxels], m0[:args.max_voxels], sd[:args.max_voxels], \
            names[:args.max_voxels]
    ard = model.param_names.index(args.ard) if args.ard else -1
    opts = VBOptions(s0=args.s0, c0=args.c0, max_iter=args.max_iter, ard=ard)
    dtype = jnp.float32 if args.precision == "float32" else jnp.float64
    w_roi = {c: weights[c].to_numpy() for c in weights}
    t0 = time.perf_counter()
    last = [0.0]

    def progress(done, total):
        if time.perf_counter() - last[0] > 30 or done == total:
            last[0] = time.perf_counter()
            log.info("voxels %d/%d (%.0f vox/s)", done, total, done / (last[0] - t0))

    res = fit_voxels(model, inp, frames, data, idx, m0, sd, w_roi, names, opts=opts,
                     chunk_size=args.chunk_size, dtype=dtype, progress=progress)
    log.info("voxel VB: %d voxels in %.1f s", len(idx), time.perf_counter() - t0)

    shape = data.shape[:3]
    meta = {**common, "PriorLambda": args.prior_lambda, "s0": args.s0, "c0": args.c0,
            "ARD": args.ard, "Precision": args.precision}
    ref = nib.Nifti1Image(np.zeros(shape, np.float32), img.affine, img.header)
    for k, p in enumerate(model.param_names):
        w.nifti(to_volume(res["m"][:, k], idx, shape), ref, f"model-{model.name}", f"meas-{p}",
                meta={**meta, "Description": f"VB posterior mean of {p}"})
        w.nifti(to_volume(np.sqrt(np.clip(res["cov"][:, k, k], 0, None)), idx, shape), ref,
                f"model-{model.name}", f"meas-{p}", "desc-sd",
                meta={**meta, "Description": f"VB posterior sd of {p}"})
    for name in model.derived:
        w.nifti(to_volume(res[f"derived_{name}"], idx, shape), ref, f"model-{model.name}",
                f"meas-{name}", meta={**meta, "Description": f"{name} from VB posterior mean"})
        w.nifti(to_volume(np.sqrt(np.clip(res[f"derived_{name}_var"], 0, None)), idx, shape), ref,
                f"model-{model.name}", f"meas-{name}", "desc-sd",
                meta={**meta, "Description": f"sd of {name} (delta method)"})
    w.nifti(to_volume(res["F"], idx, shape), ref, f"model-{model.name}", "meas-FreeEnergy", meta=meta)
    w.nifti(to_volume(1 / np.sqrt(res["s"] * res["c"]), idx, shape), ref, f"model-{model.name}",
            "meas-NoiseSD", meta={**meta, "Description": "sqrt(1/E[noise precision]), weighted units"})
    w.nifti(to_volume(res["n_iter"], idx, shape), ref, f"model-{model.name}", "meas-NIter", meta=meta)


if __name__ == "__main__":
    sys.exit(main())
