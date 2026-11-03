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
    g.add_argument("--model", default="2TCM",
                   help="plasma input: 1TCM 2TCM 2TCM1K irr2TCM 1K 5K; reference: SRTM SRTM2. "
                        "A comma-separated list (e.g. 1TCM,2TCM) fits all and adds "
                        "voxel-wise Bayesian model selection (model-selection outputs)")
    g.add_argument("--param-space", choices=["log", "linear"],
                   help="VB parameterisation: log (default) = log θ for parameters bounded "
                        "at 0, never hits the bound; linear = MATLAB (clipped at 0)")
    g.add_argument("--fix", nargs="+", default=[], metavar="PARAM=VALUE",
                   help="hold parameters fixed, e.g. --fix Vb=0.05")
    g.add_argument("--blood-derivative", type=Path,
                   help="derivative folder with *_inputfunction.tsv (e.g. bloodstream)")
    g.add_argument("--fit-delay", action="store_true",
                   help="estimate a blood arrival delay per region/voxel (parameter 'delay', "
                        "min; recommended for whole-body PET)")
    g.add_argument("--blood-shift", type=float, default=0.0,
                   help="delay applied to the blood curves (minutes, negative = earlier)")
    g.add_argument("--ref-label", nargs="+",
                   help="reference region for SRTM/SRTM2: dseg label names or indices "
                        "(several are merged, voxel weighted)")
    g.add_argument("--idif-label", nargs="+",
                   help="image-derived input: dseg label(s) of a blood pool (e.g. aorta, "
                        "left ventricle) used instead of blood samples (plasma-input models)")
    g.add_argument("--idif-plasma-ratio", type=float, default=1.0,
                   help="plasma/whole-blood ratio applied to the IDIF (FDG ~1.0-1.1)")
    g.add_argument("--k2p", type=float,
                   help="SRTM2 k2' (1/min); default: median k2/R1 of the regional SRTM fits")
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
                        "--input-step matlab --param-space linear (each can still be set)")
    g = p.add_argument_group("VB")
    g.add_argument("--s0", type=float, default=1e6)
    g.add_argument("--c0", type=float, default=50.0)
    g.add_argument("--max-iter", type=int, default=100)
    g.add_argument("--ard", help="parameter name with ARD prior (e.g. Vb)")
    g.add_argument("--prior-types",
                   help="one letter per (free) parameter: R = regional prior (default), "
                        "S = regional + spatial MRF prior, e.g. RRRSR for 2TCM")
    g.add_argument("--spatial-sweeps", type=int, default=10,
                   help="maximum spatial sweeps (stops earlier on the spatial free energy)")
    g.add_argument("--spatial-eps", type=float, default=0.1,
                   help="weight of the regional anchor in the spatial prior")
    g.add_argument("--spatial-within-roi", action="store_true",
                   help="spatial neighbours only inside the same dseg label")
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
        try:
            # keep the CPU backend with the GPU: the regional stage runs on the CPU
            jax.config.update("jax_platforms", "cpu" if args.device == "cpu" else "cuda,cpu")
        except Exception as e:  # noqa: BLE001 - backend already initialised (e.g. in tests)
            log.warning("could not select device %s: %s", args.device, e)
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


def parse_fixed(items):
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        if not v:
            raise SystemExit(f"--fix expects PARAM=VALUE, got {it!r}")
        out[k] = float(v)
    return out


def resolve_labels(items, labels):
    """dseg label names or indices -> indices."""
    by_name = {v.lower(): k for k, v in labels.items()}
    ids = []
    for it in items:
        if it.isdigit():
            ids.append(int(it))
        elif it.lower() in by_name:
            ids.append(by_name[it.lower()])
        else:
            raise SystemExit(f"label {it!r} not in the dseg labels")
    return ids


def process_run(run, args):
    import nibabel as nib
    import numpy as np
    import pandas as pd

    from .input import make_plasma_input
    from .io.bids import find_dseg, load_blood, load_masked, read_labels
    from .io.derivatives import RunWriter
    from .models import INPUT_KIND
    from .models.base import Frames
    from .models.reference import reference_input
    from .regional import frame_weights, roi_tacs

    w = RunWriter(args.output_dir, run)
    mc = args.matlab_compat
    conv = args.convolution or ("rect" if mc else "linear")
    sampling = args.frame_sampling or ("mid" if mc else "mean")
    step = args.input_step or ("matlab" if mc else "auto")
    space = args.param_space or ("linear" if mc else "log")
    try:
        step = float(step)
    except ValueError:
        pass
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    for m in models:
        if m not in INPUT_KIND:
            raise SystemExit(f"unknown model {m!r}; available: {sorted(INPUT_KIND)}")
    kinds = {INPUT_KIND[m] for m in models}
    if len(kinds) > 1:
        raise SystemExit("models compared in one run must share the input (blood or reference)")
    kind = kinds.pop()

    # --- data (kBq/mL, minutes): only voxels inside the dseg are loaded
    img = nib.load(str(run.path))
    frames = Frames.from_start_duration(run.frame_start_min, run.frame_duration_min)
    t_end = float(frames.end[-1])
    shape = img.shape[:3]

    dseg_path = find_dseg(run, args.derivatives, args.dseg_desc, args.dseg_space)
    dseg = np.asarray(nib.load(str(dseg_path)).dataobj).astype(int)
    labels = read_labels(dseg_path)
    log.info("dseg: %s (%d labels)", dseg_path, len(np.unique(dseg)) - 1)
    t0 = time.perf_counter()
    idx_all, X = load_masked(run.path, dseg > 0, scale=run.unit_factor / 1e3)
    bad = ~np.isfinite(X).all(1)
    if bad.any():
        log.warning("%d voxels with non-finite data excluded", int(bad.sum()))
        idx_all, X = idx_all[~bad], X[~bad]
    lab_all = dseg.reshape(-1)[idx_all]
    log.info("loaded %d voxels x %d frames (%.1f GB) in %.1f s", *X.shape, X.nbytes / 1e9,
             time.perf_counter() - t0)
    tacs, roi_info = roi_tacs(X, lab_all, labels, args.min_roi_voxels)
    dur = np.asarray(frames.end - frames.start)

    def region_tac(items, what):
        ids = resolve_labels(items, labels)
        names = [labels.get(i, f"label-{i}") for i in ids]
        sel = np.isin(lab_all, ids)
        if sel.sum() < args.min_roi_voxels:
            raise SystemExit(f"{what} {names} has {sel.sum()} voxels")
        return X[sel].mean(0), names, int(sel.sum())

    # --- input function
    exclude = []  # regions used as input are not fitted
    grid = step if isinstance(step, float) else 1 / 60
    if kind == "blood" and args.idif_label:
        tac, exclude, nvox = region_tac(args.idif_label, "IDIF region")
        inp = reference_input(frames, tac, t_end, step=grid)
        inp = type(inp)(inp.tp, inp.Cp * args.idif_plasma_ratio, inp.Cp)
        source = (f"IDIF {'+'.join(exclude)} ({nvox} voxels, plasma/blood ratio "
                  f"{args.idif_plasma_ratio:g})")
    elif kind == "blood":
        blood = load_blood(run, args.blood_derivative)
        inp = make_plasma_input(blood, t_end, step=step, scale=1e-3, shift=args.blood_shift)
        source = blood.source
    else:
        if not args.ref_label:
            raise SystemExit(f"{','.join(models)} needs --ref-label")
        tac, exclude, nvox = region_tac(args.ref_label, "reference region")
        inp = reference_input(frames, tac, t_end, step=grid)
        source = f"reference region {'+'.join(exclude)} ({nvox} voxels)"
    if kind == "blood" and exclude:
        tacs = tacs.drop(columns=exclude, errors="ignore")
        roi_info = roi_info[~roi_info["name"].isin(exclude)].reset_index(drop=True)
    log.info("input: %s (grid %d pts, dt=%.3f min)", source, len(inp.tp),
             float(inp.tp[1] - inp.tp[0]))
    weights = pd.DataFrame({c: frame_weights(tacs[c], dur, args.weights) for c in tacs})

    tac_df = pd.concat([pd.DataFrame({"frame_start": np.asarray(frames.start),
                                      "frame_end": np.asarray(frames.end)}), tacs], axis=1)
    base_meta = dict(Convolution=conv, FrameSampling=sampling, InputSource=source,
                     BloodShiftMin=args.blood_shift, Segmentation=str(dseg_path),
                     InputGridStepMin=float(inp.tp[1] - inp.tp[0]), ParameterSpace=space,
                     TimeUnits="min", ActivityUnits="kBq/mL")
    w.tsv(tac_df, "desc-dseg", suffix="tacs", meta=base_meta)
    w.tsv(roi_info, "desc-dseg", suffix="morph")

    ctx = dict(w=w, img=img, shape=shape, frames=frames, inp=inp, tacs=tacs, roi_info=roi_info,
               weights=weights, idx_all=idx_all, X=X, lab_all=lab_all, exclude=exclude,
               conv=conv, sampling=sampling, space=space, kind=kind, base_meta=base_meta)
    results = {}
    for name in models:
        log.info("--- model %s", name)
        results[name] = fit_model(name, ctx, args)
    if len(models) > 1 and args.stage != "regional":
        select_models(results, ctx, args)


def fit_model(name, ctx, args):
    """Regional + voxel stages for one model; returns what model selection needs."""
    import jax.numpy as jnp
    import nibabel as nib
    import numpy as np
    import pandas as pd

    from .models import get_model
    from .models.reference import k2p_from_srtm
    from .regional import fit_regions, make_priors
    from .vb.core import VBOptions
    from .voxel import (QC_AT_BOUND, QC_DERIVED_NONFINITE, QC_MAX_ITER, fit_voxels, qc_flags,
                        to_volume, voxel_priors)

    w, inp, frames, tacs, weights = ctx["w"], ctx["inp"], ctx["frames"], ctx["tacs"], ctx["weights"]
    conv, sampling, shape = ctx["conv"], ctx["sampling"], ctx["shape"]
    k2p = args.k2p
    if name == "SRTM2" and k2p is None:
        srtm = get_model("SRTM", conv=conv, sampling=sampling)
        kin_srtm, _ = fit_regions(srtm, inp, frames,
                                  tacs.drop(columns=ctx["exclude"], errors="ignore"), weights)
        w.tsv(kin_srtm, "model-SRTM", "desc-regional", suffix="kinpar")
        k2p = k2p_from_srtm(kin_srtm)
        log.info("SRTM2: k2' = %.4f /min (median k2/R1 of regional SRTM)", k2p)
    model = get_model(name, conv=conv, sampling=sampling,
                      **({"k2p": k2p} if name == "SRTM2" else {}))
    if args.fit_delay:
        if ctx["kind"] != "blood":
            raise SystemExit("--fit-delay applies to plasma-input models only")
        model = model.with_delay()
    model = model.fix(parse_fixed([f for f in args.fix if f.split("=")[0] in model.param_names]))
    common = dict(Model=model.name, FixedParameters={**model.fixed}, **ctx["base_meta"])

    # --- stage 1 (natural parameters, bounded WNLLS)
    if args.priors_tsv is None:
        t0 = time.perf_counter()
        kin, _ = fit_regions(model, inp, frames, tacs, weights)
        w.tsv(kin, f"model-{model.name}", "desc-regional", suffix="kinpar",
              meta={**common, "Weights": args.weights})
        priors = make_priors(kin, model, lam=args.prior_lambda)
        w.tsv(priors, f"model-{model.name}", "desc-prior", suffix="kinpar",
              meta={**common, "PriorLambda": args.prior_lambda})
        log.info("regional fits: %d ROIs in %.1f s", len(kin), time.perf_counter() - t0)
    else:
        kin = None
        priors = pd.read_csv(args.priors_tsv, sep="\t")
    if args.stage == "regional":
        return dict(model=model, kin=kin)

    # --- stage 2 (VB in log space for parameters bounded at 0, unless --param-space linear)
    vbm = model.log_space() if ctx["space"] == "log" else model
    sel, m0, sd, roi = voxel_priors(ctx["lab_all"], ctx["roi_info"], priors, model)
    m0, sd = vbm.prior_to_internal(m0, sd)
    idx, Z = ctx["idx_all"][sel], ctx["X"][sel]
    if args.max_voxels:
        idx, Z, m0, sd, roi = (a[:args.max_voxels] for a in (idx, Z, m0, sd, roi))
    wmat = np.stack([weights[n].to_numpy() for n in priors["name"]])  # prior rows -> weights
    W2 = (wmat[roi] ** 2).astype(np.float32)
    ard = vbm.param_names.index(args.ard) if args.ard else -1
    opts = VBOptions(s0=args.s0, c0=args.c0, max_iter=args.max_iter, ard=ard)
    dtype = jnp.float32 if args.precision == "float32" else jnp.float64
    t0 = time.perf_counter()
    last = [0.0]

    def progress(done, total):
        if time.perf_counter() - last[0] > 30 or done == total:
            last[0] = time.perf_counter()
            log.info("voxels %d/%d (%.0f vox/s)", done, total, done / (last[0] - t0))

    res = fit_voxels(vbm, inp, frames, Z, idx, shape, m0, sd, W2, roi, opts=opts,
                     chunk_size=args.chunk_size, dtype=dtype, progress=progress,
                     prior_types=args.prior_types, spatial_sweeps=args.spatial_sweeps,
                     spatial_eps=args.spatial_eps, spatial_within_roi=args.spatial_within_roi)
    log.info("voxel VB: %d voxels in %.1f s", len(idx), time.perf_counter() - t0)

    meta = {**common, "PriorLambda": args.prior_lambda, "s0": args.s0, "c0": args.c0,
            "ARD": args.ard, "Precision": args.precision, "LogParameters": list(vbm.log_params),
            "PriorTypes": args.prior_types or "R" * vbm.n_params,
            **({"SpatialPrecision": {p: float(v) for p, v in zip(vbm.param_names, res["phi"])
                                     if np.isfinite(v)},
                "SpatialEps": args.spatial_eps, "SpatialWithinROI": args.spatial_within_roi,
                "SpatialFreeEnergyHistory": [float(f) for f in res["F_history"]],
                "SpatialBestSweep": int(res["best_sweep"])}
               if "phi" in res else {})}
    img = ctx["img"]
    ref = nib.Nifti1Image(np.zeros(shape, np.float32), img.affine, img.header)
    est, est_sd = vbm.to_natural(res["m"], res["cov"])
    tag = f"model-{model.name}"
    for k, p in enumerate(vbm.param_names):
        how = "median exp(m) of the log-normal posterior" if p in vbm.log_params else "posterior mean"
        w.nifti(to_volume(est[:, k], idx, shape), ref, tag, f"meas-{p}",
                meta={**meta, "Description": f"VB {how} of {p}"})
        w.nifti(to_volume(est_sd[:, k], idx, shape), ref, tag, f"meas-{p}", "desc-sd",
                meta={**meta, "Description": f"VB posterior sd of {p}"})
    for dn in vbm.derived:
        w.nifti(to_volume(res[f"derived_{dn}"], idx, shape), ref, tag, f"meas-{dn}",
                meta={**meta, "Description": f"{dn} at the VB posterior point estimate"})
        w.nifti(to_volume(np.sqrt(np.clip(res[f"derived_{dn}_var"], 0, None)), idx, shape), ref,
                tag, f"meas-{dn}", "desc-sd", meta={**meta, "Description": f"sd of {dn} (delta method)"})
    w.nifti(to_volume(res["F"], idx, shape), ref, tag, "meas-FreeEnergy", meta=meta)
    w.nifti(to_volume(1 / np.sqrt(res["s"] * res["c"]), idx, shape), ref, tag, "meas-NoiseSD",
            meta={**meta, "Description": "sqrt(1/E[noise precision]), weighted units"})
    w.nifti(to_volume(res["n_iter"], idx, shape), ref, tag, "meas-NIter", meta=meta)
    flags = qc_flags(res, vbm, args.max_iter)
    counts = {"AtLowerBound": int(np.sum(flags & QC_AT_BOUND > 0)),
              "MaxIterReached": int(np.sum(flags & QC_MAX_ITER > 0)),
              "DerivedNonFinite": int(np.sum(flags & QC_DERIVED_NONFINITE > 0)),
              "Voxels": int(len(flags))}
    w.nifti(to_volume(flags, idx, shape, fill=0), ref, tag, "meas-QCFlags",
            meta={**meta, "Description": "bitmask: 1 parameter at lower bound, 2 iteration "
                  "limit reached, 4 derived parameter not finite (map set to n/a)",
                  "Counts": counts})
    log.info("QC: %s", counts)
    return dict(model=model, kin=kin, idx=idx, roi=roi, priors=priors, F=res["F"],
                derived={dn: res[f"derived_{dn}"] for dn in vbm.derived}, ref=ref)


def select_models(results, ctx, args):
    """Voxel-wise Bayesian model selection with the VB free energy.

    p(m | y) ∝ exp(F_m) (equal prior model probabilities). Valid because all
    candidates are fitted to the same voxel data with the same weights and
    noise prior; F is a lower bound on log evidence, so small differences
    (|dF| < ~1-3) are not decisive. Writes the winning model, the posterior model
    probabilities, model-averaged derived parameters and a per-ROI summary.
    """
    import numpy as np
    import pandas as pd

    from .voxel import to_volume

    w, shape = ctx["w"], ctx["shape"]
    names = list(results)
    idx = results[names[0]]["idx"]
    for n in names[1:]:
        idx = np.intersect1d(idx, results[n]["idx"])
    F = np.stack([results[n]["F"][np.searchsorted(results[n]["idx"], idx)] for n in names], 1)
    ok = np.isfinite(F).all(1)
    Fz = np.where(np.isfinite(F), F, -np.inf)
    Fz = Fz - Fz.max(1, keepdims=True)
    post = np.exp(Fz)
    post /= post.sum(1, keepdims=True)
    best = np.argmax(F, 1)
    meta = dict(ctx["base_meta"], Models=names, ModelIndex={i + 1: n for i, n in enumerate(names)},
                Method="voxel-wise Bayesian model selection, p(m|y) ∝ exp(F_m), equal priors")
    ref = results[names[0]]["ref"]
    w.nifti(to_volume(np.where(ok, best + 1, 0), idx, shape, fill=0), ref, "model-selection",
            "meas-BestModel", meta={**meta, "Description": "index of the model with the highest F"})
    for i, n in enumerate(names):
        w.nifti(to_volume(np.where(ok, post[:, i], np.nan), idx, shape), ref, "model-selection",
                f"meas-P{n}", meta={**meta, "Description": f"posterior probability of {n}"})
    shared = set.intersection(*[set(results[n]["derived"]) for n in names])
    for dn in sorted(shared):
        vals = np.stack([results[n]["derived"][dn][np.searchsorted(results[n]["idx"], idx)]
                         for n in names], 1)
        finite = np.isfinite(vals)
        pw = np.where(finite, post, 0.0)
        bma = (np.where(finite, vals, 0) * pw).sum(1)
        bma = bma / np.where(pw.sum(1) > 0, pw.sum(1), np.nan)
        w.nifti(to_volume(bma, idx, shape), ref, "model-selection", f"meas-{dn}", "desc-bma",
                meta={**meta, "Description": f"{dn} averaged over models with p(m|y)"})
    # per-ROI summary: regional AIC/BIC (WNLLS) + voxel F (fixed-effects sum) + voting
    roi_names = results[names[0]]["priors"]["name"].to_numpy()
    roi = results[names[0]]["roi"][np.searchsorted(results[names[0]]["idx"], idx)]
    rows = []
    for r, rn in enumerate(roi_names):
        m = (roi == r) & ok
        if not m.any():
            continue
        row = dict(name=rn, n_voxels=int(m.sum()))
        for i, n in enumerate(names):
            kin = results[n]["kin"]
            if kin is not None and rn in set(kin["name"]):
                kr = kin.set_index("name").loc[rn]
                row[f"AIC_{n}"], row[f"BIC_{n}"] = float(kr["aic"]), float(kr["bic"])
            row[f"sumF_{n}"] = float(F[m, i].sum())
            row[f"frac_best_{n}"] = float(np.mean(best[m] == i))
        sumF = np.array([row[f"sumF_{n}"] for n in names])
        row["best_sumF"] = names[int(np.argmax(sumF))]
        row["dF_best_vs_second"] = float(np.sort(sumF)[-1] - np.sort(sumF)[-2])
        if all(f"AIC_{n}" in row for n in names):
            row["best_AIC"] = names[int(np.argmin([row[f"AIC_{n}"] for n in names]))]
            row["best_BIC"] = names[int(np.argmin([row[f"BIC_{n}"] for n in names]))]
        rows.append(row)
    table = pd.DataFrame(rows)
    w.tsv(table, "model-selection", "desc-regional", suffix="kinpar", meta=meta)
    log.info("model selection: voxel winners %s",
             {n: int(np.sum(best[ok] == i)) for i, n in enumerate(names)})

if __name__ == "__main__":
    sys.exit(main())
