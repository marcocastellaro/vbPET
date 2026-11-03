"""Stage 1: regional TACs, WNLLS fits and region-based priors.

Port of the ROI part of MATLAB ``VB_estimation.m`` / ``wnllsEstimation.m``:
ROI TAC -> weighted NLLS with theta >= 0 -> prior mean = ROI estimate,
prior sd = lambda * estimate.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

log = logging.getLogger("vbpet")


def roi_tacs(X, lab, labels=None, min_voxels=10):
    """Mean TAC per label from voxel data ``X`` (V, T) and voxel labels ``lab`` (V,).

    Returns (DataFrame of TACs [frames x rois], info DataFrame).
    """
    flat = np.asarray(lab).astype(int).reshape(-1)
    X = np.asarray(X)
    order = np.argsort(flat, kind="stable")
    ids, start, counts = np.unique(flat[order], return_index=True, return_counts=True)
    sums = np.add.reduceat(X[order].astype(np.float64), start, axis=0)
    tacs, rows = {}, []
    for i, n, s_ in zip(ids, counts, sums):
        if i == 0 or n < min_voxels:
            continue
        name = (labels or {}).get(int(i), f"label-{i}")
        tacs[name] = s_ / n
        rows.append(dict(index=int(i), name=name, n_voxels=int(n)))
    return pd.DataFrame(tacs), pd.DataFrame(rows)


def frame_weights(tac, frame_dur, kind="matlab"):
    """Data weights w (sd = 1/w). ``matlab``: sqrt(dt / C), 0 where C <= 0."""
    tac = np.asarray(tac, float)
    dur = np.asarray(frame_dur, float)
    if kind == "matlab":
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.sqrt(dur / tac)
        w[~np.isfinite(w) | (tac <= 0)] = 0.0
        return w
    if kind == "duration":
        return np.sqrt(dur / dur.mean())
    if kind == "uniform":
        return np.ones_like(tac)
    raise ValueError(f"unknown weighting {kind!r}")


def compiled(model, inp, frames):
    """Jitted model and Jacobian (compile once, reuse for all ROIs)."""
    f = jax.jit(lambda th: model.predict(th, inp, frames))
    jf = jax.jit(jax.jacfwd(lambda th: model.predict(th, inp, frames)))
    return f, jf


def wnlls(model, inp, frames, tac, w, p0=None, n_starts=5, seed=0, upper=None, fns=None):
    """Weighted NLLS within the model bounds, multi-start. Returns dict(par, cov, ...).

    ``fns``: output of :func:`compiled` to avoid recompiling for every TAC.
    """
    p0 = np.asarray(model.p0 if p0 is None else p0, float)
    P = len(p0)
    f, jf = fns if fns is not None else compiled(model, inp, frames)
    w = np.asarray(w, float)
    z = np.asarray(tac, float)

    def res(th):
        return (z - np.asarray(f(th))) * w

    def jac(th):
        return -np.asarray(jf(th)) * w[:, None]

    ub = np.full(P, np.inf) if upper is None else np.asarray(upper, float)
    lb = np.asarray(model.lower_bounds, float)
    rng = np.random.default_rng(seed)
    # multiplicative jitter; additive for parameters that may be <= 0 (delay)
    jit = lambda: np.where(lb < 0, p0 + 0.1 * rng.normal(size=P), p0 * np.exp(rng.normal(0, 0.7, P)))
    starts = [p0] + [jit() for _ in range(n_starts - 1)]
    best, errors = None, []
    for s in starts:
        s = np.clip(np.where(lb < 0, s, np.maximum(s, 1e-8)), lb + 1e-9,
                    np.where(np.isfinite(ub), ub * 0.999, np.inf))
        try:
            r = least_squares(res, s, jac=jac, bounds=(lb, ub), method="trf",
                              x_scale="jac", max_nfev=3000, ftol=1e-10, xtol=1e-10)
        except Exception as e:  # noqa: BLE001 - a failed start is just skipped
            errors.append(f"{type(e).__name__}: {e}")
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        log.warning("WNLLS failed for all %d starts: %s", len(starts), errors[:1])
        nan = np.full(P, np.nan)
        return dict(par=nan, cov=np.full((P, P), np.nan), sigma2=np.nan, wrss=np.nan,
                    fit=np.full(len(z), np.nan), aic=np.nan, bic=np.nan, rss=np.nan,
                    success=False)
    th = best.x
    nd = int((w > 0).sum())
    wrss = float(2 * best.cost)
    sigma2 = wrss / max(nd - P, 1)
    J = best.jac
    cov = sigma2 * np.linalg.pinv(J.T @ J)
    fit = np.asarray(f(th))
    rss = float(np.sum((z - fit) ** 2))
    return dict(par=th, cov=cov, sigma2=sigma2, wrss=wrss, fit=fit,
                aic=nd * np.log(wrss / nd) + 2 * P, bic=nd * np.log(wrss / nd) + P * np.log(nd),
                rss=rss, success=bool(best.success))


def fit_regions(model, inp, frames, tacs, weights, **kw):
    """Fit all ROI TACs. ``weights``: DataFrame like ``tacs`` (or one vector).

    Runs on the CPU: each fit is tiny and sequential (scipy), so GPU launch and
    transfer overheads would dominate.
    """
    try:
        cpu = jax.devices("cpu")[0]
    except RuntimeError:  # CPU backend disabled (jax_platforms): use the default device
        return _fit_regions(model, inp, frames, tacs, weights, **kw)
    with jax.default_device(cpu):
        inp = jax.tree_util.tree_map(lambda x: jax.device_put(jnp.asarray(x)), inp)
        frames = jax.tree_util.tree_map(lambda x: jax.device_put(jnp.asarray(x)), frames)
        return _fit_regions(model, inp, frames, tacs, weights, **kw)


def _fit_regions(model, inp, frames, tacs, weights, **kw):
    rows, fits = [], {}
    fns = compiled(model, inp, frames)
    for name in tacs.columns:
        w = weights[name].to_numpy() if isinstance(weights, pd.DataFrame) else weights
        r = wnlls(model, inp, frames, tacs[name].to_numpy(), w, fns=fns, **kw)
        fits[name] = r["fit"]
        sd = np.sqrt(np.clip(np.diag(r["cov"]), 0, None))
        row = dict(name=name)
        for p, v, s in zip(model.param_names, r["par"], sd):
            row[p] = v
            row[f"{p}_se"] = s
        dv = model.derived_with_var(jnp.asarray(r["par"]), jnp.asarray(r["cov"]))
        for k, (v, var) in dv.items():
            row[k] = float(v)
            row[f"{k}_se"] = float(np.sqrt(max(float(var), 0.0)))
        row.update(wrss=r["wrss"], sigma2=r["sigma2"], aic=r["aic"], bic=r["bic"],
                   success=r["success"])
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(fits)


def make_priors(kinpar, model, lam=0.5, sd_floor_rel=0.05, max_cv=None, fallback=None):
    """Prior table: mean = regional estimate, sd = lam * mean (MATLAB).

    ``sd_floor_rel``: sd >= sd_floor_rel * (median of that parameter over ROIs), so
    a parameter estimated at 0 does not get an infinite prior precision.
    ROIs with CV > ``max_cv`` (percent) or failed fits get ``fallback`` (a dict
    param -> (mean, sd)), default the median over the good ROIs with sd = mean.
    """
    rows = []
    ok = kinpar["success"].astype(bool) if "success" in kinpar else np.ones(len(kinpar), bool)
    med = {p: float(np.nanmedian(kinpar.loc[ok, p])) for p in model.param_names}
    for _, r in kinpar.iterrows():
        bad = not bool(r.get("success", True))
        if max_cv is not None:
            for p in model.param_names:
                if p in model.prior_sd_min:  # no CV for parameters around 0
                    continue
                if r[p] > 0 and 100 * r[f"{p}_se"] / r[p] > max_cv:
                    bad = True
        row = dict(name=r["name"], fallback=bad)
        for p in model.param_names:
            if bad:
                mean, sd = (fallback or {}).get(p, (med[p], abs(med[p])))
            else:
                mean = float(r[p])
                sd = lam * abs(mean)
            sd = max(sd, sd_floor_rel * abs(med[p]), model.prior_sd_min.get(p, 0.0), 1e-12)
            row[f"{p}_mean"] = mean
            row[f"{p}_sd"] = sd
        rows.append(row)
    return pd.DataFrame(rows)
