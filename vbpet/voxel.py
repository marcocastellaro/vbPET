"""Stage 2: voxel-wise VB with region-based priors."""

import jax
import jax.numpy as jnp
import numpy as np

from .vb.core import VBOptions, fit_vb
from .vb.spatial import fit_spatial, neighbours


def voxel_priors(lab, roi_info, priors, model):
    """Per-voxel prior mean/sd from the ROI each voxel belongs to.

    ``lab`` (V,): dseg label of each masked voxel. Voxels whose label has no
    prior (unfitted or too small ROI) are dropped.
    Returns (sel (V,) bool, m0 (Vs,P), sd (Vs,P), roi (Vs,) row index into ``priors``).
    """
    pnames = model.param_names
    lab = np.asarray(lab).astype(int)
    row_of_name = {n: i for i, n in enumerate(priors["name"])}
    lut = -np.ones(max(int(lab.max()), int(roi_info["index"].max())) + 1, np.int64)
    for i, n in zip(roi_info["index"], roi_info["name"]):
        lut[int(i)] = row_of_name.get(n, -1)
    roi = np.where(lab > 0, lut[np.clip(lab, 0, None)], -1)
    sel = roi >= 0
    roi = roi[sel]
    mean = priors[[f"{p}_mean" for p in pnames]].to_numpy(float)
    sd = priors[[f"{p}_sd" for p in pnames]].to_numpy(float)
    return sel, mean[roi], sd[roi], roi


def fit_voxels(model, inp, frames, Z, idx, shape3d, m0, sd, W2, roi,
               opts=VBOptions(), chunk_size=8192, dtype=jnp.float64, progress=None,
               prior_types=None, spatial_sweeps=10, spatial_eps=0.1, spatial_within_roi=False):
    """VB for voxel TACs ``Z`` (V, T) at flat positions ``idx`` in ``shape3d``.

    ``W2`` (V, T) or (T,): inverse data variance (MATLAB: the ROI TAC weights
    squared for all voxels of the ROI). ``roi`` (V,): region id, used by
    ``spatial_within_roi``. ``prior_types``: one letter per parameter, ``R``
    regional (default) or ``S`` regional + spatial MRF (see vb.spatial).
    """
    L0 = np.einsum("vi,ij->vij", 1.0 / sd**2, np.eye(model.n_params))
    kw = dict(opts=opts, chunk_size=chunk_size, dtype=dtype, progress=progress)
    res = fit_vb(model, inp, frames, Z, W2, m0, L0, **kw)
    if prior_types and "S" in prior_types.upper():
        nb = neighbours(idx, shape3d, roi if spatial_within_roi else None)
        res = fit_spatial(model, inp, frames, Z, W2, m0, sd, prior_types, nb, init=res,
                          n_sweeps=spatial_sweeps, eps=spatial_eps, **kw)
    # derived parameters (delta method, autodiff gradients)
    for name, fn in model.derived.items():
        g = jax.jit(jax.vmap(jax.grad(fn)))
        v = jax.jit(jax.vmap(fn))
        vals, vars_ = [], []
        for i0 in range(0, len(Z), chunk_size):
            m = jnp.asarray(res["m"][i0:i0 + chunk_size])
            c = jnp.asarray(res["cov"][i0:i0 + chunk_size])
            gr = g(m)
            vals.append(np.asarray(v(m)))
            vars_.append(np.asarray(jnp.einsum("vi,vij,vj->v", gr, c, gr)))
        res[f"derived_{name}"] = np.concatenate(vals)
        res[f"derived_{name}_var"] = np.concatenate(vars_)
    return res


def to_volume(values, idx, shape3d, fill=np.nan):
    """Scatter voxel values into a 3D volume; non-finite values become ``fill``."""
    out = np.full(int(np.prod(shape3d)), fill, dtype=np.float32)
    v = np.asarray(values, np.float32)
    out[idx] = np.where(np.isfinite(v), v, fill)
    return out.reshape(shape3d)


QC_AT_BOUND, QC_MAX_ITER, QC_DERIVED_NONFINITE = 1, 2, 4


def qc_flags(res, model, max_iter):
    """Bitmask per voxel: 1 parameter at its lower bound, 2 iteration limit,
    4 a derived parameter (VT/Ki/...) is not finite."""
    lb = np.asarray(model.lower_bounds)
    flags = np.zeros(len(res["m"]), np.int16)
    flags[np.any(res["m"] <= lb + 1e-12, 1)] |= QC_AT_BOUND
    flags[res["n_iter"] >= max_iter] |= QC_MAX_ITER
    for name in model.derived:
        flags[~np.isfinite(res[f"derived_{name}"])] |= QC_DERIVED_NONFINITE
    return flags
