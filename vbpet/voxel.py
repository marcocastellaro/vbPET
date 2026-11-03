"""Stage 2: voxel-wise VB with region-based priors."""

import jax
import jax.numpy as jnp
import numpy as np

from .vb.core import VBOptions, fit_vb


def voxel_priors(dseg, roi_info, priors, model, mask):
    """Per-voxel prior mean/sd from the ROI each voxel belongs to.

    Voxels in ``mask`` whose label has no prior are left out (``valid=False``).
    Returns (idx (V,), m0 (V,P), sd (V,P), roi_name per voxel).
    """
    pnames = model.param_names
    lab = np.asarray(dseg).astype(int)
    by_index = dict(zip(roi_info["index"], roi_info["name"]))
    table = priors.set_index("name")
    flat_lab = lab.reshape(-1)
    idx = np.flatnonzero(np.asarray(mask).reshape(-1) & (flat_lab > 0))
    names = np.array([by_index.get(int(l), None) for l in flat_lab[idx]], dtype=object)
    ok = np.array([n is not None and n in table.index for n in names])
    idx, names = idx[ok], names[ok]
    m0 = np.stack([table.loc[names, f"{p}_mean"].to_numpy(float) for p in pnames], 1)
    sd = np.stack([table.loc[names, f"{p}_sd"].to_numpy(float) for p in pnames], 1)
    return idx, m0, sd, names


def fit_voxels(model, inp, frames, data4d, idx, m0, sd, w_roi, roi_names,
               opts=VBOptions(), chunk_size=8192, dtype=jnp.float64, progress=None):
    """VB for the voxels ``idx`` (flat indices into the 3D grid).

    ``w_roi``: dict roi name -> frame weights (MATLAB: ROI TAC weights for all
    voxels of the ROI).
    """
    X = np.asarray(data4d).reshape(-1, data4d.shape[-1])
    Z = X[idx]
    W2 = np.stack([np.asarray(w_roi[n]) ** 2 for n in roi_names])
    L0 = np.einsum("vi,ij->vij", 1.0 / sd**2, np.eye(model.n_params))
    res = fit_vb(model, inp, frames, Z, W2, m0, L0, opts=opts, chunk_size=chunk_size,
                 dtype=dtype, progress=progress)
    # derived parameters (delta method, autodiff gradients)
    for name, fn in model.derived.items():
        g = jax.jit(jax.vmap(jax.grad(fn)))
        v = jax.jit(jax.vmap(fn))
        vals, vars_ = [], []
        for i0 in range(0, len(idx), chunk_size):
            m = jnp.asarray(res["m"][i0:i0 + chunk_size])
            c = jnp.asarray(res["cov"][i0:i0 + chunk_size])
            gr = g(m)
            vals.append(np.asarray(v(m)))
            vars_.append(np.asarray(jnp.einsum("vi,vij,vj->v", gr, c, gr)))
        res[f"derived_{name}"] = np.concatenate(vals)
        res[f"derived_{name}_var"] = np.concatenate(vars_)
    return res


def to_volume(values, idx, shape3d, fill=np.nan):
    out = np.full(int(np.prod(shape3d)), fill, dtype=np.float32)
    out[idx] = values
    return out.reshape(shape3d)
