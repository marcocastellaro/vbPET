"""Spatial (MRF) priors on top of the region-based priors.

Per-parameter prior type string (one letter per parameter, fabber-like):

* ``R``: regional prior N(mu_ROI, (lambda mu_ROI)^2) (MATLAB vbPET behaviour);
* ``S``: spatial prior on the deviation from the regional mean,
  d_k = theta_k - mu_k,  p(d_k) ∝ exp(-phi_k/2 · d_kᵀ (Λ + eps·I) d_k),
  with Λ the graph Laplacian of the fitted voxels (6-neighbourhood) and phi_k
  learned (Gamma posterior, as fabber's ``M`` prior).

Mean-field scheme (Jacobi): given the current posterior means, each voxel sees
a Gaussian prior with mean mu_v + (Σ_nb d_nb)/(deg_v + eps) and precision
phi_k·(deg_v + eps); phi_k is updated from E[d_kᵀ(Λ+eps I)d_k]; the voxel VB
is re-run warm-started from the previous solution. As in VB_BBB/fabber
spatialvb, the first sweep starts from the non-spatial VB solution.
"""

import logging

import numpy as np

from .core import VBOptions, fit_vb

log = logging.getLogger("vbpet")

OFFSETS6 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])


def neighbours(idx, shape3d, roi=None):
    """(V, 6) neighbour positions (into ``idx``) of each fitted voxel, -1 if none.

    ``roi`` (V,) labels: when given, edges between different labels are cut.
    """
    idx = np.asarray(idx)
    lut = -np.ones(int(np.prod(shape3d)), np.int64)
    lut[idx] = np.arange(len(idx))
    ijk = np.stack(np.unravel_index(idx, shape3d), 1)
    nb = -np.ones((len(idx), 6), np.int64)
    for j, off in enumerate(OFFSETS6):
        q = ijk + off
        ok = np.all((q >= 0) & (q < np.asarray(shape3d)), 1)
        flat = np.ravel_multi_index(q[ok].T, shape3d)
        nb[ok, j] = lut[flat]
    if roi is not None:
        roi = np.asarray(roi)
        bad = (nb >= 0) & (roi[np.clip(nb, 0, None)] != roi[:, None])
        nb[bad] = -1
    return nb


def _nb_sum(x, nb):
    """Σ over valid neighbours of x (V,) -> (V,)."""
    valid = nb >= 0
    return np.where(valid, x[np.clip(nb, 0, None)], 0.0).sum(1)


def fit_spatial(model, inp, frames, Z, W2, mu, sd, prior_types, nb, opts=VBOptions(),
                n_sweeps=5, eps=0.1, phi_prior=(1e-3, 1e-3), init=None, tol=1e-2,
                **fit_kw):
    """VB with regional (R) and spatial (S) priors.

    ``mu``, ``sd``: (V,P) regional prior mean/sd; ``nb``: from :func:`neighbours`;
    ``init``: non-spatial ``fit_vb`` result (computed if None).
    Returns the last ``fit_vb`` result plus ``phi`` (P,) and ``phi_history``.
    """
    P = model.n_params
    if len(prior_types) != P:
        raise ValueError(f"prior types {prior_types!r} must have one letter per parameter "
                         f"{model.param_names}")
    spatial = [k for k, t in enumerate(prior_types.upper()) if t == "S"]
    eye = np.eye(P)
    L0_reg = np.einsum("vi,ij->vij", 1.0 / sd**2, eye)
    res = init if init is not None else fit_vb(model, inp, frames, Z, W2, mu, L0_reg,
                                                opts=opts, **fit_kw)
    if not spatial:
        return res
    deg = (nb >= 0).sum(1).astype(float)
    a0, b0 = phi_prior
    V = len(Z)
    phi = np.full(P, np.nan)
    history = []
    for sweep in range(n_sweeps):
        m = res["m"]
        var = np.einsum("vii->vi", res["cov"])
        M0 = mu.copy()
        prec = 1.0 / sd**2
        for k in spatial:
            d = m[:, k] - mu[:, k]
            # E[dᵀ(Λ+eps I)d] with a diagonal posterior:
            # dᵀΛd = Σ_v deg_v d_v² - Σ_v d_v Σ_nb d_nb (= Σ_edges (d_i - d_j)²)
            quad = np.sum(deg * d**2 - d * _nb_sum(d, nb)) + eps * np.sum(d**2)
            trace = np.sum((deg + eps) * var[:, k])
            phi[k] = (a0 + V / 2) / (b0 + 0.5 * (quad + trace))
            M0[:, k] = mu[:, k] + _nb_sum(d, nb) / (deg + eps)
            prec[:, k] = phi[k] * (deg + eps)
        history.append(phi[spatial].copy())
        log.info("spatial sweep %d: phi = %s", sweep + 1,
                 {model.param_names[k]: float(phi[k]) for k in spatial})
        L0 = np.einsum("vi,ij->vij", prec, eye)
        new = fit_vb(model, inp, frames, Z, W2, M0, L0, opts=opts, M_init=m, **fit_kw)
        change = np.nanmax(np.abs(new["m"][:, spatial] - m[:, spatial]) /
                           (np.sqrt(var[:, spatial]) + 1e-12), axis=0)
        res = new
        if sweep > 0 and np.all(np.abs(history[-1] / history[-2] - 1) < tol) and \
                np.median(change) < tol:
            break
    res["phi"] = phi
    res["phi_history"] = np.array(history)
    return res
