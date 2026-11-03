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
from scipy.special import digamma, gammaln

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


def _gamma_kl(a, b, a0, b0):
    """KL( Gamma(a, rate b) || Gamma(a0, rate b0) )."""
    return ((a - a0) * digamma(a) - gammaln(a) + gammaln(a0)
            + a0 * (np.log(b) - np.log(b0)) + a * (b0 - b) / b)


def spatial_free_energy(res, mu, sd, spatial, nb, deg, eps, a, b, a0, b0):
    """Free energy of the whole spatial model (up to constants fixed by the graph).

    F = Σ_v F_data,v - KL(q(θ) || p(θ | φ)) - Σ_k KL(q(φ_k) || p(φ_k)), with the
    regional Gaussian prior for R parameters and the joint MRF prior on the
    deviations for S parameters (log det(Λ + eps I) omitted: same in every sweep).
    """
    m, cov = res["m"], res["cov"]
    V, P = m.shape
    var = np.einsum("vii->vi", cov)
    _, logdet = np.linalg.slogdet(cov)
    entropy = 0.5 * np.sum(logdet) + 0.5 * V * P * (1 + np.log(2 * np.pi))
    cross = 0.0
    for k in range(P):
        if k in spatial:
            d = m[:, k] - mu[:, k]
            quad = np.sum(deg * d**2 - d * _nb_sum(d, nb)) + eps * np.sum(d**2)
            trace = np.sum((deg + eps) * var[:, k])
            Ephi, Elogphi = a[k] / b[k], digamma(a[k]) - np.log(b[k])
            cross += 0.5 * V * np.log(2 * np.pi) - 0.5 * V * Elogphi + 0.5 * Ephi * (quad + trace)
            cross += _gamma_kl(a[k], b[k], a0, b0)
        else:
            cross += np.sum(0.5 * np.log(2 * np.pi * sd[:, k] ** 2)
                            + 0.5 * ((m[:, k] - mu[:, k]) ** 2 + var[:, k]) / sd[:, k] ** 2)
    return float(np.sum(res["F_data"]) - cross + entropy)


def fit_spatial(model, inp, frames, Z, W2, mu, sd, prior_types, nb, opts=VBOptions(),
                n_sweeps=10, eps=0.1, phi_prior=(1e-3, 1e-3), init=None, tol=1e-5,
                **fit_kw):
    """VB with regional (R) and spatial (S) priors.

    ``mu``, ``sd``: (V,P) regional prior mean/sd; ``nb``: from :func:`neighbours`;
    ``init``: non-spatial ``fit_vb`` result (computed if None).

    Each sweep updates q(phi_k) (Gamma) from the current posteriors, gives every
    voxel its conditional MRF prior and re-runs the voxel VB (warm start). The
    free energy of the whole spatial model (:func:`spatial_free_energy`) is the
    stopping criterion: stop when its relative increase is below ``tol``, or when
    it decreases (the best sweep is returned), at most ``n_sweeps`` sweeps.
    Returns the selected ``fit_vb`` result plus ``phi``, ``phi_history``,
    ``F_history`` (index 0 = non-spatial start) and ``best_sweep``.
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

    def phi_posterior(r):
        a, b = np.full(P, np.nan), np.full(P, np.nan)
        m, var = r["m"], np.einsum("vii->vi", r["cov"])
        for k in spatial:
            d = m[:, k] - mu[:, k]
            # dᵀΛd = Σ_v deg_v d_v² - Σ_v d_v Σ_nb d_nb (= Σ_edges (d_i - d_j)²)
            quad = np.sum(deg * d**2 - d * _nb_sum(d, nb)) + eps * np.sum(d**2)
            trace = np.sum((deg + eps) * var[:, k])
            a[k], b[k] = a0 + V / 2, b0 + 0.5 * (quad + trace)
        return a, b

    # F of the start: phi at its optimum for the non-spatial solution
    a, b = phi_posterior(res)
    F_hist = [spatial_free_energy(res, mu, sd, spatial, nb, deg, eps, a, b, a0, b0)]
    phi_hist, best, best_F, best_sweep = [], res, F_hist[0], 0
    for sweep in range(1, n_sweeps + 1):
        a, b = phi_posterior(res)
        phi = a / b
        phi_hist.append(phi[spatial].copy())
        M0, prec = mu.copy(), 1.0 / sd**2
        for k in spatial:
            d = res["m"][:, k] - mu[:, k]
            M0[:, k] = mu[:, k] + _nb_sum(d, nb) / (deg + eps)
            prec[:, k] = phi[k] * (deg + eps)
        L0 = np.einsum("vi,ij->vij", prec, eye)
        res = fit_vb(model, inp, frames, Z, W2, M0, L0, opts=opts, M_init=res["m"], **fit_kw)
        a, b = phi_posterior(res)
        F = spatial_free_energy(res, mu, sd, spatial, nb, deg, eps, a, b, a0, b0)
        F_hist.append(F)
        log.info("spatial sweep %d: F = %.6g (dF = %.3g), phi = %s", sweep, F, F - F_hist[-2],
                 {model.param_names[k]: float(phi[k]) for k in spatial})
        if F > best_F:
            best, best_F, best_sweep = res, F, sweep
        if F < F_hist[-2] or abs(F - F_hist[-2]) < tol * abs(F_hist[-2]):
            break
    best = dict(best)
    a, b = phi_posterior(best)
    best["phi"] = a / b
    best["phi_history"] = np.array(phi_hist)
    best["F_history"] = np.array(F_hist)
    best["best_sweep"] = best_sweep
    return best
