"""Random-effects Bayesian model selection (RFX-BMS).

Stephan, Penny, Daunizeau, Moran & Friston (2009) NeuroImage 46:1004-1017
(variational Dirichlet scheme, exceedance probabilities) and Rigoux, Stephan,
Friston & Daunizeau (2014) NeuroImage 84:971-985 (Bayesian omnibus risk and
protected exceedance probabilities).

Here the "subjects" are the voxels of a region: each voxel may follow a
different model, and the model frequencies r ~ Dir(alpha) are inferred from
the voxel log-evidences (VB free energies). Caveat: neighbouring voxels are not
independent (PET resolution is several voxels), so with thousands of voxels
the exceedance probabilities saturate quickly; the expected frequencies are
the more informative summary.
"""

import numpy as np
from scipy.special import digamma, gammaln


def rfx_bms(lme, alpha0=1.0, tol=1e-6, max_iter=1000, n_samples=100_000, seed=0):
    """RFX-BMS on log model evidences ``lme`` (N, K).

    Returns dict(alpha, r (expected frequencies), xp (exceedance probabilities),
    bor (Bayesian omnibus risk), pxp (protected exceedance probabilities),
    F1, F0, g (N,K posterior model assignment per sample)).
    """
    lme = np.asarray(lme, float)
    lme = lme[np.all(np.isfinite(lme), 1)]
    N, K = lme.shape
    a0 = np.full(K, float(alpha0))
    alpha = a0.copy()
    for _ in range(max_iter):
        log_u = lme + digamma(alpha) - digamma(alpha.sum())
        log_u -= log_u.max(1, keepdims=True)
        g = np.exp(log_u)
        g /= g.sum(1, keepdims=True)
        new = a0 + g.sum(0)
        done = np.max(np.abs(new - alpha)) < tol
        alpha = new
        if done:
            break
    r = alpha / alpha.sum()
    rng = np.random.default_rng(seed)
    samples = rng.dirichlet(alpha, size=n_samples)
    xp = np.bincount(samples.argmax(1), minlength=K) / n_samples
    # free energy of the RFX model (F1) and of the null "all models equally likely" (F0)
    Elogr = digamma(alpha) - digamma(alpha.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        glog = np.where(g > 0, g * np.log(g), 0.0)
    F1 = (np.sum(g * (lme + Elogr)) - np.sum(glog)
          + gammaln(a0.sum()) - gammaln(a0).sum() + np.sum((a0 - 1) * Elogr)
          - (gammaln(alpha.sum()) - gammaln(alpha).sum() + np.sum((alpha - 1) * Elogr)))
    mx = lme.max(1, keepdims=True)
    F0 = float(np.sum(mx[:, 0] + np.log(np.exp(lme - mx).sum(1)) - np.log(K)))
    bor = float(1.0 / (1.0 + np.exp(np.clip(F1 - F0, -700, 700))))
    pxp = (1 - bor) * xp + bor / K
    return dict(alpha=alpha, r=r, xp=xp, bor=bor, pxp=pxp, F1=float(F1), F0=F0, g=g, n=N)
