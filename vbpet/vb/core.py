"""Batched Variational Bayes for non-linear kinetic models.

Port of the MATLAB vbPET code (``trialsMethodW``/``updateVB``/``freeEnergy``):

* Gaussian prior N(m0, L0^-1) on the parameters, Gamma(s, c) on the noise
  precision with hyper-priors (s0, c0); data covariance diag(1/w2).
* Linearised VB updates with the parameters clipped at zero after each update.
* Stop at ``max_iter``, |ΔF| < ``tol`` or ``max_trials`` consecutive
  non-increasing F; return the iterate with the highest free energy.
* Optional ARD on one parameter (prior mean 0, precision re-estimated).

Differences from MATLAB (deliberate):

* exact Jacobian by forward-mode autodiff (``jacobian="numeric"`` reproduces
  ``computeJac2``);
* Cholesky/solve instead of ``pinv`` (identical for non-singular matrices);
* the data log-determinant uses only frames with non-zero weight. In MATLAB a
  zero weight makes ``det(pinv(inv_Sigma)) = 0`` and F = +Inf, which stops the
  iterations after the first step;
* computations run in rescaled coordinates theta = D*u (D = |m0|), which
  leaves F and the updates unchanged but conditions float32 well.

Each voxel runs its own ``lax.while_loop``; ``vmap`` batches voxels.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import digamma, gammaln


@dataclass(frozen=True)
class VBOptions:
    s0: float = 1e6
    c0: float = 50.0
    max_iter: int = 100
    max_trials: int = 10
    tol: float = 1e-3
    ard: int = -1  # index of the ARD parameter, -1 = none
    jacobian: str = "autodiff"  # or "numeric" (MATLAB computeJac2)
    nonneg: bool = True


def _jac_numeric(f, u, D):
    """MATLAB computeJac2 in original units, returned in scaled units."""
    m = u * D
    base = f(u)
    delta = jnp.maximum(jnp.abs(m) * 1e-6, 1e-12)
    eye = jnp.eye(u.shape[0])

    def col(i):
        return (f(u + eye[i] * delta[i] / D[i]) - base) / delta[i]

    J = jax.vmap(col)(jnp.arange(u.shape[0])).T
    return J * D[None, :]


def _vb_single(z, w2, m0, L0, D, *, f_orig, opts: VBOptions):
    """VB fit of one TAC. All vectors are in original units on input/output."""
    n = z.shape[0]
    P = m0.shape[0]
    s0, c0 = opts.s0, opts.c0
    ard = opts.ard

    def f(u):
        return f_orig(u * D)

    if opts.jacobian == "autodiff":
        jac = jax.jacfwd(f)
    else:
        jac = partial(_jac_numeric, f, D=D)

    # scaled prior
    m0u = m0 / D
    L0u = L0 * D[:, None] * D[None, :]
    if ard >= 0:
        m0u = m0u.at[ard].set(0.0)
        L0u = L0u.at[ard, ard].set(1e-6 * D[ard] ** 2)

    logdet_data = -jnp.sum(jnp.where(w2 > 0, jnp.log(jnp.where(w2 > 0, w2, 1.0)), 0.0))

    def update(s, c, J, L_old, k, m_old):
        L0e = L0u
        if ard >= 0:
            L0e = L0e.at[ard, ard].set(
                1.0 / (m_old[ard] ** 2 + 1.0 / L_old[ard, ard]))
        JtW = J.T * w2[None, :]
        JtWJ = JtW @ J
        c_new = (n - 1) / 2 + c0
        tr = jnp.trace(jnp.linalg.solve(L_old, JtWJ))
        s_new = 1.0 / (1.0 / s0 + (jnp.sum(w2 * k * k) + tr) / 2)
        sc = s_new * c_new
        L = JtWJ * sc + L0e
        rhs = JtW @ (k + J @ m_old) * sc + L0e @ m0u
        m_new = jnp.linalg.solve(L, rhs)
        if opts.nonneg:
            m_new = jnp.maximum(m_new, 0.0)
        return L, m_new, c_new, s_new

    def free_energy(s, c, m, L, k, J):
        _, logdetL = jnp.linalg.slogdet(L)
        _, logdetL0 = jnp.linalg.slogdet(L0u)
        Linv = jnp.linalg.inv(L)
        dm = m - m0u
        logtheta = (0.5 * logdetL - 0.5 * logdetL0 + 0.5 * jnp.trace(Linv @ L0u)
                    + 0.5 * dm @ L0u @ dm - 0.5 * P)
        logphi = ((c - 1) * digamma(c) - jnp.log(s) - c - gammaln(c) + gammaln(c0)
                  + c0 * jnp.log(s0) - (c0 - 1) * (jnp.log(s) + digamma(c)) + s * c / s0)
        JtWJ = (J.T * w2[None, :]) @ J
        logpost = (-0.5 * n * jnp.log(2 * jnp.pi) - 0.5 * logdet_data
                   + 0.5 * n * (jnp.log(s) + digamma(c))
                   - 0.5 * s * c * (jnp.sum(w2 * k * k) + jnp.trace(JtWJ @ Linv)))
        F = logpost - logphi - logtheta
        if ard >= 0:
            d = 0.5
            # in original units (r is not scale invariant)
            r = 2.0 / ((m[ard] * D[ard]) ** 2 + 1.0 / (L[ard, ard] / D[ard] ** 2))
            F = F + (-1.5 * (jnp.log(r) + digamma(d)) - d - gammaln(d) - d * jnp.log(r))
        return F

    def evaluate(u):
        y = f(u)
        return jac(u), z - y

    # step 1 (from the prior)
    J0, k0 = evaluate(m0u)
    L, m, c, s = update(s0, c0, J0, L0u, k0, m0u)
    J, k = evaluate(m)
    F = free_energy(s, c, m, L, k, J)

    dtype = z.dtype
    state = dict(
        count=jnp.array(1), trials=jnp.array(0),
        F_new=F, F_old=jnp.array(-jnp.inf, dtype),
        m=m, L=L, s=s, c=jnp.asarray(c, dtype), J=J, k=k,
        bF=jnp.array(-jnp.inf, dtype), bm=jnp.zeros_like(m), bL=jnp.zeros_like(L),
        bs=jnp.zeros((), dtype), bc=jnp.zeros((), dtype),
    )

    # |dF| below the rounding level of F cannot be resolved (matters in float32)
    eps = jnp.finfo(dtype).eps

    def cond(st):
        tol = jnp.maximum(opts.tol, 10 * eps * jnp.abs(st["F_new"]))
        return ((st["count"] < opts.max_iter)
                & (jnp.abs(st["F_new"] - st["F_old"]) > tol)
                & (st["trials"] < opts.max_trials))

    def body(st):
        trials = jnp.where(st["F_new"] > st["F_old"], 0, st["trials"] + 1)
        L, m, c, s = update(st["s"], st["c"], st["J"], st["L"], st["k"], st["m"])
        J, k = evaluate(m)
        F = free_energy(s, c, m, L, k, J)
        better = F > st["bF"]
        return dict(
            count=st["count"] + 1, trials=trials, F_new=F, F_old=st["F_new"],
            m=m, L=L, s=s, c=jnp.asarray(c, dtype), J=J, k=k,
            bF=jnp.where(better, F, st["bF"]),
            bm=jnp.where(better, m, st["bm"]),
            bL=jnp.where(better, L, st["bL"]),
            bs=jnp.where(better, s, st["bs"]),
            bc=jnp.where(better, c, st["bc"]),
        )

    st = jax.lax.while_loop(cond, body, state)

    m_best = st["bm"] * D
    L_best = st["bL"] / (D[:, None] * D[None, :])
    cov = jnp.linalg.inv(st["bL"]) * (D[:, None] * D[None, :])
    fit = f_orig(m_best)
    return dict(m=m_best, cov=cov, L=L_best, s=st["bs"], c=st["bc"], F=st["bF"],
                n_iter=st["count"], fit=fit)


def make_vb_fn(model, inp, frames, opts: VBOptions = VBOptions()):
    """Return a jitted function fitting a batch of TACs.

    Signature: ``fn(Z (V,n), W2 (V,n), M0 (V,P), L0 (V,P,P)) -> dict of (V,...)``.
    """

    def f_orig(theta):
        return model.predict(theta, inp, frames)

    def single(z, w2, m0, L0):
        D = jnp.where(jnp.abs(m0) > 0, jnp.abs(m0), 1.0)
        return _vb_single(z, w2, m0, L0, D, f_orig=f_orig, opts=opts)

    return jax.jit(jax.vmap(single))


def fit_vb(model, inp, frames, Z, W2, M0, L0, opts: VBOptions = VBOptions(),
           chunk_size=4096, dtype=jnp.float64, progress=None):
    """Fit V TACs in chunks. ``W2``, ``M0``, ``L0`` may be shared (no V axis).

    ``W2`` is the inverse data variance per frame (0 = frame ignored).
    Returns a dict of numpy arrays with a leading V axis.
    """
    Z = np.asarray(Z)
    V, n = Z.shape
    P = model.n_params
    W2 = np.broadcast_to(np.asarray(W2), (V, n))
    M0 = np.broadcast_to(np.asarray(M0), (V, P))
    L0 = np.broadcast_to(np.asarray(L0), (V, P, P))

    cast = lambda a: jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype), a)
    fn = make_vb_fn(model, cast(inp), cast(frames), opts)

    chunk = int(min(chunk_size, V))
    outs = []
    for i0 in range(0, V, chunk):
        sl = slice(i0, min(i0 + chunk, V))
        nv = sl.stop - sl.start
        pad = chunk - nv
        args = [Z[sl], W2[sl], M0[sl], L0[sl]]
        if pad:
            # pad with copies of the first row to keep a fixed (compiled) shape
            args = [np.concatenate([a, np.repeat(a[:1], pad, axis=0)]) for a in args]
        res = fn(*[jnp.asarray(a, dtype) for a in args])
        res = jax.tree_util.tree_map(lambda x: np.asarray(x)[:nv], res)
        outs.append(res)
        if progress is not None:
            progress(sl.stop, V)
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def regional_prior(m_roi, lam=0.5, sd_floor=None):
    """MATLAB prior: mean = ROI estimate, precision = 1/(lam*m)^2 (diagonal)."""
    m_roi = np.asarray(m_roi, float)
    sd = lam * np.abs(m_roi)
    if sd_floor is not None:
        sd = np.maximum(sd, sd_floor)
    return m_roi, np.einsum("...i,ij->...ij", 1.0 / sd**2, np.eye(m_roi.shape[-1]))
