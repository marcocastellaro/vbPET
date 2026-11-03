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
    ard_init_zero: bool = True  # MATLAB: the ARD parameter starts from 0


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


def _kernel(z, w2, m0, L0, *, f_orig, opts: VBOptions, lower):
    """Per-voxel closures (all in scaled coordinates u = theta / D, D = |m0|)."""
    n = z.shape[0]
    P = m0.shape[0]
    s0, c0 = opts.s0, opts.c0
    ard = opts.ard
    D = jnp.where(jnp.abs(m0) > 0, jnp.abs(m0), 1.0)

    def f(u):
        return f_orig(u * D)

    if opts.jacobian == "autodiff":
        jac = jax.jacfwd(f)
    else:
        jac = partial(_jac_numeric, f, D=D)

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
        if opts.nonneg:  # MATLAB: m >= 0; models may set other lower bounds
            m_new = jnp.maximum(m_new, lower / D)
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
        return jac(u), z - f(u)

    return dict(D=D, f=f, update=update, free_energy=free_energy, evaluate=evaluate)


def _vb_init(z, w2, m0, L0, m_init, **kw):
    """Step 1 (MATLAB: from the prior mean; warm starts pass m_init)."""
    K = _kernel(z, w2, m0, L0, **kw)
    opts = kw["opts"]
    s0, c0, ard = opts.s0, opts.c0, opts.ard
    mi = m_init / K["D"]
    if ard >= 0 and opts.ard_init_zero:
        mi = mi.at[ard].set(0.0)
    J0, k0 = K["evaluate"](mi)
    L0u = L0 * K["D"][:, None] * K["D"][None, :]
    if ard >= 0:
        L0u = L0u.at[ard, ard].set(1e-6 * K["D"][ard] ** 2)
    L, m, c, s = K["update"](s0, c0, J0, L0u, k0, mi)
    J, k = K["evaluate"](m)
    F = K["free_energy"](s, c, m, L, k, J)
    dtype = z.dtype
    return dict(
        count=jnp.array(1), trials=jnp.array(0),
        F_new=F, F_old=jnp.array(-jnp.inf, dtype),
        m=m, L=L, s=s, c=jnp.asarray(c, dtype), J=J, k=k,
        # best iterate: any loop iterate replaces it (MATLAB keeps only those), but if
        # the loop cannot run (non-finite F) the step-1 solution is returned, not zeros
        bF=jnp.array(-jnp.inf, dtype), bm=m, bL=L,
        bs=jnp.asarray(s, dtype), bc=jnp.asarray(c, dtype),
    )


def _active(st, opts):
    # |dF| below the rounding level of F cannot be resolved (matters in float32)
    eps = jnp.finfo(st["F_new"].dtype).eps
    tol = jnp.maximum(opts.tol, 10 * eps * jnp.abs(st["F_new"]))
    return ((st["count"] < opts.max_iter)
            & (jnp.abs(st["F_new"] - st["F_old"]) > tol)
            & (st["trials"] < opts.max_trials))


def _vb_run(st, z, w2, m0, L0, limit, **kw):
    """Iterate (MATLAB trialsMethodW loop) until the stop rules or ``count == limit``."""
    K = _kernel(z, w2, m0, L0, **kw)
    opts = kw["opts"]

    def body(st):
        trials = jnp.where(st["F_new"] > st["F_old"], 0, st["trials"] + 1)
        L, m, c, s = K["update"](st["s"], st["c"], st["J"], st["L"], st["k"], st["m"])
        J, k = K["evaluate"](m)
        F = K["free_energy"](s, c, m, L, k, J)
        better = F > st["bF"]
        return dict(
            count=st["count"] + 1, trials=trials, F_new=F, F_old=st["F_new"],
            m=m, L=L, s=s, c=jnp.asarray(c, st["F_new"].dtype), J=J, k=k,
            bF=jnp.where(better, F, st["bF"]),
            bm=jnp.where(better, m, st["bm"]),
            bL=jnp.where(better, L, st["bL"]),
            bs=jnp.where(better, s, st["bs"]),
            bc=jnp.where(better, c, st["bc"]),
        )

    return jax.lax.while_loop(lambda st: _active(st, opts) & (st["count"] < limit), body, st)


def _vb_final(st, z, w2, m0, L0, **kw):
    K = _kernel(z, w2, m0, L0, **kw)
    D = K["D"]
    DD = D[:, None] * D[None, :]
    m_best = st["bm"] * D
    return dict(m=m_best, cov=jnp.linalg.inv(st["bL"]) * DD, L=st["bL"] / DD,
                s=st["bs"], c=st["bc"], F=st["bF"], n_iter=st["count"],
                fit=K["f"](st["bm"]))


def make_vb_fns(model, inp, frames, opts: VBOptions = VBOptions()):
    """Jitted, voxel-batched (init, run, active, final) functions."""

    def f_orig(theta):
        return model.predict(theta, inp, frames)

    kw = dict(f_orig=f_orig, opts=opts,
              lower=jnp.asarray(model.lower_bounds, dtype=inp.tp.dtype))
    init = jax.jit(jax.vmap(partial(_vb_init, **kw)))
    run = jax.jit(jax.vmap(partial(_vb_run, **kw)))
    active = jax.jit(jax.vmap(partial(_active, opts=opts)))
    final = jax.jit(jax.vmap(partial(_vb_final, **kw)))
    return init, run, active, final


def _take(tree, idx):
    return jax.tree_util.tree_map(lambda x: x[idx], tree)


def _put(tree, idx, sub):
    return jax.tree_util.tree_map(lambda x, y: x.at[idx].set(y), tree, sub)


def fit_vb(model, inp, frames, Z, W2, M0, L0, opts: VBOptions = VBOptions(),
           chunk_size=4096, dtype=jnp.float64, progress=None, M_init=None, segment=8):
    """Fit V TACs in chunks. ``W2``, ``M0``, ``L0`` may be shared (no V axis).

    ``W2`` is the inverse data variance per frame (0 = frame ignored).
    Within a chunk all voxels iterate ``segment`` steps at a time; voxels still
    running are then gathered into smaller blocks (chunk/4, /16, ...), so a few
    slow voxels do not keep a whole chunk iterating (vmap runs each block until
    its slowest member stops). Results are identical to a plain per-voxel loop.
    Returns a dict of numpy arrays with a leading V axis.
    """
    Z = np.asarray(Z)
    V, n = Z.shape
    P = model.n_params
    W2 = np.broadcast_to(np.asarray(W2), (V, n))
    M0 = np.broadcast_to(np.asarray(M0), (V, P))
    M_init = M0 if M_init is None else np.broadcast_to(np.asarray(M_init), (V, P))
    L0 = np.broadcast_to(np.asarray(L0), (V, P, P))

    cast = lambda a: jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype), a)
    init, run, active, final = make_vb_fns(model, cast(inp), cast(frames), opts)

    chunk = int(min(chunk_size, V))
    min_block = max(16, chunk // 256)
    outs = []
    for i0 in range(0, V, chunk):
        sl = slice(i0, min(i0 + chunk, V))
        nv = sl.stop - sl.start
        args = [Z[sl], W2[sl], M0[sl], L0[sl], M_init[sl]]
        if nv < chunk:
            # pad with copies of the first row to keep a fixed (compiled) shape
            args = [np.concatenate([a, np.repeat(a[:1], chunk - nv, axis=0)]) for a in args]
        z, w2, m0, l0, mi = [jnp.asarray(a, dtype) for a in args]
        st = init(z, w2, m0, l0, mi)
        data = (z, w2, m0, l0)
        while True:
            act = np.flatnonzero(np.asarray(active(st)))
            if act.size == 0:
                break
            block = chunk
            while block // 4 >= max(act.size, min_block):
                block //= 4
            sel = np.concatenate([act, np.repeat(act[:1], block - act.size)]) if \
                block > act.size else act
            sel = jnp.asarray(sel)
            sub = _take(st, sel)
            limit = sub["count"] + segment
            sub = run(sub, *_take(data, sel), limit)
            st = _put(st, sel, sub)
        res = final(st, *data)
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
    with np.errstate(divide="ignore"):
        prec = 1.0 / sd**2
    L0 = np.zeros(m_roi.shape + (m_roi.shape[-1],))
    d = np.arange(m_roi.shape[-1])
    L0[..., d, d] = prec  # no 0*inf off the diagonal
    return m_roi, L0
