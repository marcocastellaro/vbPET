"""Convolution of an input function with exponentials on a uniform time grid.

Two discretisations are provided:

* ``"rect"``: what the MATLAB code does,
  ``dt * filter(a*exp(-b*tp), 1, Cp)``, i.e. a rectangle rule
  ``y[n] = sum_{j<=n} dt * exp(-b*dt*(n-j)) * x[j]``.
* ``"linear"``: exact convolution of ``exp(-b t)`` with the piecewise-linear
  interpolant of the input (more accurate on coarse grids).

Both are written as the linear recurrence ``y[n] = E*y[n-1] + u[n]`` and
evaluated with ``lax.associative_scan`` (log-depth, GPU friendly).
"""

import jax
import jax.numpy as jnp


def _linear_recurrence(decay, u):
    """y[n] = decay * y[n-1] + u[n], y[-1] = 0, with scalar ``decay``."""

    def combine(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return a_l * a_r, a_r * b_l + b_r

    a = jnp.full_like(u, decay)
    _, y = jax.lax.associative_scan(combine, (a, u))
    return y


def exp_conv(x, b, dt, method="rect"):
    """Convolve ``x`` (uniform grid starting at t=0, step ``dt``) with exp(-b t).

    Returns an array with the same shape as ``x``; ``b`` may be 0.
    """
    E = jnp.exp(-b * dt)
    if method == "rect":
        return _linear_recurrence(E, dt * x)
    if method == "linear":
        # I0 = int_0^dt e^{-b(dt-u)} du, I1 = int_0^dt e^{-b(dt-u)} u/dt du
        bdt = b * dt
        small = jnp.abs(bdt) < 1e-4
        bdt_s = jnp.where(small, 1.0, bdt)
        I0 = jnp.where(small, dt * (1 - bdt / 2 + bdt**2 / 6), -jnp.expm1(-bdt_s) / bdt_s * dt)
        I1 = jnp.where(
            small,
            dt * (0.5 - bdt / 6 + bdt**2 / 24),
            dt * (1.0 / bdt_s + jnp.expm1(-bdt_s) / bdt_s**2),
        )
        x_prev = jnp.concatenate([jnp.zeros_like(x[:1]), x[:-1]])
        u = x_prev * (I0 - I1) + x * I1
        u = u.at[0].set(0.0)  # y(0) = 0
        return _linear_recurrence(E, u)
    raise ValueError(f"unknown convolution method {method!r}")


def cumtrapz(x, dt):
    """Cumulative trapezoid with initial 0 (MATLAB ``cumtrapz``) on a uniform grid."""
    inc = 0.5 * dt * (x[1:] + x[:-1])
    return jnp.concatenate([jnp.zeros_like(x[:1]), jnp.cumsum(inc)])


def sample_frames(y, tp, frames, mode="mid"):
    """Sample a fine-grid curve at the PET frames.

    ``frames`` is a dict with ``mid``, ``start``, ``end`` arrays (same time units as
    ``tp``). ``mode="mid"`` interpolates at mid-frame (MATLAB behaviour);
    ``mode="mean"`` returns the frame-averaged value.
    """
    if mode == "mid":
        return jnp.interp(frames["mid"], tp, y)
    if mode == "mean":
        dt = tp[1] - tp[0]
        Y = cumtrapz(y, dt)
        a = jnp.interp(frames["start"], tp, Y)
        b = jnp.interp(frames["end"], tp, Y)
        return (b - a) / (frames["end"] - frames["start"])
    raise ValueError(f"unknown frame sampling mode {mode!r}")
