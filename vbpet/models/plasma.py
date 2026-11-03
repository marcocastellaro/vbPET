"""Plasma-input compartment models (ports of MATLAB ``computeModel_*``).

All models: tissue = (1-Vb) * (Cp ⊗ IRF) + Vb * Cb, computed on the uniform
input grid and then sampled at the PET frames. Time in minutes.
"""

from functools import partial

import jax.numpy as jnp

from ..conv import cumtrapz, exp_conv, sample_frames
from .base import Model


def _finish(YY, inp, frames, sampling):
    return sample_frames(YY, inp.tp, frames._asdict(), sampling)


def _expo(inp, a, b, conv):
    dt = inp.tp[1] - inp.tp[0]
    return a * exp_conv(inp.Cp, b, dt, conv)


def _integ(inp, a, conv):
    dt = inp.tp[1] - inp.tp[0]
    if conv == "rect":
        return a * cumtrapz(inp.Cp, dt)  # MATLAB uses cumtrapz for the irreversible term
    return a * exp_conv(inp.Cp, 0.0, dt, "linear")


def _2tcm_rates(k2, k3, k4):
    s = k2 + k3 + k4
    d = jnp.sqrt(jnp.maximum(s**2 - 4 * k2 * k4, 0.0))
    return (s - d) / 2, (s + d) / 2


def predict_1tcm(theta, inp, frames, conv="rect", sampling="mid"):
    K1, k2, Vb = theta
    YY = _expo(inp, (1 - Vb) * K1, k2, conv) + Vb * inp.Cb
    return _finish(YY, inp, frames, sampling)


def predict_2tcm(theta, inp, frames, conv="rect", sampling="mid"):
    K1, k2, k3, k4, Vb = theta
    b1, b2 = _2tcm_rates(k2, k3, k4)
    a1 = (1 - Vb) * K1 * (b1 - k3 - k4) / (b1 - b2)
    a2 = (1 - Vb) * K1 * (k3 + k4 - b2) / (b1 - b2)
    YY = _expo(inp, a1, b1, conv) + _expo(inp, a2, b2, conv) + Vb * inp.Cb
    return _finish(YY, inp, frames, sampling)


def predict_2tcm1k(theta, inp, frames, conv="rect", sampling="mid"):
    K1, k2, k3, k4, Ki, Vb = theta
    b1, b2 = _2tcm_rates(k2, k3, k4)
    a1 = (1 - Vb) * K1 * (b1 - k3 - k4) / (b1 - b2)
    a2 = (1 - Vb) * K1 * (k3 + k4 - b2) / (b1 - b2)
    # MATLAB: c3 = Vb*Ki*cumtrapz(tp, Cp) (vascular trapping term)
    YY = (_expo(inp, a1, b1, conv) + _expo(inp, a2, b2, conv)
          + _integ(inp, Vb * Ki, conv) + Vb * inp.Cb)
    return _finish(YY, inp, frames, sampling)


def predict_irr2tcm(theta, inp, frames, conv="rect", sampling="mid"):
    K1, k2, k3, Vb = theta
    b1 = k2 + k3
    a1 = (1 - Vb) * K1 * k2 / (k2 + k3)
    a2 = (1 - Vb) * K1 * k3 / (k2 + k3)
    YY = _expo(inp, a1, b1, conv) + _integ(inp, a2, conv) + Vb * inp.Cb
    return _finish(YY, inp, frames, sampling)


def predict_1k(theta, inp, frames, conv="rect", sampling="mid"):
    K1, Vb = theta
    YY = _integ(inp, (1 - Vb) * K1, conv) + Vb * inp.Cb
    return _finish(YY, inp, frames, sampling)


def predict_5k(theta, inp, frames, conv="rect", sampling="mid"):
    K1, k2, k3, k4, k5, Vb = theta
    s = k2 + k3 + k4 + k5
    q = k2 * k4 + k2 * k5 + k3 * k5
    d = jnp.sqrt(jnp.maximum(s**2 - 4 * q, 0.0))
    b1, b2 = (s - d) / 2, (s + d) / 2
    a3 = (1 - Vb) * (K1 * k3 * k5) / (b1 * b2)
    a2 = (1 - Vb) * (K1 * k4 + K1 * k5 + K1 * k3 - K1 * b2 - a3 * b1) / (b1 - b2)
    a1 = (1 - Vb) * (K1 - a2 - a3)
    YY = (_expo(inp, a1, b1, conv) + _expo(inp, a2, b2, conv)
          + _integ(inp, a3, conv) + Vb * inp.Cb)
    return _finish(YY, inp, frames, sampling)


def _vt_1tcm(t):
    return t[0] / t[1]


def _vt_2tcm(t):
    return t[0] / t[1] * (1 + t[2] / t[3])


def _ki_irr(t):
    return t[0] * t[2] / (t[1] + t[2])


def _ki_5k(t):
    K1, k2, k3, k4, k5 = t[:5]
    return K1 * k3 * k5 / (k2 * k4 + k2 * k5 + k3 * k5)


# name -> (param names, MATLAB default p0, predict fn, derived)
_SPECS = {
    "1TCM": (("K1", "k2", "Vb"), (0.05, 0.18, 0.05), predict_1tcm, {"VT": _vt_1tcm}),
    "2TCM": (("K1", "k2", "k3", "k4", "Vb"), (0.05, 0.18, 0.05, 0.03, 0.05),
             predict_2tcm, {"VT": _vt_2tcm}),
    "2TCM1K": (("K1", "k2", "k3", "k4", "Ki", "Vb"), (0.05, 0.18, 0.05, 0.03, 0.07, 0.05),
               predict_2tcm1k, {"VT": _vt_2tcm}),
    "irr2TCM": (("K1", "k2", "k3", "Vb"), (0.05, 0.18, 0.05, 0.05),
                predict_irr2tcm, {"Ki": _ki_irr}),
    "1K": (("K1", "Vb"), (0.05, 0.05), predict_1k, {"Ki": lambda t: t[0]}),
    "5K": (("K1", "k2", "k3", "k4", "k5", "Vb"), (0.05, 0.18, 0.05, 0.03, 0.02, 0.05),
           predict_5k, {"Ki": _ki_5k}),
}

ALIASES = {"3K": "irr2TCM", "2TCM-1K": "2TCM1K"}


def get_model(name, conv="rect", sampling="mid"):
    """Build a plasma-input model. ``conv``: rect|linear, ``sampling``: mid|mean."""
    name = ALIASES.get(name, name)
    if name not in _SPECS:
        raise KeyError(f"unknown plasma-input model {name!r}; available: {sorted(_SPECS)}")
    pnames, p0, fn, derived = _SPECS[name]
    return Model(name, pnames, p0, partial(fn, conv=conv, sampling=sampling), derived)


AVAILABLE = tuple(_SPECS)
