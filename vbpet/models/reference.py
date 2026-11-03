"""Reference-region models (no blood sampling).

The reference TAC plays the role of the input: it is interpolated on the
uniform grid (``reference_input``) and stored in ``PlasmaInput.Cp`` (Cb unused).

SRTM (Lammertsma & Hume 1996), parameters (R1, k2, BPnd):
    C_T = R1 C_R + (k2 - R1 k2a) C_R ⊗ exp(-k2a t),   k2a = k2 / (1 + BPnd)

SRTM2 (Wu & Carson 2002), k2' fixed, parameters (R1, BPnd):
    C_T = R1 C_R + R1 (k2' - k2a) C_R ⊗ exp(-k2a t),  k2a = R1 k2' / (1 + BPnd)
"""

from functools import partial

import jax.numpy as jnp
import numpy as np

from ..conv import exp_conv, sample_frames
from .base import Model, PlasmaInput


def reference_input(frames, tac, t_end=None, step=1 / 60):
    """Interpolate a frame-wise reference TAC on a uniform grid from t=0.

    Frame values are placed at mid-frame, with C_R(0) = 0 and the last value
    held to ``t_end``.
    """
    mid = np.asarray(frames.mid, float)
    tac = np.asarray(tac, float)
    t_end = float(frames.end[-1]) if t_end is None else t_end
    tp = np.arange(0.0, t_end + 2 * step, step)
    Cr = np.interp(tp, np.r_[0.0, mid], np.r_[0.0, tac])
    return PlasmaInput(tp, Cr, np.zeros_like(Cr))


def _sample(y, inp, frames, sampling):
    return sample_frames(y, inp.tp, frames._asdict(), sampling)


def predict_srtm(theta, inp, frames, conv="linear", sampling="mean"):
    R1, k2, BP = theta
    k2a = k2 / (1 + BP)
    dt = inp.tp[1] - inp.tp[0]
    y = R1 * inp.Cp + (k2 - R1 * k2a) * exp_conv(inp.Cp, k2a, dt, conv)
    return _sample(y, inp, frames, sampling)


def predict_srtm2(theta, inp, frames, k2p, conv="linear", sampling="mean"):
    R1, BP = theta
    k2a = R1 * k2p / (1 + BP)
    dt = inp.tp[1] - inp.tp[0]
    y = R1 * inp.Cp + R1 * (k2p - k2a) * exp_conv(inp.Cp, k2a, dt, conv)
    return _sample(y, inp, frames, sampling)


AVAILABLE = ("SRTM", "SRTM2")


def get_model(name, conv="linear", sampling="mean", k2p=None):
    if name == "SRTM":
        return Model("SRTM", ("R1", "k2", "BPnd"), (1.0, 0.3, 1.0),
                     partial(predict_srtm, conv=conv, sampling=sampling),
                     {"k2p": lambda t: t[1] / t[0]})
    if name == "SRTM2":
        if k2p is None:
            raise ValueError("SRTM2 needs k2p (k2 of the reference region, 1/min)")
        k2p = float(k2p)
        return Model("SRTM2", ("R1", "BPnd"), (1.0, 1.0),
                     partial(predict_srtm2, k2p=k2p, conv=conv, sampling=sampling),
                     {"k2": lambda t: t[0] * k2p}, fixed={"k2p": k2p})
    raise KeyError(f"unknown reference-region model {name!r}; available: {AVAILABLE}")


def k2p_from_srtm(kinpar, min_bp=0.0, exclude=()):
    """k2' = median of k2/R1 over regional SRTM fits with BPnd > ``min_bp``."""
    ok = (kinpar["BPnd"] > min_bp) & (kinpar["R1"] > 0) & ~kinpar["name"].isin(exclude)
    if not ok.any():
        ok = kinpar["R1"] > 0
    return float(jnp.median(jnp.asarray((kinpar["k2"] / kinpar["R1"])[ok].to_numpy())))
