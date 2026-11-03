"""Input functions on a uniform grid (minutes)."""

import numpy as np

from .models.base import PlasmaInput


def grid_step(t, cp, rule="auto", max_step=10 / 60, min_step=1 / 60):
    """Grid step in minutes.

    ``"matlab"``: t_peak/3 (MAIN_parametricMaps_VB.m). ``"auto"``: the same rule
    clipped to [min_step, max_step] (default 1-10 s). A float is used as is.
    """
    if isinstance(rule, (int, float)):
        return float(rule)
    t_peak = float(t[np.nanargmax(cp)])
    dt = t_peak / 3 if t_peak > 0 else max_step
    if rule == "matlab":
        return dt
    return float(np.clip(dt, min_step, max_step))


def make_plasma_input(blood, t_end, step="auto", scale=1.0, shift=0.0):
    """Interpolate blood curves on [0, t_end + step] (zero before the first sample).

    ``scale`` multiplies activities (unit harmonisation with the PET image).
    ``shift`` (minutes) delays the blood curves: C_used(t) = C(t - shift)
    (negative = blood earlier, e.g. kinfitr/petfit ``inpshift``).
    """
    t = np.asarray(blood.t, float) + shift
    order = np.argsort(t)
    t = t[order]
    cp = np.asarray(blood.plasma_parent, float)[order] * scale
    cb = np.asarray(blood.whole_blood, float)[order] * scale
    if t[0] > 0:  # anchor at time 0 with zero activity
        t, cp, cb = np.r_[0.0, t], np.r_[0.0, cp], np.r_[0.0, cb]
    dt = grid_step(t, cp, step)
    tp = np.arange(0.0, t_end + 2 * dt, dt)
    # beyond the last sample: hold the last value (BIDS curves usually cover the scan)
    Cp = np.interp(tp, t, cp)
    Cb = np.interp(tp, t, cb)
    return PlasmaInput(tp, Cp, Cb)
