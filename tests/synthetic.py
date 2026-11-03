"""Synthetic inputs/TACs shared by the tests and benchmarks."""

import numpy as np

# ds004869-like framing (seconds)
FRAME_START_S = np.array([0, 30, 60, 90, 120, 150, 180, 240, 300, 360, 480, 600, 900,
                          1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900,
                          4200, 4500, 4800, 5100, 5400, 5700, 6000, 6300, 6600, 6900],
                         float)
FRAME_DUR_S = np.array([30] * 6 + [60] * 3 + [120] * 2 + [300] * 22, float)


def feng_input(t, A=(851.1, 21.88, 20.81), lam=(-4.134, -0.1191, -0.0104), tau=0.3):
    """Feng (1993) plasma input, t in minutes (zero before tau)."""
    tt = np.maximum(t - tau, 0.0)
    A1, A2, A3 = A
    l1, l2, l3 = lam
    cp = (A1 * tt - A2 - A3) * np.exp(l1 * tt) + A2 * np.exp(l2 * tt) + A3 * np.exp(l3 * tt)
    return np.where(t > tau, cp, 0.0)


def make_problem(model_name="2TCM", n_vox=64, seed=0, noise=0.05):
    from vbpet.models.base import Frames, PlasmaInput
    from vbpet.models.plasma import get_model

    rng = np.random.default_rng(seed)
    start = FRAME_START_S / 60
    dur = FRAME_DUR_S / 60
    frames = Frames.from_start_duration(start, dur)
    t_end = float(start[-1] + dur[-1])

    # MATLAB MAIN: virtual grid with step t_peak/3
    t_fine = np.linspace(0, t_end, 20001)
    cp_fine = feng_input(t_fine)
    t_peak = t_fine[np.argmax(cp_fine)]
    tv = t_peak / 3
    tp = np.arange(0, t_end + tv, tv)
    Cp = feng_input(tp)
    Cb = 0.9 * Cp
    inp = PlasmaInput(tp, Cp, Cb)

    model = get_model(model_name)
    p_true = np.array(model.p0)
    theta = p_true * np.exp(rng.normal(0, 0.15, (n_vox, model.n_params)))
    import jax
    clean = np.asarray(jax.vmap(lambda th: model.predict(th, inp, frames))(theta))
    roi = clean.mean(0)
    sd = noise * np.sqrt(np.maximum(roi, 1e-6) / dur) * np.sqrt(dur.mean())
    Z = np.abs(clean + rng.normal(0, 1, clean.shape) * sd)
    weights = np.sqrt(dur / np.maximum(roi, 1e-6))  # MATLAB MAIN weights
    return dict(model=model, inp=inp, frames=frames, theta=theta, Z=Z, roi=roi,
                weights=weights, p_true=p_true)
