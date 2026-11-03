"""Unit tests: convolution, models, WNLLS and VB on noise-free synthetic data."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from synthetic import make_problem
from vbpet.conv import exp_conv, sample_frames
from vbpet.models import get_model
from vbpet.models.base import Frames, PlasmaInput
from vbpet.models.reference import reference_input
from vbpet.regional import wnlls
from vbpet.vb.core import VBOptions, fit_vb, regional_prior


@pytest.mark.parametrize("b", [0.0, 0.05, 0.7, 5.0])
def test_exp_conv_linear_matches_analytic(b):
    a = 0.3
    dt = 0.01
    t = np.arange(0, 30, dt)
    x = np.exp(-a * t)
    exact = t if b == a else (np.exp(-a * t) - np.exp(-b * t)) / (b - a)
    if b == 0.0:
        exact = (1 - np.exp(-a * t)) / a
    y = np.asarray(exp_conv(jnp.asarray(x), b, dt, "linear"))
    np.testing.assert_allclose(y, exact, atol=2e-6 * exact.max())


def test_exp_conv_rect_is_matlab_filter():
    rng = np.random.default_rng(0)
    x = rng.random(200)
    b, dt = 0.4, 0.1
    k = np.exp(-b * dt * np.arange(200))
    ref = dt * np.convolve(x, k)[:200]  # = dt * filter(exp(-b*tp), 1, x)
    np.testing.assert_allclose(np.asarray(exp_conv(jnp.asarray(x), b, dt, "rect")), ref,
                               rtol=1e-12)


def test_frame_mean_of_linear_curve_is_mid_value():
    tp = jnp.linspace(0, 10, 1001)
    y = 2 * tp + 1
    fr = Frames.from_start_duration(jnp.array([0.0, 1.0, 3.0]), jnp.array([1.0, 2.0, 5.0]))
    m = sample_frames(y, tp, fr._asdict(), "mean")
    np.testing.assert_allclose(np.asarray(m), 2 * np.asarray(fr.mid) + 1, rtol=1e-10)


def test_fix_matches_full_model():
    prob = make_problem("2TCM", n_vox=2)
    full = get_model("2TCM", conv="rect", sampling="mid")
    fixed = full.fix({"Vb": 0.03})
    th = np.array([0.1, 0.2, 0.05, 0.02])
    a = fixed.predict(jnp.asarray(th), prob["inp"], prob["frames"])
    b = full.predict(jnp.asarray(np.r_[th, 0.03]), prob["inp"], prob["frames"])
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12)
    assert fixed.param_names == ("K1", "k2", "k3", "k4")
    v = fixed.derived["VT"](jnp.asarray(th))
    np.testing.assert_allclose(float(v), 0.1 / 0.2 * (1 + 0.05 / 0.02))


def _fine_frames():
    start = np.r_[np.arange(0, 2, 0.25), np.arange(2, 10, 1.0), np.arange(10, 90, 5.0)]
    dur = np.diff(np.r_[start, 90.0])
    return Frames.from_start_duration(start, dur)


def test_srtm_identity_and_recovery():
    frames = _fine_frames()
    t = np.arange(0, 92, 1 / 60)
    from synthetic import feng_input
    cp = feng_input(t)
    pin = PlasmaInput(t, cp, np.zeros_like(cp))
    one = get_model("1TCM", conv="linear", sampling="mean")
    # reference: K1'=0.1, k2'=0.2 ; target: K1=0.12, k2a=0.06 -> BP = k2/k2a - 1
    ref = np.asarray(one.predict(jnp.array([0.1, 0.2, 0.0]), pin, frames))
    tgt = np.asarray(one.predict(jnp.array([0.12, 0.06, 0.0]), pin, frames))
    R1, k2 = 0.12 / 0.1, 0.12 / 0.1 * 0.2
    BP = k2 / 0.06 - 1
    inp = reference_input(frames, ref, step=1 / 120)  # mid-frames on the grid
    srtm = get_model("SRTM", conv="linear", sampling="mid")
    same = srtm.predict(jnp.array([1.0, 0.3, 0.0]), inp, frames)
    np.testing.assert_allclose(np.asarray(same), ref, rtol=1e-10)
    r = wnlls(get_model("SRTM"), inp, frames, tgt, np.ones_like(tgt))
    np.testing.assert_allclose(r["par"], [R1, k2, BP], rtol=0.03)


def test_wnlls_and_vb_recover_truth_noise_free():
    prob = make_problem("2TCM", n_vox=4, seed=3, noise=0.0)
    model, inp, fr = prob["model"], prob["inp"], prob["frames"]
    th = prob["theta"]
    z = np.asarray(jax.vmap(lambda t: model.predict(t, inp, fr))(th))
    w = np.ones(z.shape[1])
    r = wnlls(model, inp, fr, z[0], w)
    vt = lambda t: t[0] / t[1] * (1 + t[2] / t[3])
    assert abs(vt(r["par"]) / vt(th[0]) - 1) < 1e-3
    m0, L0 = regional_prior(th.mean(0), lam=2.0)  # weak prior
    res = fit_vb(model, inp, fr, z, w, m0, L0, VBOptions())
    vts = np.array([vt(m) for m in res["m"]])
    vtt = np.array([vt(t) for t in th])
    assert np.max(np.abs(vts / vtt - 1)) < 0.02
    assert np.all(np.isfinite(res["F"]))


def test_vb_float32_converges_like_float64():
    prob = make_problem("2TCM", n_vox=64, seed=4)
    m0, L0 = regional_prior(prob["p_true"], 0.5)
    w2 = prob["weights"] ** 2
    args = (prob["model"], prob["inp"], prob["frames"], prob["Z"], w2, m0, L0, VBOptions())
    r64 = fit_vb(*args, dtype=jnp.float64)
    r32 = fit_vb(*args, dtype=jnp.float32)
    assert np.median(r32["n_iter"]) <= np.median(r64["n_iter"]) + 3
    rel = np.abs(r32["m"] - r64["m"]) / np.sqrt(np.einsum("vii->vi", r64["cov"]))
    assert np.median(rel) < 0.05


def test_spatial_prior_reduces_error_on_smooth_field():
    from vbpet.vb.spatial import fit_spatial, neighbours
    prob = make_problem("1TCM", n_vox=1, seed=5)
    model, inp, fr = prob["model"], prob["inp"], prob["frames"]
    nx = 16
    x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, nx), indexing="ij")
    K1 = 0.05 * (1 + 0.4 * np.sin(2 * np.pi * x) * np.cos(np.pi * y))
    th = np.stack([K1.ravel(), np.full(nx * nx, 0.18), np.full(nx * nx, 0.05)], 1)
    clean = np.asarray(jax.vmap(lambda t: model.predict(t, inp, fr))(th))
    rng = np.random.default_rng(0)
    sd_noise = 0.15 * clean.mean(0) + 1e-3
    Z = np.abs(clean + rng.normal(size=clean.shape) * sd_noise)
    w2 = 1 / sd_noise**2
    mu = np.broadcast_to(np.array([0.05, 0.18, 0.05]), th.shape).copy()
    sd = 0.5 * mu
    idx = np.arange(nx * nx)
    nb = neighbours(idx, (nx, nx, 1))
    L0 = np.einsum("vi,ij->vij", 1 / sd**2, np.eye(3))
    r_reg = fit_vb(model, inp, fr, Z, w2, mu, L0)
    r_sp = fit_spatial(model, inp, fr, Z, w2, mu, sd, "SRR", nb, init=r_reg, n_sweeps=6)
    err = lambda r: np.sqrt(np.mean((r["m"][:, 0] - th[:, 0]) ** 2))
    assert np.isfinite(r_sp["phi"][0])
    # measured on DEI: RMSE 0.00574 -> 0.00482 (-16%), free energy higher
    assert err(r_sp) < 0.9 * err(r_reg), (err(r_sp), err(r_reg))
    assert np.median(r_sp["F"] - r_reg["F"]) > 0


def test_delay_recovered_by_wnlls_and_vb():
    prob = make_problem("irr2TCM", n_vox=1, seed=6)
    base = prob["model"]
    model = base.with_delay()
    assert model.param_names[-1] == "delay" and model.lower_bounds[-1] < 0
    inp, fr = prob["inp"], prob["frames"]
    th = np.array([0.1, 0.15, 0.05, 0.05, 0.25])  # +0.25 min delay
    z = np.asarray(model.predict(jnp.asarray(th), inp, fr))
    r = wnlls(model, inp, fr, z, np.ones_like(z))
    np.testing.assert_allclose(r["par"][-1], 0.25, atol=0.02)
    m0, L0 = regional_prior(np.array([0.1, 0.15, 0.05, 0.05, 0.0]), lam=0.5)
    L0[-1, -1] = 1 / 0.3**2
    res = fit_vb(model, inp, fr, z[None], np.ones_like(z), m0, L0)
    assert abs(res["m"][0, -1] - 0.25) < 0.05
    neg = np.r_[th[:4], -0.2]
    zn = np.asarray(model.predict(jnp.asarray(neg), inp, fr))
    rn = wnlls(model, inp, fr, zn, np.ones_like(zn))
    assert rn["par"][-1] < -0.1  # negative delays allowed
