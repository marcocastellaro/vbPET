"""Compare the JAX port with the original MATLAB vbPET code.

Requires MATLAB on PATH and the MATLAB sources (``VBPET_MATLAB_DIR``, default
third_party/vbPET_matlab, a copy of the lab code). Results are cached in tests/fixtures/.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from synthetic import make_problem
from vbpet.vb.core import VBOptions, fit_vb, regional_prior

HERE = Path(__file__).parent
FIX = HERE / "fixtures"
UTIL = os.environ.get("VBPET_MATLAB_DIR",
                      str(HERE.parent / "third_party" / "vbPET_matlab" / "UTILITY_VB"))
MATLAB_NAME = {"irr2TCM": "3K"}

pytestmark = pytest.mark.matlab
HAVE_MATLAB = shutil.which("matlab") is not None and Path(UTIL).is_dir()

LAMBDA = 0.5
S0, C0 = 1e6, 50.0


def matlab_reference(model_name, n_vox=32, seed=0):
    prob = make_problem(model_name, n_vox=n_vox, seed=seed)
    FIX.mkdir(exist_ok=True)
    tag = f"{model_name}_n{n_vox}_s{seed}"
    fin, fout = FIX / f"{tag}_in.mat", FIX / f"{tag}_out.mat"
    if not fout.exists():
        if not HAVE_MATLAB:
            pytest.skip("no MATLAB and no cached fixture " + fout.name)
        inp, fr = prob["inp"], prob["frames"]
        scipy.io.savemat(fin, dict(
            tp=np.asarray(inp.tp), Cp=np.asarray(inp.Cp), Cb=np.asarray(inp.Cb),
            t=np.asarray(fr.mid), weights=prob["weights"], s0=S0, c0=C0,
            ard=np.zeros(1), model=MATLAB_NAME.get(model_name, model_name),
            theta_pred=prob["theta"][:8].T, m0=prob["p_true"], lam=LAMBDA,
            **{"lambda": LAMBDA}, Z=prob["Z"].T))
        cmd = (f"addpath('{HERE / 'matlab_ref'}'); "
               f"run_reference('{fin}', '{fout}', '{UTIL}')")
        subprocess.run(["matlab", "-batch", cmd], check=True, timeout=1800)
    return prob, scipy.io.loadmat(fout, squeeze_me=False)


MODELS = ["1TCM", "2TCM", "2TCM1K", "irr2TCM", "1K", "5K"]


@pytest.mark.parametrize("model_name", MODELS)
def test_model_predictions(model_name):
    prob, ref = matlab_reference(model_name)
    import jax
    m = prob["model"]
    th = prob["theta"][:8]
    ours = np.asarray(jax.vmap(lambda t: m.predict(t, prob["inp"], prob["frames"]))(th)).T
    np.testing.assert_allclose(ours, ref["pred"], rtol=1e-9, atol=1e-9 * np.abs(ref["pred"]).max())


@pytest.mark.parametrize("model_name", ["1TCM", "2TCM", "irr2TCM"])
def test_vb_numeric_jacobian_matches_matlab(model_name):
    """With the MATLAB numerical Jacobian the port should reproduce MATLAB closely."""
    prob, ref = matlab_reference(model_name)
    m0, L0 = regional_prior(prob["p_true"], LAMBDA)
    w2 = prob["weights"] ** 2
    res = fit_vb(prob["model"], prob["inp"], prob["frames"], prob["Z"], w2, m0, L0,
                 VBOptions(s0=S0, c0=C0, jacobian="numeric"))
    par_ref = ref["par"].T
    F_ref = ref["FreeEnergy"].ravel()
    # same iterates up to floating point (pinv vs solve): tight agreement
    np.testing.assert_allclose(res["m"], par_ref, rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(res["F"], F_ref, rtol=1e-6, atol=1e-4)


@pytest.mark.parametrize("model_name", ["1TCM", "2TCM", "irr2TCM"])
def test_vb_autodiff_close_to_matlab(model_name):
    prob, ref = matlab_reference(model_name)
    m0, L0 = regional_prior(prob["p_true"], LAMBDA)
    res = fit_vb(prob["model"], prob["inp"], prob["frames"], prob["Z"],
                 prob["weights"] ** 2, m0, L0, VBOptions(s0=S0, c0=C0))
    par_ref = ref["par"].T
    # exact Jacobian: same optimum within a fraction of the posterior sd
    sd = np.sqrt(np.einsum("vii->vi", ref["COV"].transpose(2, 0, 1)))
    assert np.median(np.abs(res["m"] - par_ref) / sd) < 0.05
    # free energy at least as good (up to the stopping tolerance)
    assert np.median(res["F"] - ref["FreeEnergy"].ravel()) > -1e-2
