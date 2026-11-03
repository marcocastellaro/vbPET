"""Benchmark: MATLAB vbEstimation vs JAX fit_vb (2TCM).

Usage: python tests/bench_vs_matlab.py [n_matlab] [n_jax]
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).parent))
from synthetic import make_problem  # noqa: E402
from test_matlab_compat import HERE, UTIL  # noqa: E402
from vbpet.vb.core import VBOptions, fit_vb, regional_prior  # noqa: E402

n_mat = int(sys.argv[1]) if len(sys.argv) > 1 else 200
n_jax = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000

prob = make_problem("2TCM", n_vox=max(n_mat, 1), seed=1)
m0, L0 = regional_prior(prob["p_true"], 0.5)
w2 = prob["weights"] ** 2

# n_mat == 0: no MATLAB here (e.g. DEI), use the throughput measured locally
MATLAB_LOCAL_VOX_PER_S = 312.1  # 2TCM, i9-13900K, R2026b, 2026-11-03
if n_mat == 0:
    n_mat, t_mat = 1, 1 / MATLAB_LOCAL_VOX_PER_S
else:
  with tempfile.TemporaryDirectory() as td:
      fin, fout = Path(td) / "in.mat", Path(td) / "out.mat"
      inp, fr = prob["inp"], prob["frames"]
      scipy.io.savemat(fin, dict(tp=np.asarray(inp.tp), Cp=np.asarray(inp.Cp), Cb=np.asarray(inp.Cb),
                                 t=np.asarray(fr.mid), weights=prob["weights"], s0=1e6, c0=50.0,
                                 ard=np.zeros(1), model="2TCM", theta_pred=prob["theta"][:2].T,
                                 m0=prob["p_true"], **{"lambda": 0.5}, Z=prob["Z"].T))
      subprocess.run(["matlab", "-batch", f"addpath('{HERE / 'matlab_ref'}'); "
                      f"run_reference('{fin}', '{fout}', '{UTIL}')"], check=True)
      t_mat = float(scipy.io.loadmat(fout)["elapsed"].squeeze())
print(f"MATLAB : {n_mat} voxels in {t_mat:.1f} s -> {n_mat / t_mat:.1f} vox/s")

big = make_problem("2TCM", n_vox=n_jax, seed=2)
for dev in ["gpu", "cpu"]:
    try:
        device = jax.devices(dev)[0]
    except RuntimeError:
        continue
    for dtype, chunk in [(jnp.float64, 16384), (jnp.float32, 32768)]:
        with jax.default_device(device):
            kw = dict(opts=VBOptions(), chunk_size=chunk, dtype=dtype)
            fit_vb(big["model"], big["inp"], big["frames"], big["Z"][:chunk], w2, m0, L0, **kw)
            t0 = time.perf_counter()
            res = fit_vb(big["model"], big["inp"], big["frames"], big["Z"], w2, m0, L0, **kw)
            dt = time.perf_counter() - t0
        print(f"JAX {dev} {jnp.dtype(dtype).name}: {n_jax} voxels in {dt:.1f} s -> "
              f"{n_jax / dt:.0f} vox/s (x{(n_jax / dt) / (n_mat / t_mat):.0f}), "
              f"median iters {np.median(res['n_iter']):.0f}, "
              f"nonfinite {np.sum(~np.isfinite(res['F']))}")
