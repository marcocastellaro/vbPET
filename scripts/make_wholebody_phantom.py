"""Synthetic total-body dynamic FDG phantom in BIDS (for M5 validation).

    python scripts/make_wholebody_phantom.py <out_bids_dir> [--seed 0] [--noise 0.1]

Geometric organs in a 64x64x128 grid (4 mm), irreversible 2TCM (K1, k2, k3,
Vb) per organ with smooth intra-organ variation, blood arrival delay growing
from head to feet, Quadra-like framing (60 min), count-proportional noise.
Writes:
  sub-01/pet/sub-01_pet.nii.gz (+ .json; no blood file: use the aorta IDIF)
  sub-01/pet/sub-01_recording-manual_blood.tsv (true input, for comparison only,
      under derivatives/truth so the app does not pick it up)
  derivatives/seg/sub-01/pet/sub-01_space-pet_seg-organs_dseg.nii.gz (+ .tsv)
  derivatives/truth/sub-01/pet/sub-01_meas-<param>_truth.nii.gz
The values are plausible, not a validated physiological model.
"""

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vbpet.models import get_model  # noqa: E402
from vbpet.models.base import Frames, PlasmaInput  # noqa: E402

SHAPE = (64, 64, 128)
VOX = 4.0  # mm

# label: (name, K1, k2, k3, Vb)  [1/min], literature-like FDG ranges
ORGANS = {
    1: ("aorta", 0.0, 0.0, 0.0, 1.0),
    2: ("left-ventricle", 0.0, 0.0, 0.0, 1.0),
    3: ("myocardium", 0.60, 1.20, 0.10, 0.10),
    4: ("brain", 0.10, 0.13, 0.065, 0.04),
    5: ("liver", 0.80, 0.95, 0.008, 0.25),
    6: ("kidney-left", 0.70, 1.00, 0.02, 0.30),
    7: ("kidney-right", 0.70, 1.00, 0.02, 0.30),
    8: ("spleen", 0.60, 0.90, 0.01, 0.30),
    9: ("muscle", 0.03, 0.20, 0.01, 0.02),
    10: ("fat", 0.01, 0.15, 0.005, 0.01),
    11: ("lesion-1", 0.25, 0.40, 0.12, 0.05),
    12: ("lesion-2", 0.15, 0.50, 0.05, 0.05),
}


def frames_quadra():
    dur = [5] * 12 + [10] * 6 + [30] * 8 + [60] * 6 + [300] * 9  # s, 60 min total
    start = np.r_[0, np.cumsum(dur)[:-1]]
    return np.asarray(start, float), np.asarray(dur, float)


def feng(t, A=(851.1, 21.88, 20.81), lam=(-4.134, -0.1191, -0.0104), tau=0.2):
    tt = np.maximum(t - tau, 0)
    cp = ((A[0] * tt - A[1] - A[2]) * np.exp(lam[0] * tt) + A[1] * np.exp(lam[1] * tt)
          + A[2] * np.exp(lam[2] * tt))
    return np.where(t > tau, cp, 0.0) / 30.0  # kBq/mL scale for FDG


def ellipsoid(c, r):
    x, y, z = np.indices(SHAPE)
    return ((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2 <= 1


def build_labels():
    lab = np.zeros(SHAPE, np.int16)
    body = ellipsoid((32, 32, 60), (24, 16, 70))
    head = ellipsoid((32, 32, 115), (11, 12, 13))
    lab[body | head] = 9  # muscle everywhere inside the body
    fat = body & ~ellipsoid((32, 32, 60), (21, 13, 68))
    lab[fat] = 10
    lab[ellipsoid((32, 32, 116), (9, 10, 10))] = 4  # brain
    lab[ellipsoid((30, 30, 86), (7, 6, 6))] = 3  # myocardium
    lab[ellipsoid((30, 30, 86), (4, 3, 3))] = 2  # left ventricle cavity
    lab[ellipsoid((40, 30, 70), (12, 9, 8))] = 5  # liver
    lab[ellipsoid((22, 36, 66), (5, 4, 7))] = 8  # spleen
    lab[ellipsoid((24, 40, 58), (3, 3, 5))] = 6  # kidneys
    lab[ellipsoid((40, 40, 58), (3, 3, 5))] = 7
    zz = np.arange(SHAPE[2])
    for k in zz[(zz > 20) & (zz < 84)]:  # descending aorta, a tube along z
        lab[np.ix_([31, 32, 33], [38, 39, 40], [k])] = 1
    lab[ellipsoid((38, 28, 72), (2.5, 2.5, 2.5))] = 11  # lesion in the liver
    lab[ellipsoid((28, 26, 40), (2, 2, 2))] = 12  # lesion in muscle (pelvis)
    return lab


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.10, help="relative noise at 1 min frames")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    out = a.out
    lab = build_labels()
    start_s, dur_s = frames_quadra()
    frames = Frames.from_start_duration(start_s / 60, dur_s / 60)
    t = np.arange(0, 61, 1 / 60)
    cp = feng(t)
    inp = PlasmaInput(t, cp, cp)  # FDG: plasma ~ whole blood
    model = get_model("irr2TCM", conv="linear", sampling="mean").with_delay()

    # parameters per voxel: organ value x smooth field (+-20%), delay along z
    idx = np.flatnonzero(lab)
    ijk = np.stack(np.unravel_index(idx, SHAPE), 1).astype(float)
    th = np.array([ORGANS[l][1:] for l in lab.reshape(-1)[idx]], float)
    smooth = 1 + 0.2 * np.sin(ijk[:, 0] / 9) * np.cos(ijk[:, 1] / 11) * np.sin(ijk[:, 2] / 13)
    th[:, :3] *= smooth[:, None]
    delay = 0.05 + 0.25 * (SHAPE[2] - ijk[:, 2]) / SHAPE[2]  # min: head ~0.05, feet ~0.3
    blood = th[:, 3] >= 1.0
    delay[blood] = 0.05 + 0.10 * (SHAPE[2] - ijk[blood, 2]) / SHAPE[2]
    theta = np.c_[th, delay]

    f = jax.jit(jax.vmap(lambda p: model.predict(p, inp, frames)))
    tacs = np.concatenate([np.asarray(f(jnp.asarray(theta[i:i + 65536])))
                           for i in range(0, len(theta), 65536)])
    dur_min = dur_s / 60
    sd = a.noise * np.sqrt(np.clip(tacs, 1e-6, None) * tacs.max() / dur_min) / np.sqrt(tacs.max())
    noisy = np.clip(tacs + rng.normal(size=tacs.shape) * sd, 0, None).astype(np.float32)

    data = np.zeros(SHAPE + (len(dur_s),), np.float32)
    data.reshape(-1, len(dur_s))[idx] = noisy * 1e3  # Bq/mL
    aff = np.diag([VOX, VOX, VOX, 1.0])
    pet = out / "sub-01" / "pet"
    pet.mkdir(parents=True, exist_ok=True)
    (out / "dataset_description.json").write_text(json.dumps(
        {"Name": "vbPET synthetic total-body FDG phantom", "BIDSVersion": "1.10.0"}, indent=2))
    nib.save(nib.Nifti1Image(data, aff), pet / "sub-01_pet.nii.gz")
    (pet / "sub-01_pet.json").write_text(json.dumps({
        "Units": "Bq/mL", "TracerName": "FDG", "FrameTimesStart": start_s.tolist(),
        "FrameDuration": dur_s.tolist(), "ImageDecayCorrected": True,
        "Description": "synthetic phantom (vbpet scripts/make_wholebody_phantom.py)"}, indent=2))
    seg = out / "derivatives" / "seg" / "sub-01" / "pet"
    seg.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(lab, aff), seg / "sub-01_space-pet_seg-organs_dseg.nii.gz")
    pd.DataFrame({"index": list(ORGANS), "name": [v[0] for v in ORGANS.values()]}).to_csv(
        seg / "sub-01_space-pet_seg-organs_dseg.tsv", sep="\t", index=False)
    truth = out / "derivatives" / "truth" / "sub-01" / "pet"
    truth.mkdir(parents=True, exist_ok=True)
    names = list(model.param_names) + ["Ki"]
    vals = np.c_[theta, theta[:, 0] * theta[:, 2] / np.maximum(theta[:, 1] + theta[:, 2], 1e-12)]
    for k, n in enumerate(names):
        vol = np.full(SHAPE, np.nan, np.float32)
        vol.reshape(-1)[idx] = vals[:, k]
        nib.save(nib.Nifti1Image(vol, aff), truth / f"sub-01_meas-{n}_truth.nii.gz")
    pd.DataFrame({"time": t * 60, "plasma_radioactivity": cp,
                  "whole_blood_radioactivity": cp}).to_csv(
        truth / "sub-01_recording-true_blood.tsv", sep="\t", index=False)
    print(f"phantom: {len(idx)} voxels, {len(dur_s)} frames -> {out}")


if __name__ == "__main__":
    main()
