"""Compare voxel VB runs with different iteration limits / stall rule.

    python scripts/iter_study.py <run_dir_100> <run_dir_1000> <run_dir_stall> <model> <derived>
Each run_dir is a vbpet output folder containing exactly one PET run.
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def load(d, model, meas):
    f = sorted(Path(d).glob(f"**/*_model-{model}_meas-{meas}_mimap.nii.gz"))
    return nib.load(f[0]).get_fdata()


def main():
    d100, d1000, dst, model, der = sys.argv[1:6]
    it100, it1000, itst = (load(d, model, "NIter") for d in (d100, d1000, dst))
    F100, F1000, Fst = (load(d, model, "FreeEnergy") for d in (d100, d1000, dst))
    st1000, stst = load(d1000, model, "StopReason"), load(dst, model, "StopReason")
    fl1000 = load(d1000, model, "Oscillations")
    v100, v1000, vst = (load(d, model, der) for d in (d100, d1000, dst))
    fit = np.isfinite(F100) & np.isfinite(F1000) & np.isfinite(Fst)
    hit = fit & (it100 >= 100)
    print(f"voxels {fit.sum()}, hit 100 iterations: {hit.sum()} ({100 * hit.mean() / fit.mean():.1f}%)")
    names = {1: "converged", 2: "trials", 3: "stall", 4: "max_iter"}
    for lab, st in [("max_iter 1000", st1000), ("1000 + stall", stst)]:
        print(f"  {lab}: stop reasons of those voxels:",
              {names[k]: int(np.sum(st[hit] == k)) for k in names})
    print("  iterations used (1000 run), p50/p90/p99:", np.percentile(it1000[hit], [50, 90, 99]).astype(int))
    print("  iterations used (stall run), p50/p90/p99:", np.percentile(itst[hit], [50, 90, 99]).astype(int))
    osc = fl1000[hit] / np.maximum(it1000[hit] - 1, 1)
    print("  oscillation index (sign changes / iteration) p50/p90:", np.percentile(osc, [50, 90]).round(2),
          "| voxels with index > 0.5:", int(np.sum(osc > 0.5)))
    dF = F1000[hit] - F100[hit]
    print("  F gain 100 -> 1000 iterations p50/p90/p99:", np.percentile(dF, [50, 90, 99]).round(3))
    print("  F loss stall vs 1000 p50/p99:", np.percentile(F1000[hit] - Fst[hit], [50, 99]).round(4))
    ok = hit & np.isfinite(v100) & np.isfinite(v1000) & (np.abs(v1000) > 0)
    rel = np.abs(v100[ok] - v1000[ok]) / np.abs(v1000[ok])
    print(f"  |{der}(100) - {der}(1000)| / {der}(1000) p50/p90/p99:", np.percentile(rel, [50, 90, 99]).round(4))
    ok = hit & np.isfinite(vst) & np.isfinite(v1000) & (np.abs(v1000) > 0)
    rel = np.abs(vst[ok] - v1000[ok]) / np.abs(v1000[ok])
    print(f"  |{der}(stall) - {der}(1000)| / {der}(1000) p50/p90/p99:", np.percentile(rel, [50, 90, 99]).round(4))
    print("  total iterations: 100-run", int(it100[fit].sum()), "| 1000-run", int(it1000[fit].sum()),
          "| stall-run", int(itst[fit].sum()))


if __name__ == "__main__":
    main()
