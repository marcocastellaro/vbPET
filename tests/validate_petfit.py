"""Regional 2TCM on the petfit (kinfitr) TACs of ds004869 sub-01 baseline vs petfit.

Same TACs, same input (bloodstream AIF), Vb fixed at 0.05, blood shift and
weights as petfit -> differences come from the fitting code only.
Writes a TSV next to the outputs and prints a summary.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "testdata" / "ds004869"
PF = DS / "derivatives/petfit/Primary_Analysis/sub-01/ses-baseline/pet"
OUT = ROOT / "testdata" / "out" / "validation"


def main():
    from vbpet.input import make_plasma_input
    from vbpet.io.bids import find_pet_runs, load_blood
    from vbpet.models.base import Frames
    from vbpet.models.plasma import get_model
    from vbpet.regional import fit_regions

    tac = pd.read_csv(PF / "sub-01_ses-baseline_desc-combinedregions_tacs.tsv", sep="\t")
    wts = pd.read_csv(PF / "sub-01_ses-baseline_desc-weights_weights.tsv", sep="\t")
    ref = pd.read_csv(PF / "sub-01_ses-baseline_model_2TCM_desc-model1_kinpar.tsv", sep="\t")
    shift = float(pd.read_csv(PF / "sub-01_ses-baseline_model-1TCM_desc-delayfit_kinpar.tsv",
                              sep="\t")["blood_timeshift"].iloc[0])
    wide = tac.pivot(index="frame_start", columns="region", values="TAC").sort_index()
    fr = tac.drop_duplicates("frame_start").sort_values("frame_start")
    frames = Frames.from_start_duration(fr["frame_start"].to_numpy() / 60,
                                        fr["frame_dur"].to_numpy() / 60)
    w = np.sqrt(wts.sort_values("frame_start")["weights"].to_numpy())  # kinfitr: weights on res^2

    run = find_pet_runs(DS, ["01"], ["baseline"])[0]
    blood = load_blood(run, DS / "derivatives/bloodstream")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for conv, sampling in [("linear", "mid"), ("linear", "mean")]:
        inp = make_plasma_input(blood, float(frames.end[-1]), step=1 / 60, scale=1e-3, shift=shift)
        model = get_model("2TCM", conv=conv, sampling=sampling).fix({"Vb": 0.05})
        kin, _ = fit_regions(model, inp, frames, wide.reset_index(drop=True), w)
        m = kin.merge(ref, left_on="name", right_on="region", suffixes=("", "_petfit"))
        for p in ["K1", "k2", "k3", "k4", "VT"]:
            rel = 100 * (m[p] - m[f"{p}_petfit"]) / m[f"{p}_petfit"]
            rows.append(dict(conv=conv, sampling=sampling, param=p,
                             median_abs_rel_diff_pct=float(np.median(np.abs(rel))),
                             max_abs_rel_diff_pct=float(np.max(np.abs(rel)))))
        m.to_csv(OUT / f"petfit_2TCM_{conv}_{sampling}.tsv", sep="\t", index=False)
        print(f"--- {conv}/{sampling}")
        print(m[["name", "K1", "K1_petfit", "VT", "VT_petfit", "wrss"]].to_string(index=False))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "petfit_2TCM_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
