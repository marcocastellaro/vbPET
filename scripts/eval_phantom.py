"""Compare vbpet maps with the phantom ground truth, per organ.

    python scripts/eval_phantom.py <phantom_bids> <out_dir> [<out_dir> ...]
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def main():
    ph = Path(sys.argv[1])
    seg = ph / "derivatives/seg/sub-01/pet/sub-01_space-pet_seg-organs_dseg"
    lab = np.asarray(nib.load(str(seg) + ".nii.gz").dataobj).astype(int)
    names = pd.read_csv(str(seg) + ".tsv", sep="\t").set_index("index")["name"]
    rows = []
    for out in map(Path, sys.argv[2:]):
        d = out / "sub-01" / "pet"
        for p in ["K1", "k3", "Ki", "delay"]:
            est = d / f"sub-01_model-irr2TCM_meas-{p}_mimap.nii.gz"
            tru = ph / f"derivatives/truth/sub-01/pet/sub-01_meas-{p}_truth.nii.gz"
            if not est.exists():
                continue
            e = nib.load(est).get_fdata()
            t = nib.load(tru).get_fdata()
            if p == "delay":  # the IDIF comes from the aorta: delays are relative to it
                aorta = [i for i, n in names.items() if n == "aorta"]
                t = t - np.nanmedian(t[np.isin(lab, aorta)])
            for i, n in names.items():
                m = (lab == i) & np.isfinite(e) & np.isfinite(t)
                if m.sum() < 5 or n in ("aorta",):
                    continue
                err = e[m] - t[m]
                rows.append(dict(run=out.name, param=p, organ=n, n=int(m.sum()),
                                 truth=float(np.median(t[m])),
                                 bias_pct=float(100 * np.median(err) / np.median(t[m])) if np.median(t[m]) else np.nan,
                                 rmse=float(np.sqrt(np.mean(err ** 2))),
                                 corr=float(np.corrcoef(e[m], t[m])[0, 1]) if t[m].std() > 0 else np.nan))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no finite estimates to compare")
        return
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    if df.run.nunique() > 1:
        piv = df.pivot_table(index=["param", "organ"], columns="run", values="rmse")
        print("\nRMSE by run:\n", piv.round(5).to_string())
    df.to_csv(Path(sys.argv[2]).parent / "phantom_eval.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
