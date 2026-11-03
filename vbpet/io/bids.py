"""Minimal BIDS-PET reader: runs, frame timing, blood/input function, dseg.

File discovery is glob-based (no index of the whole dataset), which keeps start
up fast on network filesystems and with large whole-body datasets.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ENTITY_ORDER = ("sub", "ses", "task", "trc", "rec", "run")

# conversion factors to Bq/mL
_UNIT = {
    "bq/ml": 1.0, "bq/cc": 1.0, "bq": 1.0, "kbq/ml": 1e3, "kbq/cc": 1e3, "kbq": 1e3,
    "mbq/ml": 1e6, "mbq": 1e6, "nci/ml": 37.0, "nci/cc": 37.0, "uci/ml": 37e3,
    "cps/ml": 1.0,
}


def unit_factor(unit):
    """Factor converting ``unit`` to Bq/mL (unknown units -> 1 with a warning)."""
    if isinstance(unit, (list, tuple)):
        unit = unit[0] if unit else ""
    u = str(unit).strip().lower().replace(" ", "").replace("μ", "u")
    if u in _UNIT:
        return _UNIT[u]
    import warnings
    warnings.warn(f"unknown radioactivity unit {unit!r}, assuming Bq/mL")
    return 1.0


def parse_entities(path):
    name = Path(path).name
    ents = dict(re.findall(r"(?:^|_)([a-zA-Z0-9]+)-([a-zA-Z0-9]+)", name))
    return {k: v for k, v in ents.items()}


def entity_prefix(ents, keys=ENTITY_ORDER):
    return "_".join(f"{k}-{ents[k]}" for k in keys if k in ents)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def sidecar(path):
    p = Path(path)
    stem = p.name.split(".")[0]
    js = p.with_name(stem + ".json")
    return read_json(js) if js.exists() else {}


@dataclass
class PetRun:
    path: Path
    entities: dict
    meta: dict = field(default_factory=dict)

    @property
    def prefix(self):
        return entity_prefix(self.entities)

    @property
    def frame_start_min(self):
        return np.asarray(self.meta["FrameTimesStart"], float) / 60.0

    @property
    def frame_duration_min(self):
        return np.asarray(self.meta["FrameDuration"], float) / 60.0

    @property
    def unit_factor(self):
        return unit_factor(self.meta.get("Units", "Bq/mL"))

    def load(self):
        img = nib.load(str(self.path))
        return img


def find_pet_runs(bids_dir, participant_label=None, session_label=None, filters=None):
    bids_dir = Path(bids_dir)
    subs = participant_label or [p.name[4:] for p in sorted(bids_dir.glob("sub-*")) if p.is_dir()]
    runs = []
    for sub in subs:
        sub = sub.removeprefix("sub-")
        for f in sorted((bids_dir / f"sub-{sub}").glob("**/pet/*_pet.nii*")):
            ents = parse_entities(f)
            if session_label and ents.get("ses") not in session_label:
                continue
            if filters and any(ents.get(k) != v for k, v in filters.items()):
                continue
            runs.append(PetRun(f, ents, sidecar(f)))
    return runs


# --------------------------------------------------------------------- blood
@dataclass
class BloodData:
    """Blood curves (time in minutes, activity in Bq/mL)."""

    t: np.ndarray
    plasma_parent: np.ndarray  # metabolite-corrected plasma (AIF)
    whole_blood: np.ndarray
    source: str


def _same_run_files(run, root, suffix_glob):
    """Files under ``root`` matching the run's entities (sub/ses/trc/rec/run)."""
    ents = run.entities
    sub = ents["sub"]
    cands = sorted(Path(root).glob(f"**/sub-{sub}/**/*{suffix_glob}"))
    out = []
    for c in cands:
        e = parse_entities(c)
        if all(e.get(k) == ents.get(k) for k in ("sub", "ses", "trc", "rec", "run") if k in ents):
            out.append(c)
    return out


def load_blood_derivative(run, deriv_dir):
    """bloodstream-style ``*_inputfunction.tsv`` (time s, AIF, whole_blood_radioactivity)."""
    files = _same_run_files(run, deriv_dir, "_inputfunction.tsv")
    if not files:
        return None
    f = files[0]
    df = pd.read_csv(f, sep="\t")
    meta = sidecar(f)
    fac_aif = unit_factor(meta.get("AIF", {}).get("Units", "Bq/mL"))
    fac_wb = unit_factor(meta.get("whole_blood_radioactivity", {}).get("Units", "Bq/mL"))
    wb = df["whole_blood_radioactivity"].to_numpy(float) * fac_wb if \
        "whole_blood_radioactivity" in df else np.zeros(len(df))
    return BloodData(df["time"].to_numpy(float) / 60.0, df["AIF"].to_numpy(float) * fac_aif,
                     wb, str(f))


def load_blood_raw(run):
    """Combine ``*_recording-<label>_blood.tsv`` files next to the PET run."""
    pet_dir = run.path.parent
    stem = run.path.name.split("_pet.")[0]
    files = sorted(pet_dir.glob(stem + "_recording-*_blood.tsv"))
    if not files:
        return None
    recs = {}
    for f in files:
        rec = parse_entities(f).get("recording", f.name)
        df = pd.read_csv(f, sep="\t").replace("n/a", np.nan)
        meta = sidecar(f)
        cols = {}
        for c in df.columns:
            if c == "time":
                continue
            u = meta.get(c, {}).get("Units", None) if isinstance(meta.get(c), dict) else None
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            if c.endswith("radioactivity") and u is not None:
                v = v * unit_factor(u)
            cols[c] = v
        cols["time"] = pd.to_numeric(df["time"], errors="coerce").to_numpy(float) / 60.0
        recs[rec] = cols

    def series(name):
        best = None
        for rec, cols in recs.items():
            if name in cols and np.isfinite(cols[name]).sum() > 1:
                ok = np.isfinite(cols[name])
                cand = (cols["time"][ok], cols[name][ok])
                if best is None or len(cand[0]) > len(best[0]):  # densest sampling
                    best = cand
        return best

    pl = series("plasma_radioactivity")
    wb = series("whole_blood_radioactivity")
    pf = series("metabolite_parent_fraction")
    if pl is None and wb is None:
        raise ValueError(f"no plasma/whole blood radioactivity in {files}")
    if pl is None:
        pl = wb  # assume plasma = whole blood
    if wb is None:
        wb = pl
    t = np.unique(np.concatenate([pl[0], wb[0]]))
    plasma = np.interp(t, *pl)
    whole = np.interp(t, *wb)
    parent = np.interp(t, *pf) if pf is not None else np.ones_like(t)
    return BloodData(t, plasma * parent, whole, ",".join(str(f) for f in files))


def load_blood(run, blood_derivative=None):
    if blood_derivative:
        b = load_blood_derivative(run, blood_derivative)
        if b is not None:
            return b
    b = load_blood_raw(run)
    if b is None:
        raise FileNotFoundError(f"no blood data found for {run.path}")
    return b


# ---------------------------------------------------------------------- dseg
def find_dseg(run, deriv_dirs, desc=None, space=None):
    """dseg derivative for this run, on the same grid as the PET image."""
    pet = nib.load(str(run.path))
    cands = []
    for d in deriv_dirs:
        for f in _same_run_files(run, d, "_dseg.nii*") + sorted(
                Path(d).glob(f"**/sub-{run.entities['sub']}/**/anat/*_dseg.nii*")):
            e = parse_entities(f)
            if desc and e.get("desc") != desc and e.get("seg") != desc:
                continue
            if space and e.get("space") != space:
                continue
            cands.append(f)
    for f in dict.fromkeys(cands):
        img = nib.load(str(f))
        if img.shape[:3] == pet.shape[:3] and np.allclose(img.affine, pet.affine, atol=1e-3):
            return f
    raise FileNotFoundError(
        f"no dseg on the PET grid for {run.path.name} in {deriv_dirs} (candidates: "
        f"{[c.name for c in cands]}); resample the segmentation to PET space first")


def read_labels(dseg_path):
    """BIDS dseg TSV (index, name). Falls back to 'label-<n>'."""
    p = Path(dseg_path)
    tsv = p.with_name(p.name.split(".")[0] + ".tsv")
    if tsv.exists():
        df = pd.read_csv(tsv, sep="\t")
        idx_col = "index" if "index" in df else df.columns[0]
        name_col = "name" if "name" in df else df.columns[1]
        return dict(zip(df[idx_col].astype(int), df[name_col].astype(str)))
    return {}
