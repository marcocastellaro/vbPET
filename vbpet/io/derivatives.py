"""BIDS derivative writer (PET kinetic modelling conventions: kinpar/mimap/tacs)."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .. import __version__


def write_dataset_description(out_dir, bids_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dd = {
        "Name": "vbPET parametric maps",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "vbpet", "Version": __version__,
                         "CodeURL": "https://github.com/marcocastellaro/vbPET"}],
        "SourceDatasets": [{"URL": str(Path(bids_dir).resolve())}],
    }
    with open(out_dir / "dataset_description.json", "w") as f:
        json.dump(dd, f, indent=2)


class RunWriter:
    """Writes files for one PET run under <out>/sub-X[/ses-Y]/pet/."""

    def __init__(self, out_dir, run):
        e = run.entities
        d = Path(out_dir) / f"sub-{e['sub']}"
        if "ses" in e:
            d = d / f"ses-{e['ses']}"
        self.dir = d / "pet"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix = run.prefix
        self.ref_img = None

    def path(self, *parts, suffix, ext):
        name = "_".join([self.prefix, *[p for p in parts if p], suffix]) + ext
        return self.dir / name

    def tsv(self, df, *parts, suffix, meta=None):
        p = self.path(*parts, suffix=suffix, ext=".tsv")
        df.to_csv(p, sep="\t", index=False, na_rep="n/a")
        if meta is not None:
            self.json(meta, p)
        return p

    def json(self, meta, data_path):
        p = Path(str(data_path).split(".")[0] + ".json")
        with open(p, "w") as f:
            json.dump(meta, f, indent=2, default=_jsonable)
        return p

    def nifti(self, vol, ref_img, *parts, suffix="mimap", meta=None):
        p = self.path(*parts, suffix=suffix, ext=".nii.gz")
        hdr = ref_img.header.copy()
        hdr.set_data_dtype(np.float32)
        img = nib.Nifti1Image(np.asarray(vol, np.float32), ref_img.affine, hdr)
        if vol.ndim == 3:
            img.header.set_data_shape(vol.shape)
        nib.save(img, p)
        if meta is not None:
            self.json(meta, p)
        return p


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
