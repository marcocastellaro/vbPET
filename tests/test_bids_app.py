"""End-to-end BIDS App run on a tiny synthetic BIDS dataset (CPU, seconds)."""

import json

import jax
import jax.numpy as jnp
import nibabel as nib
import numpy as np
import pandas as pd

from synthetic import FRAME_DUR_S, FRAME_START_S, feng_input
from vbpet.cli import main
from vbpet.models import get_model
from vbpet.models.base import Frames, PlasmaInput

TRUE = {1: (0.10, 0.15, 0.04, 0.03, 0.04), 2: (0.06, 0.12, 0.02, 0.03, 0.03)}  # label -> 2TCM


def make_dataset(root, rng):
    sub = root / "sub-01" / "pet"
    sub.mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "synth", "BIDSVersion": "1.10.0"}))
    frames = Frames.from_start_duration(FRAME_START_S / 60, FRAME_DUR_S / 60)
    tb = np.arange(0, 121, 1 / 60)
    cp_bq = feng_input(tb) * 1e3  # Bq/mL
    pd.DataFrame({"time": tb * 60, "plasma_radioactivity": cp_bq / 1e3,
                  "whole_blood_radioactivity": 0.9 * cp_bq / 1e3,
                  "metabolite_parent_fraction": np.ones_like(tb)}).to_csv(
        sub / "sub-01_recording-manual_blood.tsv", sep="\t", index=False)
    (sub / "sub-01_recording-manual_blood.json").write_text(json.dumps({
        "plasma_radioactivity": {"Units": "kBq/mL"},
        "whole_blood_radioactivity": {"Units": "kBq/mL"}, "time": {"Units": "s"}}))
    model = get_model("2TCM", conv="linear", sampling="mean")
    inp = PlasmaInput(tb, feng_input(tb), 0.9 * feng_input(tb))
    shape = (6, 6, 4)
    dseg = np.zeros(shape, np.int16)
    dseg[:, :, :2] = 1
    dseg[:, :, 2:] = 2
    dseg[0, 0, 0] = 0
    data = np.zeros(shape + (len(FRAME_START_S),), np.float32)
    for lab, th in TRUE.items():
        tac = np.asarray(model.predict(jnp.asarray(th), inp, frames))  # kBq/mL
        sel = dseg == lab
        data[sel] = tac * (1 + 0.02 * rng.normal(size=(sel.sum(), tac.size)))
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(data * 1e3, aff), sub / "sub-01_pet.nii.gz")  # Bq/mL
    (sub / "sub-01_pet.json").write_text(json.dumps({
        "Units": "Bq/mL", "FrameTimesStart": FRAME_START_S.tolist(),
        "FrameDuration": FRAME_DUR_S.tolist(), "TracerName": "synth"}))
    der = root / "derivatives" / "seg" / "sub-01" / "pet"
    der.mkdir(parents=True)
    nib.save(nib.Nifti1Image(dseg, aff), der / "sub-01_space-pet_seg-test_dseg.nii.gz")
    pd.DataFrame({"index": [1, 2], "name": ["high", "low"]}).to_csv(
        der / "sub-01_space-pet_seg-test_dseg.tsv", sep="\t", index=False)


def test_bids_app_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    bids = tmp_path / "bids"
    make_dataset(bids, rng)
    out = tmp_path / "out"
    rc = main([str(bids), str(out), "participant", "--model", "2TCM", "--device", "cpu",
               "--derivatives", str(bids / "derivatives" / "seg"), "--dseg-desc", "test",
               "--chunk-size", "64"])
    assert rc == 0
    d = out / "sub-01" / "pet"
    kin = pd.read_csv(d / "sub-01_model-2TCM_desc-regional_kinpar.tsv", sep="\t").set_index("name")
    vt = lambda t: t[0] / t[1] * (1 + t[2] / t[3])
    assert abs(kin.loc["high", "VT"] / vt(TRUE[1]) - 1) < 0.05
    vtmap = nib.load(d / "sub-01_model-2TCM_meas-VT_mimap.nii.gz").get_fdata()
    seg = nib.load(bids / "derivatives/seg/sub-01/pet/sub-01_space-pet_seg-test_dseg.nii.gz").get_fdata()
    for lab, th in TRUE.items():
        med = np.nanmedian(vtmap[seg == lab])
        assert abs(med / vt(th) - 1) < 0.1, (lab, med, vt(th))
    assert np.isnan(vtmap[0, 0, 0])
    meta = json.loads((d / "sub-01_model-2TCM_meas-VT_mimap.json").read_text())
    assert meta["FrameSampling"] == "mean" and meta["Convolution"] == "linear"
    assert (out / "dataset_description.json").exists()


def test_bids_app_srtm_reference_region(tmp_path):
    rng = np.random.default_rng(1)
    bids = tmp_path / "bids"
    make_dataset(bids, rng)
    out = tmp_path / "out"
    rc = main([str(bids), str(out), "participant", "--model", "SRTM", "--ref-label", "low",
               "--device", "cpu", "--derivatives", str(bids / "derivatives" / "seg"),
               "--dseg-desc", "test", "--chunk-size", "64"])
    assert rc == 0
    d = out / "sub-01" / "pet"
    kin = pd.read_csv(d / "sub-01_model-SRTM_desc-regional_kinpar.tsv", sep="\t").set_index("name")
    # reference fitted against itself: BPnd = 0. R1 is not identifiable here: R1 -> 0
    # with a very large k2 reproduces C_R as well (k2 C_R ⊗ exp(-k2 t) ≈ C_R)
    assert abs(kin.loc["low", "BPnd"]) < 0.02
    # high/low VT ratio = 1 + BPnd (DVR) for these 2TCM regions, approximately
    vt = lambda t: t[0] / t[1] * (1 + t[2] / t[3])
    dvr = vt(TRUE[1]) / vt(TRUE[2])
    assert abs((1 + kin.loc["high", "BPnd"]) / dvr - 1) < 0.15
    bp = nib.load(d / "sub-01_model-SRTM_meas-BPnd_mimap.nii.gz").get_fdata()
    assert np.isfinite(bp).sum() > 100


def test_bids_app_model_selection(tmp_path):
    rng = np.random.default_rng(2)
    bids = tmp_path / "bids"
    make_dataset(bids, rng)
    out = tmp_path / "out"
    rc = main([str(bids), str(out), "participant", "--model", "1TCM,2TCM", "--device", "cpu",
               "--derivatives", str(bids / "derivatives" / "seg"), "--dseg-desc", "test",
               "--chunk-size", "64"])
    assert rc == 0
    d = out / "sub-01" / "pet"
    best = nib.load(d / "sub-01_model-selection_meas-BestModel_mimap.nii.gz").get_fdata()
    p2 = nib.load(d / "sub-01_model-selection_meas-P2TCM_mimap.nii.gz").get_fdata()
    seg = nib.load(bids / "derivatives/seg/sub-01/pet/sub-01_space-pet_seg-test_dseg.nii.gz").get_fdata()
    # data are 2TCM with a clear second compartment in label 1 ("high")
    assert np.mean(best[seg == 1] == 2) > 0.8
    assert np.nanmedian(p2[seg == 1]) > 0.9
    vt = nib.load(d / "sub-01_model-selection_meas-VT_desc-bma_mimap.nii.gz").get_fdata()
    assert np.isfinite(vt[seg > 0]).mean() > 0.95
    tab = pd.read_csv(d / "sub-01_model-selection_desc-regional_kinpar.tsv", sep="\t").set_index("name")
    assert tab.loc["high", "best_sumF"] == "2TCM"
