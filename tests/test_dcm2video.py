import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("dcm2video", os.path.join(ROOT, "scripts", "dcm2video.py"))
dcm2video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dcm2video)


def _slice(n, value, **tags):
    img = np.full((64, 64), value, dtype=np.int16)
    img[:8, :8] = -1000  # air
    return SimpleNamespace(pixel_array=img, InstanceNumber=n, **tags)


def _notebook_frames(volume, resize=512, wc=50, ww=180):
    """Reference implementation copied from the original dcm2video notebook."""
    def windowing(img, wc=50, ww=180):
        min_val = wc - ww / 2
        max_val = wc + ww / 2
        img = np.clip(img, min_val, max_val)
        img = (img - min_val) / (max_val - min_val)
        img = (img * 255).astype(np.uint8)
        return img

    frames = [windowing(f, wc, ww) for f in volume]
    frames = [cv2.resize(f, (resize, resize)) for f in frames]
    return [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)[:, :, ::-1] for f in frames]


def test_default_matches_notebook():
    rng = np.random.default_rng(0)
    volume = rng.integers(-200, 300, size=(6, 64, 64)).astype(np.int16)
    datasets = [SimpleNamespace(pixel_array=v, InstanceNumber=i + 1) for i, v in enumerate(volume)]
    cfg = dcm2video.load_config(os.path.join(ROOT, "configs", "dcm2video.yaml"))
    name, params = dcm2video.select_params(datasets[0], cfg)
    assert name == "default" and params == dcm2video.DEFAULT_PARAMS
    ours = dcm2video.series_to_frames(datasets, params)
    ref = _notebook_frames(volume)
    assert ours.shape == (6, 512, 512)
    for a, b in zip(ours, ref):
        assert np.array_equal(np.repeat(a[:, :, None], 3, axis=2), b)


def test_device_profiles():
    cfg = {
        "default": dict(dcm2video.DEFAULT_PARAMS),
        "devices": {
            "scanner_b": {"match": {"Manufacturer": "ge"}, "rescale": True, "window_center": 40},
            "scanner_c": {"window_width": 400},
        },
    }
    ds_ge = _slice(1, 0, Manufacturer="GE MEDICAL SYSTEMS")
    ds_other = _slice(1, 0, Manufacturer="SIEMENS")
    ds_none = _slice(1, 0)
    assert dcm2video.select_params(ds_ge, cfg)[0] == "scanner_b"
    assert dcm2video.select_params(ds_ge, cfg)[1]["window_center"] == 40
    assert dcm2video.select_params(ds_ge, cfg)[1]["window_width"] == 180  # inherited from default
    assert dcm2video.select_params(ds_other, cfg)[0] == "default"
    assert dcm2video.select_params(ds_none, cfg)[0] == "default"  # tag missing
    assert dcm2video.select_params(ds_none, cfg, device="scanner_c")[1]["window_width"] == 400
    with pytest.raises(KeyError):
        dcm2video.select_params(ds_none, cfg, device="unknown")


def test_rescale_to_hu():
    # stored value 1074 with intercept -1024 is 50 HU -> centre of the 50/180 window
    ds = SimpleNamespace(pixel_array=np.full((8, 8), 1074, np.int16), InstanceNumber=1, RescaleSlope=1,
                         RescaleIntercept=-1024)
    params = dict(dcm2video.DEFAULT_PARAMS, rescale=True, size=8)
    assert dcm2video.series_to_frames([ds], params)[0, 0, 0] == 127
    params["rescale"] = False  # stored value above the window -> white
    assert dcm2video.series_to_frames([ds], params)[0, 0, 0] == 255


def test_sorting_and_video_roundtrip(tmp_path):
    from urojepa.data.video_io import load_video

    datasets = [_slice(n, v) for n, v in [(3, 140), (1, -40), (2, 50)]]
    ordered = dcm2video.sort_slices(datasets)
    assert [d.InstanceNumber for d in ordered] == [1, 2, 3]
    frames = dcm2video.series_to_frames(ordered, dict(dcm2video.DEFAULT_PARAMS, size=64))
    out = tmp_path / "P1" / "axial.mp4"
    dcm2video.write_video(frames, str(out), fps=10)
    video, length = load_video(str(out))
    assert length == 3 and video.shape == (3, 64, 64, 3)
    centre = video[:, 32, 32, 0].astype(int)
    assert abs(centre[0] - 0) <= 10 and abs(centre[1] - 127) <= 10 and abs(centre[2] - 255) <= 10  # lossy mp4v


def test_find_series_dirs(tmp_path):
    for d in ("P1/axial", "P1/coronal", "P2/axial"):
        os.makedirs(tmp_path / d)
        (tmp_path / d / "0001.dcm").write_bytes(b"")
    os.makedirs(tmp_path / "P3" / "notes")
    (tmp_path / "P3" / "notes" / "readme.txt").write_text("x")
    found = [os.path.relpath(p, tmp_path) for p in dcm2video.find_series_dirs(str(tmp_path))]
    assert found == [os.path.join("P1", "axial"), os.path.join("P1", "coronal"), os.path.join("P2", "axial")]


def _write_dicom(path, instance, arr, series_uid):
    """Minimal CT slice; skips the test if this pydicom version cannot write it."""
    pydicom = pytest.importorskip("pydicom")
    try:
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        ct_storage = "1.2.840.10008.5.1.4.1.1.2"
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = ct_storage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = ct_storage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "CT"
        ds.InstanceNumber = instance
        ds.Rows, ds.Columns = arr.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PixelData = arr.astype(np.int16).tobytes()
        try:
            ds.save_as(str(path), enforce_file_format=True)  # pydicom >= 3
        except TypeError:
            ds.is_little_endian, ds.is_implicit_VR = True, False  # pydicom 2.x
            ds.save_as(str(path), write_like_original=False)
        pydicom.dcmread(str(path)).pixel_array
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cannot create a synthetic DICOM with pydicom {pydicom.__version__}: {e}")


def test_cli_single_and_batch(tmp_path):
    import subprocess
    import sys

    import pandas as pd

    from urojepa.data.video_io import count_frames

    root = tmp_path / "dicom"
    for pid, n in (("P1", 5), ("P2", 3)):
        d = root / pid / "axial"
        os.makedirs(d)
        uid = f"1.2.3.{pid[1]}"
        for k in range(n):  # file names in reverse slice order: sorting must use InstanceNumber
            _write_dicom(d / f"{n - k:04d}.dcm", k + 1, np.full((32, 32), k * 40 - 40), uid)

    script = os.path.join(ROOT, "scripts", "dcm2video.py")
    single = tmp_path / "single.mp4"
    subprocess.run([sys.executable, script, "--dicom-dir", str(root / "P1" / "axial"), "--out", str(single)],
                   check=True, cwd=ROOT, capture_output=True)
    assert count_frames(str(single)) == 5

    out_root = tmp_path / "videos"
    subprocess.run([sys.executable, script, "--dicom-root", str(root), "--out-root", str(out_root), "--workers", "2"],
                   check=True, cwd=ROOT, capture_output=True)
    log = pd.read_csv(out_root / "conversion_log.csv")
    assert (log.status == "ok").all() and sorted(log.num_frames) == [3, 5]
    assert count_frames(str(out_root / "P1" / "axial.mp4")) == 5
    assert count_frames(str(out_root / "P2" / "axial.mp4")) == 3
