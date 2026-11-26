import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from urojepa.data.manifest import read_manifest, subsample_patients

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _labels(n_pat=200, pos_rate=0.25, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pat):
        y = int(rng.random() < pos_rate)
        for view in ("axial", "coronal")[: 1 + int(rng.random() < 0.9)]:
            rows.append({"patient_id": f"P{i:04d}", "video_path": f"P{i:04d}/{view}-0.mp4", "label": y, "view": view})
    return pd.DataFrame(rows)


def test_subsample_patients_is_stratified_and_patient_level():
    df = _labels()
    pl = df.groupby("patient_id").label.max()
    for frac in (0.1, 0.25, 0.5, 0.75):
        sub = subsample_patients(df, frac, seed=1)
        spl = sub.groupby("patient_id").label.max()
        for y in (0, 1):
            assert abs((spl == y).sum() - round(frac * (pl == y).sum())) <= 1
        # all videos of a kept patient are kept
        assert len(sub) == df.patient_id.isin(spl.index).sum()
    assert subsample_patients(df, 1.0, 0) is df
    a = subsample_patients(df, 0.25, 3)
    b = subsample_patients(df, 0.25, 3)
    assert a.equals(b)


def test_read_manifest_root(tmp_path):
    df = _labels(5)
    df.to_csv(tmp_path / "m.csv", index=False)
    m = read_manifest(str(tmp_path / "m.csv"), root="/data", require_label=True)
    assert m.video_path.str.startswith("/data/").all()


def _write_video(path, n_frames, size=64):
    cv2 = pytest.importorskip("cv2")
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size))
    for i in range(n_frames):
        frame = np.full((size, size, 3), i * 5 % 256, dtype=np.uint8)
        vw.write(frame)
    vw.release()


def test_video_io_and_manifest_builder(tmp_path):
    from urojepa.data.video_io import count_frames, load_video

    vdir = tmp_path / "videos"
    for pid, lens in {"P1": (40, 25), "P2": (12,)}.items():
        os.makedirs(vdir / pid)
        for k, n in enumerate(lens):
            _write_video(vdir / pid / f"{'axial' if k == 0 else 'coronal'}-0.mp4", n)

    assert count_frames(str(vdir / "P1" / "axial-0.mp4")) == 40
    frames, _ = load_video(str(vdir / "P1" / "axial-0.mp4"), [0, 2, 39])
    assert frames.shape == (3, 64, 64, 3) and frames.dtype == np.uint8
    assert abs(int(frames[1].mean()) - 10) <= 3  # frame index preserved
    allf, length = load_video(str(vdir / "P1" / "coronal-0.mp4"))
    assert length == 25 and len(allf) == 25
    with pytest.raises(IndexError):
        load_video(str(vdir / "P1" / "coronal-0.mp4"), [30])

    out = tmp_path / "manifest.csv"
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "build_manifest.py"), "--video-dir", str(vdir),
         "--output", str(out), "--min-frames", "16", "--workers", "2"],
        check=True, cwd=ROOT, capture_output=True, text=True,
    )
    m = pd.read_csv(out)
    assert sorted(m.num_frames.tolist()) == [25, 40]  # 12-slice video dropped
    assert set(m.view) == {"axial", "coronal"}
    assert "local (d=1) 35, broader (d=2) 10" in r.stdout
