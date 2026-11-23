import numpy as np
import pytest

from urojepa.data.clips import (
    clip_indices,
    clip_inventory,
    draw_hierarchical_clips,
    eval_clip_indices,
    num_clips,
    sample_train_clip,
)


def test_clip_definition_matches_paper():
    # I(s, d) = (s, s + d, ..., s + 15 d)
    assert clip_indices(3, 16, 1).tolist() == list(range(3, 19))
    assert clip_indices(0, 16, 2).tolist() == list(range(0, 31, 2))
    # N(L, d) = max(0, L - 15 d)
    assert num_clips(76, 16, 1) == 61 and num_clips(76, 16, 2) == 46
    assert num_clips(30, 16, 2) == 0 and num_clips(31, 16, 2) == 1  # < 31 slices: local only
    assert num_clips(15, 16, 1) == 0
    # the last valid start s = L - 1 - 15 d reaches the last slice exactly
    L = 53
    for d in (1, 2):
        last = clip_indices(num_clips(L, 16, d) - 1, 16, d)
        assert last[-1] == L - 1


def test_inventory():
    inv = clip_inventory([76, 53, 20, 10], 16, (1, 2))
    assert inv == {1: 61 + 38 + 5 + 0, 2: 46 + 23 + 0 + 0}


def test_hierarchical_sampling_constraints():
    lengths = np.array([76, 53, 20, 40, 16])
    patient_videos = [np.array([0, 1]), np.array([2]), np.array([3, 4])]
    rng = np.random.default_rng(0)
    n = 60000
    v, d, s = draw_hierarchical_clips(patient_videos, lengths, n, rng, 16, (1, 2))
    # every clip lies inside its video
    assert (s >= 0).all()
    assert (s + 15 * d <= lengths[v] - 1).all()
    # video 2 (20 slices) and video 4 (16 slices) only have local clips
    assert (d[v == 2] == 1).all() and (d[v == 4] == 1).all()
    # patients are sampled uniformly
    patient_of = np.array([0, 0, 1, 2, 2])
    freq = np.bincount(patient_of[v], minlength=3) / n
    assert np.allclose(freq, 1 / 3, atol=0.01)
    # within a patient, videos are sampled uniformly
    assert abs((v == 0).sum() / ((v == 0) | (v == 1)).sum() - 0.5) < 0.02
    # scale is uniform when both exist
    assert abs((d[v == 0] == 2).mean() - 0.5) < 0.02
    # starts are uniform over the valid range (video 0, d = 1 has 61 starts)
    starts = s[(v == 0) & (d == 1)]
    assert starts.min() == 0 and starts.max() == 60


def test_sampling_is_deterministic():
    lengths = np.array([76, 53])
    pv = [np.array([0]), np.array([1])]
    a = draw_hierarchical_clips(pv, lengths, 100, np.random.default_rng([0, 3]))
    b = draw_hierarchical_clips(pv, lengths, 100, np.random.default_rng([0, 3]))
    for x, y in zip(a, b):
        assert (x == y).all()


def test_train_and_eval_clips():
    rng = np.random.default_rng(0)
    for L in (16, 20, 31, 76):
        for _ in range(50):
            idx = sample_train_clip(L, rng)
            assert len(idx) == 16 and idx.min() >= 0 and idx.max() <= L - 1
            assert len(set(np.diff(idx))) == 1
    # short videos are padded by repeating the last slice (fine-tuning only)
    idx = sample_train_clip(10, rng)
    assert idx.tolist() == list(range(10)) + [9] * 6

    clips = eval_clip_indices(76, 9, 16, 1)
    assert len(clips) == 9
    assert clips[0][0] == 0 and clips[-1][-1] == 75
    starts = [c[0] for c in clips]
    assert starts == sorted(starts)
    # falls back to d = 1 when the video is too short for d = 2
    clips = eval_clip_indices(20, 9, 16, 2)
    assert all((np.diff(c) == 1).all() for c in clips)
    clips = eval_clip_indices(8, 3, 16, 1)
    assert clips[0].tolist() == list(range(8)) + [7] * 8


@pytest.mark.parametrize("bad", [[np.array([], dtype=int)]])
def test_empty_patient_raises(bad):
    with pytest.raises(ValueError):
        draw_hierarchical_clips(bad, np.array([20]), 5, np.random.default_rng(0))
