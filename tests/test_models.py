"""Smoke tests of the model code on CPU (tiny ViT). Skipped when torch is not installed."""

import pytest

torch = pytest.importorskip("torch")

from urojepa.masks.distance import compute_mask_distance  # noqa: E402
from urojepa.masks.multiblock3d import MaskCollator  # noqa: E402
from urojepa.masks.utils import apply_masks  # noqa: E402
from urojepa.models.classifier import VideoClassifier, param_groups  # noqa: E402
from urojepa.models.predictor import vit_predictor  # noqa: E402
from urojepa.models.vision_transformer import VisionTransformer  # noqa: E402
from urojepa.models.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper  # noqa: E402
from urojepa.utils.checkpoint import load_encoder_weights  # noqa: E402

CROP, T, PATCH, TUB = 64, 16, 16, 2
MASKS = [
    dict(aspect_ratio=[0.75, 1.5], num_blocks=8, spatial_scale=[0.15, 0.15], temporal_scale=[1.0, 1.0]),
    dict(aspect_ratio=[0.75, 1.5], num_blocks=2, spatial_scale=[0.7, 0.7], temporal_scale=[1.0, 1.0]),
]


def tiny_encoder(**kw):
    return VisionTransformer(
        img_size=CROP, patch_size=PATCH, num_frames=T, tubelet_size=TUB, embed_dim=48, depth=4, num_heads=4,
        use_sdpa=True, **kw,
    )


def test_encoder_shapes():
    enc = tiny_encoder()
    x = torch.randn(2, 3, T, CROP, CROP)
    n_tokens = (T // TUB) * (CROP // PATCH) ** 2
    assert enc(x).shape == (2, n_tokens, 48)
    assert enc(x, training=True).shape == (2, n_tokens, 4 * 48)
    idx = torch.stack([torch.arange(0, n_tokens, 2)] * 2)
    assert enc(x, masks=[idx], training=True).shape == (2, n_tokens // 2, 4 * 48)


def test_vjepa21_step_and_mask_distance():
    torch.manual_seed(0)
    collator = MaskCollator(MASKS, [T], crop_size=CROP, patch_size=PATCH, tubelet_size=TUB)
    batch = collator([torch.randn(3, T, CROP, CROP) for _ in range(2)])
    assert len(batch) == 1
    clips, masks_enc, masks_pred = batch[0]
    clips, masks_enc, masks_pred = [clips], [masks_enc], [masks_pred]
    for me, mp in zip(masks_enc[0], masks_pred[0]):
        assert me.shape[0] == 2 and mp.shape[0] == 2
        for b in range(2):  # context and target tokens are disjoint
            assert not set(me[b].tolist()) & set(mp[b].tolist())

    enc = MultiSeqWrapper(tiny_encoder())
    pred = PredictorMultiSeqWrapper(
        vit_predictor(img_size=CROP, patch_size=PATCH, num_frames=T, tubelet_size=TUB, embed_dim=48,
                      predictor_embed_dim=32, depth=2, num_heads=4, num_mask_tokens=2, num_levels=4)
    )
    with torch.no_grad():
        h = enc(clips, training_mode=True)
    z = enc(clips, masks_enc, training_mode=True)
    z_pred, z_ctx = pred(z, masks_enc, masks_pred)
    tgt_pred = apply_masks(h[0], masks_pred[0], concat=False)
    tgt_ctx = apply_masks(h[0], masks_enc[0], concat=False)
    for zp, tp, zc, tc in zip(z_pred[0], tgt_pred, z_ctx[0], tgt_ctx):
        assert zp.shape == tp.shape and zc.shape == tc.shape
    d = compute_mask_distance(masks_pred, masks_enc, CROP // PATCH)
    assert d[0][0].shape == masks_enc[0][0].shape and (d[0][0] >= 1).all()
    loss = sum((zp - tp).abs().mean() for zp, tp in zip(z_pred[0], tgt_pred))
    loss.backward()


def test_mask_distance_matches_reference_loop():
    torch.manual_seed(0)
    g = 4
    me = torch.stack([torch.randperm(2 * g * g)[:10] for _ in range(3)])
    mp = torch.stack([torch.tensor([i for i in range(2 * g * g) if i not in set(r.tolist())][:12]) for r in me])
    fast = compute_mask_distance([[mp]], [[me]], g)[0][0]

    def pos(ids):
        d = ids // (g * g)
        h = (ids - d * g * g) // g
        w = ids - d * g * g - h * g
        return torch.stack([d, h, w], -1).float()

    for b in range(3):
        ref = torch.cdist(pos(me[b]), pos(mp[b])).min(-1).values.sqrt()
        assert torch.allclose(fast[b], ref)


def test_classifier_and_checkpoint_roundtrip(tmp_path):
    enc = tiny_encoder()
    path = tmp_path / "ckpt.pth.tar"
    torch.save({"target_encoder": {f"backbone.{k}": v for k, v in enc.state_dict().items()}, "epoch": 1}, path)
    enc2 = load_encoder_weights(tiny_encoder(), str(path))
    for k, v in enc.state_dict().items():
        assert torch.equal(v, enc2.state_dict()[k])

    for head, depth in (("attentive", 4), ("attentive", 1), ("mean_linear", 1)):
        model = VideoClassifier(enc2, num_classes=2, head=head, head_depth=depth, head_num_heads=4)
        x = torch.randn(2, 3, T, CROP, CROP)
        out = model(x)
        assert out.shape == (2, 2)
        groups = param_groups(model, weight_decay=0.01)
        n_params = sum(p.numel() for g in groups for p in g["params"])
        assert n_params == sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt = torch.optim.AdamW(groups, lr=1e-3)
        loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1]))
        loss.backward()
        opt.step()
        # every trainable parameter receives a gradient (required by DistributedDataParallel)
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
        opt.zero_grad(set_to_none=True)
