import numpy as np
import pytest
import torch

from defectvision import data, metrics
from defectvision.backbone import DEFAULT_PATH, check_parity, resnet18
from defectvision.export import export, parity
from defectvision.patchcore import PatchCore, greedy_coreset
from defectvision.segmentation import Inspector, UNet, dice_loss

from .conftest import texture, with_defect

SIZE = 64


def batch(images):
    return torch.from_numpy(np.stack([data.preprocess(i, SIZE) for i in images]))


def test_weight_conversion_matches_onnx_runtime(pretrained):
    if not pretrained:
        pytest.skip("weights missing: run `dv fetch-weights`")
    assert check_parity(DEFAULT_PATH) < 1e-4


def test_greedy_coreset_selects_distinct_spread_points():
    torch.manual_seed(0)
    pts = torch.cat([torch.randn(200, 8) * 0.01 + c for c in (0.0, 5.0, 10.0)])  # 3 clusters
    idx = greedy_coreset(pts, 3, projection_dim=0)
    assert len(set(idx.tolist())) == 3
    assert sorted(int(i) // 200 for i in idx) == [0, 1, 2]  # one point per cluster


def test_patchcore_learns_from_good_only_and_flags_scratches(pretrained):
    rng = np.random.default_rng(0)
    model = PatchCore(resnet18(pretrained=pretrained), image_size=SIZE, blur_sigma=2)
    model.fit([batch([texture(rng, SIZE) for _ in range(12)])], coreset_ratio=0.2, log=lambda *_: None)
    goods = [texture(rng, SIZE) for _ in range(10)]
    bads, masks = zip(*[with_defect(texture(rng, SIZE), rng) for _ in range(10)], strict=True)
    scores, maps = model.predict(batch(goods + list(bads)))
    assert maps.shape == (20, SIZE, SIZE)
    assert metrics.auroc(np.repeat([0, 1], 10), scores) > 0.9
    # the anomaly map concentrates on the scratch
    assert metrics.pixel_auroc(np.stack(masks), maps[10:]) > 0.8


def test_unet_shapes_and_dice_loss():
    net = UNet(pretrained=False)
    logits = net(torch.zeros(2, 3, SIZE, SIZE))
    assert logits.shape == (2, len(data.CLASSES), SIZE, SIZE)
    target = torch.zeros(2, SIZE, SIZE, dtype=torch.long)
    target[0, 10:20, 10:20] = 3
    perfect = torch.nn.functional.one_hot(target, len(data.CLASSES)).permute(0, 3, 1, 2).float() * 50
    assert dice_loss(perfect, target) < 0.01
    score, amap, probs = Inspector(net)(torch.zeros(1, 3, SIZE, SIZE))
    assert (
        score.shape == (1,)
        and amap.shape == (1, SIZE, SIZE)
        and torch.allclose(probs.sum(1), torch.ones(1, SIZE, SIZE))
    )


def test_onnx_export_is_self_contained_and_faithful(tmp_path):
    rng = np.random.default_rng(0)
    pc = PatchCore(resnet18(pretrained=False), image_size=SIZE, blur_sigma=2)
    pc.fit([batch([texture(rng, SIZE) for _ in range(4)])], coreset_ratio=0.1, log=lambda *_: None)
    path = export(pc, tmp_path / "patchcore.onnx", "patchcore", threshold=1.5, image_size=SIZE)
    assert parity(pc, path, SIZE) < 1e-3

    unet = Inspector(UNet(pretrained=False))
    path = export(unet, tmp_path / "unet.onnx", "unet", threshold=0.5, image_size=SIZE)
    assert parity(unet, path, SIZE) < 1e-4

    from defectvision.inference import OnnxInspector

    insp = OnnxInspector(path)
    assert (insp.kind, insp.threshold, insp.image_size) == ("unet", 0.5, SIZE)
    res = insp.inspect(texture(rng, 100))
    assert res.anomaly_map.shape == (SIZE, SIZE) and res.to_dict()["decision"] in {"REJECT", "PASS"}
