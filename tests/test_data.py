from defectvision import data


def test_index_groups_six_exposures_per_piece(fake_dataset):
    samples = data.index(fake_dataset)
    assert len(samples) == 24
    groups = {}
    for s in samples:
        groups.setdefault(s.group, set()).add(s.exposure)
    assert len(groups) == 4
    assert all(exps == set(range(1, 7)) for exps in groups.values())
    assert {s.label for s in samples} == {"good", "crack"}


def test_group_split_has_no_leakage_and_is_deterministic(fake_dataset):
    samples = data.index(fake_dataset)
    a = data.split(samples, val=0.0, test=0.5, seed=1)
    b = data.split(samples, val=0.0, test=0.5, seed=1)
    assert [s.image for s in a["test"]] == [s.image for s in b["test"]]
    train_groups = {s.group for s in a["train"] + a["val"]}
    assert not train_groups & {s.group for s in a["test"]}


def test_random_split_leaks_pieces(fake_dataset):
    samples = data.index(fake_dataset)
    parts = data.split(samples, val=0.2, test=0.2, seed=0, by_group=False)
    assert {s.group for s in parts["train"]} & {s.group for s in parts["test"]}


def test_preprocess_shape_and_normalisation():
    import numpy as np

    x = data.preprocess(np.full((50, 80), 255, np.uint8), 32)
    assert x.shape == (3, 32, 32) and x.dtype == np.float32
    assert np.allclose(x[:, 0, 0], (1 - data.IMAGENET_MEAN) / data.IMAGENET_STD, atol=1e-5)
