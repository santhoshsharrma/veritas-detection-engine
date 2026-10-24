from src.data.manifest import split_rows


def _rows():
    rows = []
    for split, n in (("train", 100), ("test", 20)):
        for label in (0, 1):
            rows += [{"image_path": f"{split}/{label}/{i}.jpg", "label": label,
                      "source_generator": "x", "split": split} for i in range(n)]
    return rows


def test_split_is_stratified_and_disjoint():
    train, val, test = split_rows(_rows(), val_fraction=0.1, seed=1)
    assert len(val) == 20 and len(train) == 180 and len(test) == 40
    assert sum(r["label"] for r in val) == 10
    assert not {r["image_path"] for r in train} & {r["image_path"] for r in val}
    assert all(r["split"] == "test" for r in test)


def test_split_is_deterministic():
    a = split_rows(_rows(), seed=7)[1]
    b = split_rows(_rows(), seed=7)[1]
    assert [r["image_path"] for r in a] == [r["image_path"] for r in b]
