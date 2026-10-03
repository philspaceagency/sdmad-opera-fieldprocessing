import json

import numpy as np
import pytest
from PIL import Image

from opera_agent import yolo_train as T
from test_classify import NAMES, _Result


def noise(seed, size=(96, 64)):
    """A 'scene': smooth random blobs (coarse structure, like a real frame) plus fine grain."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, (4, 6, 3), dtype=np.uint8)
    img = np.asarray(Image.fromarray(coarse).resize(size, Image.Resampling.BICUBIC), dtype=np.int16)
    img = img + rng.integers(-10, 10, img.shape)
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def test_near_duplicates_across_splits_only(tmp_path):
    for split in ("train", "valid", "test"):
        (tmp_path / split / "sand").mkdir(parents=True)
    leaked = noise(1)
    leaked.save(tmp_path / "train" / "sand" / "a.jpg", quality=95)
    leaked.resize((80, 54)).save(tmp_path / "test" / "sand" / "a_small.jpg", quality=70)   # same frame, resized
    noise(1).save(tmp_path / "train" / "sand" / "a_copy.jpg")          # duplicate inside train: not leakage
    for i in range(2, 6):
        noise(i).save(tmp_path / ("valid" if i % 2 else "test") / "sand" / f"n{i}.jpg")
    pairs = T.find_near_duplicates(tmp_path, log=lambda *_: None)
    assert {frozenset((p["split_a"], p["split_b"])) for p in pairs} == {frozenset(("train", "test"))}
    assert len(pairs) == 2                                              # a.jpg and a_copy.jpg ↔ a_small.jpg
    assert all("test/sand/a_small.jpg" in (p["image_a"], p["image_b"]) for p in pairs)


class ByFilename:
    """Predicts the class in the file name (prefix before '_'), except files containing 'wrong' → sand."""
    def predict(self, paths, **kw):
        out = []
        for p in paths:
            p = p if isinstance(p, str) else p.info["path"]
            name = p.rsplit("/", 1)[-1]
            cls = "sand" if "wrong" in name else name.split("_")[0]
            probs = [0.05] * 5
            probs[[v for v in NAMES.values()].index(cls)] = 0.8
            out.append(_Result(probs))
        return out


def test_evaluate(tmp_path):
    split = tmp_path / "test"
    for cls, files in {"seagrass": ["seagrass_1", "seagrass_2", "seagrass_wrong"], "sand": ["sand_1"],
                       "corals": ["corals_1", "corals_2"]}.items():
        (split / cls).mkdir(parents=True)
        for i, f in enumerate(files):
            noise(i).save(split / cls / f"{f}.jpg")
    res = T.evaluate(ByFilename(), split, out_dir=tmp_path / "eval", log=lambda *_: None)
    assert res["images"] == 6 and res["accuracy"] == pytest.approx(5 / 6, abs=1e-4)
    assert res["per_class"]["seagrass"] == {"precision": 1.0, "recall": pytest.approx(0.6667, abs=1e-4),
                                            "f1": pytest.approx(0.8, abs=1e-4), "support": 3}
    assert res["per_class"]["sand"]["precision"] == 0.5 and res["per_class"]["rubble"]["support"] == 0
    assert res["macro_f1"] == pytest.approx((0.8 + 2 / 3 + 1.0) / 3, abs=1e-3)   # only classes present
    assert res["misclassified"] == [{"image": "seagrass/seagrass_wrong.jpg", "true": "seagrass",
                                     "predicted": "sand", "confidence": 0.8}]
    assert (tmp_path / "eval" / "confusion_matrix.png").exists()
    assert json.loads((tmp_path / "eval" / "metrics.json").read_text())["accuracy"] == res["accuracy"]


def test_evaluate_rejects_unknown_class_folder(tmp_path):
    (tmp_path / "test" / "kelp").mkdir(parents=True)
    noise(0).save(tmp_path / "test" / "kelp" / "seagrass_1.jpg")
    with pytest.raises(ValueError, match="kelp"):
        T.evaluate(ByFilename(), tmp_path / "test", log=lambda *_: None)


def test_save_model_card(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"weights")
    card = T.save_model_card(tmp_path / "best.pt", tmp_path / "models", "yolo11l-benthic-cls",
                             metrics={"accuracy": 0.9, "macro_f1": 0.88, "classes": ["a"], "tta": "hflip"},
                             train_args={"epochs": 100, "device": object()}, leakage_pairs=3)
    assert (tmp_path / "models" / "yolo11l-benthic-cls.pt").read_bytes() == b"weights"
    saved = json.loads((tmp_path / "models" / "yolo11l-benthic-cls.json").read_text())
    assert saved["test_metrics"]["macro_f1"] == 0.88 and saved["near_duplicate_pairs_across_splits"] == 3
    assert saved["train_args"]["epochs"] == 100 and card["classes"] == ["a"]
