"""Helpers for training and judging the YOLO benthic classifier (used by notebooks/YOLO_classification.ipynb).

  find_near_duplicates()  leakage check: near-identical images in different splits (train vs valid/test).
                          Consecutive video frames of the same spot look almost the same; if they end up in both
                          train and test, the test score says little about new surveys.
  evaluate()              per-class precision / recall / F1, confusion matrix and the misclassified images on a
                          split, with the same prediction code (and test-time augmentation) as the pipeline.
  save_model_card()       copies best.pt next to a JSON card (classes, metrics, training settings).
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .classify import IMAGE_EXT, predict

# Augmentations suited to top-down underwater frames, for model.train(**TRAIN_AUGMENT).
#  * flips both ways: the bottom seen from above has no "up", so vertical flips are as valid as horizontal ones
#  * hue / saturation / brightness: water colour and light change with depth, turbidity and time of day
#  * randaugment + random erasing + scale: robustness to blur, particles, fish and partial views
TRAIN_AUGMENT = dict(fliplr=0.5, flipud=0.5, hsv_h=0.02, hsv_s=0.6, hsv_v=0.4,
                     auto_augment="randaugment", erasing=0.4, scale=0.5)


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def dhash(path: str | Path, size: int = 8) -> int:
    """64-bit difference hash: robust to resizing / recompression, changes when the content changes."""
    from PIL import Image
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def find_near_duplicates(dataset_dir: str | Path, splits=("train", "valid", "test"), max_distance: int = 4,
                         log=print) -> list[dict]:
    """Image pairs in *different* splits whose hashes differ in at most max_distance of 64 bits (0 = identical
    look). Returns [{split_a, image_a, split_b, image_b, distance}], closest first."""
    root = Path(dataset_dir)
    files, split_of = [], []
    for s in splits:
        if (root / s).is_dir():
            imgs = _images(root / s)
            files += imgs; split_of += [s] * len(imgs)
    if not files:
        raise FileNotFoundError(f"no images in {', '.join(splits)} under {root}")
    h = np.array([dhash(f) for f in files], dtype=np.uint64)
    split_of = np.array(split_of)
    pairs = []
    for i in range(len(files) - 1):
        x = np.bitwise_xor(h[i + 1:], h[i])
        dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for j in np.flatnonzero((dist <= max_distance) & (split_of[i + 1:] != split_of[i])):
            k = i + 1 + j
            pairs.append({"split_a": split_of[i], "image_a": str(files[i].relative_to(root)),
                          "split_b": split_of[k], "image_b": str(files[k].relative_to(root)),
                          "distance": int(dist[j])})
    pairs.sort(key=lambda d: d["distance"])
    per_split = {s: len({p["image_b"] for p in pairs if p["split_b"] == s} |
                         {p["image_a"] for p in pairs if p["split_a"] == s}) for s in splits}
    log(f"{len(files)} images; {len(pairs)} near-duplicate pairs across splits "
        f"(images involved per split: {per_split})")
    return pairs


def evaluate(model, split_dir: str | Path, out_dir: str | Path | None = None, tta: str = "hflip",
             imgsz: int = 640, batch: int = 16, device: str | None = None, log=print) -> dict:
    """Score the model on a folder laid out as <split_dir>/<class>/*.jpg (e.g. the Roboflow test split)."""
    root = Path(split_dir)
    truth, paths = [], []
    for cls_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        imgs = _images(cls_dir)
        paths += [str(p) for p in imgs]; truth += [cls_dir.name] * len(imgs)
    if not paths:
        raise FileNotFoundError(f"no <class>/<image> folders under {root}")
    names, probs = predict(model, paths, imgsz, batch, device, tta)
    unknown = sorted(set(truth) - set(names))
    if unknown:
        raise ValueError(f"folders {unknown} are not classes of the model ({names})")
    pred = [names[i] for i in probs.argmax(axis=1)]
    idx = {n: i for i, n in enumerate(names)}
    cm = np.zeros((len(names), len(names)), dtype=int)               # rows: true class, columns: predicted
    for t, p in zip(truth, pred):
        cm[idx[t], idx[p]] += 1
    per_class = {}
    for n, i in idx.items():
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[n] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                        "f1": round(float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0, 4),
                        "support": int(cm[i, :].sum())}
    present = [n for n in names if per_class[n]["support"]]
    res = {"split": str(root), "images": len(paths), "tta": tta, "classes": names,
           "accuracy": round(float(np.trace(cm) / cm.sum()), 4),
           "macro_f1": round(float(np.mean([per_class[n]["f1"] for n in present])), 4),
           "per_class": per_class, "confusion_matrix": cm.tolist(),
           "misclassified": [{"image": str(Path(p).relative_to(root)), "true": t, "predicted": q,
                              "confidence": round(float(pr.max()), 4)}
                             for p, t, q, pr in zip(paths, truth, pred, probs) if t != q]}
    log(f"{root.name}: accuracy {res['accuracy']:.3f}, macro-F1 {res['macro_f1']:.3f} on {len(paths)} images "
        f"({len(res['misclassified'])} wrong)")
    for n in names:
        c = per_class[n]
        log(f"  {n:<12} precision {c['precision']:.3f}  recall {c['recall']:.3f}  F1 {c['f1']:.3f}  (n={c['support']})")
    if out_dir:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(res, indent=2))
        with open(out / "misclassified.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image", "true", "predicted", "confidence"])
            w.writeheader(); w.writerows(res["misclassified"])
        try:
            res["confusion_png"] = str(_confusion_plot(cm, names, out / "confusion_matrix.png"))
        except Exception as e:
            log(f"  (confusion plot skipped: {e})")
    return res


def _confusion_plot(cm: np.ndarray, names: list[str], path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right"); ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (row %)")
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def save_model_card(weights: str | Path, dest_dir: str | Path, name: str, metrics: dict | None = None,
                    train_args: dict | None = None, dataset: str | None = None,
                    leakage_pairs: int | None = None) -> dict:
    """Copy best.pt to dest_dir/<name>.pt and write dest_dir/<name>.json describing it."""
    dest = Path(dest_dir); dest.mkdir(parents=True, exist_ok=True)
    pt = dest / f"{name}.pt"
    shutil.copy2(weights, pt)
    card = {"name": name, "weights": str(pt), "source_weights": str(weights),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dataset": dataset,
            "classes": (metrics or {}).get("classes"), "test_metrics": metrics and
            {k: metrics[k] for k in ("accuracy", "macro_f1", "per_class", "tta", "images") if k in metrics},
            "near_duplicate_pairs_across_splits": leakage_pairs,
            "train_args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                           for k, v in (train_args or {}).items()}}
    (dest / f"{name}.json").write_text(json.dumps(card, indent=2))
    return card
