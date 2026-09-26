"""YOLO benthic classification of the geotagged frames (step after process_survey).

Reads output_dir/frame_data.csv, classifies every geotagged frame with a YOLO classification
model (e.g. the yolo11l-cls model trained in notebooks/YOLO_classification.ipynb) and writes:
  * the class into each JPG's EXIF ImageDescription. Only the EXIF block is rewritten, the image
    is not re-encoded, so GPS, DateTimeOriginal and depth stay exactly as geotagging wrote them;
  * classified/<class>/frame_*.jpg: the same frames sorted by class (copy by default);
  * benthic_class, class_conf, prob_<class> ... columns in frame_data.csv and frames.geojson;
  * qa_class_map.png and a "classification" section in report.json.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .pipeline import _frame_rows_ok

UNCERTAIN = "uncertain"
_CLASS_NOTE = "; benthic class "


def load_model(model_path: str | Path, device: str | None = None):
    from ultralytics import YOLO
    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    model = YOLO(str(model_path))
    if device:
        model.to(device)
    return model


def _probs(result) -> np.ndarray:
    data = result.probs.data
    return data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data, dtype=float)


def predict(model, paths: list[str], imgsz: int = 640, batch: int = 16, device: str | None = None):
    """(class names, probability matrix [n_images × n_classes]) for a list of image paths."""
    names, probs = None, []
    for i in range(0, len(paths), batch):
        kw = {"imgsz": imgsz, "verbose": False}
        if device:
            kw["device"] = device
        for r in model.predict(paths[i:i + batch], **kw):
            if names is None:
                names = [r.names[k] for k in sorted(r.names)]
            probs.append(_probs(r))
    return names or [], np.array(probs)


def write_class_exif(jpg: Path, cls: str, conf: float):
    """Append (or replace) the benthic class in ImageDescription, keeping all other EXIF."""
    import piexif
    exif = piexif.load(str(jpg))
    desc = exif["0th"].get(piexif.ImageIFD.ImageDescription, b"").decode("utf-8", "replace")
    desc = desc.split(_CLASS_NOTE)[0] + f"{_CLASS_NOTE}{cls} ({conf:.2f})"
    exif["0th"][piexif.ImageIFD.ImageDescription] = desc.encode()
    piexif.insert(piexif.dump(exif), str(jpg))


def classify_frames(output_dir: str | Path, model_path: str | Path | None = None, model=None,
                    conf: float = 0.0, imgsz: int = 640, batch: int = 16, device: str | None = None,
                    organize: str = "copy", log=print) -> dict:
    """Classify the geotagged frames of a process_survey output folder.

    conf: frames whose top-1 confidence is below this are labelled 'uncertain'.
    organize: 'copy' (default) copies each frame to classified/<class>/, 'move' moves it there
    (saves disk; the CSV path column follows it), 'none' only tags EXIF and the tables.
    """
    if organize not in ("copy", "move", "none"):
        raise ValueError("organize must be 'copy', 'move' or 'none'")
    out = Path(output_dir)
    csv_path, gj_path = out / "frame_data.csv", out / "frames.geojson"
    rows, ok = _frame_rows_ok(out)
    if not ok:
        raise ValueError(f"no geotagged frames listed in {csv_path}")
    if model is None:
        if not model_path:
            raise ValueError("pass model_path (the YOLO classification .pt, e.g. best.pt)")
        model = load_model(model_path, device)

    class_dir = out / "classified"
    if class_dir.exists():
        shutil.rmtree(class_dir)

    # drop columns from an earlier classification so a re-run with another model starts clean
    for r in rows:
        for k in [k for k in r if k in ("benthic_class", "class_conf", "classified_path") or k.startswith("prob_")]:
            del r[k]

    log(f"  classifying {len(ok)} geotagged frames …")
    names, probs = predict(model, [r["path"] for r in ok], imgsz, batch, device)
    counts: dict[str, int] = {}
    for r, p in zip(ok, probs):
        top = int(np.argmax(p))
        cls, c = names[top], float(p[top])
        label = cls if c >= conf else UNCERTAIN
        src = Path(r["path"])
        write_class_exif(src, label, c)
        dest = ""
        if organize != "none":
            (class_dir / label).mkdir(parents=True, exist_ok=True)
            dest = class_dir / label / src.name
            if organize == "copy":
                shutil.copy2(src, dest)
            else:
                shutil.move(str(src), dest)
                r["path"] = str(dest)
        r["benthic_class"], r["class_conf"] = label, round(c, 4)
        r["classified_path"] = str(dest)
        for n, v in zip(names, p):
            r[f"prob_{n}"] = round(float(v), 4)
        counts[label] = counts.get(label, 0) + 1

    # ---- tables: every row gets the same columns (blank for untagged frames)
    extra = ["benthic_class", "class_conf", "classified_path"] + [f"prob_{n}" for n in names]
    fields = [k for k in rows[0] if k not in extra] + extra
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader(); w.writerows(rows)
    by_name = {r["frame_filename"]: r for r in ok}
    if gj_path.exists():
        gj = json.loads(gj_path.read_text())
        for feat in gj["features"]:
            r = by_name.get(feat["properties"].get("frame_filename"))
            if r:
                feat["properties"].update({k: r[k] for k in extra if k != "classified_path"})
        gj_path.write_text(json.dumps(gj))

    summary = {"classified": len(ok), "class_counts": counts, "classes": names,
               "conf_threshold": conf, "model": str(model_path) if model_path else None,
               "classified_dir": str(class_dir) if organize != "none" else None, "organize": organize}
    try:
        summary["qa_map"] = str(_class_plot(ok, out / "qa_class_map.png"))
    except Exception as e:
        log(f"  (class map skipped: {e})")
    rep_path = out / "report.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())
        rep["classification"] = summary
        rep_path.write_text(json.dumps(rep, indent=2, default=str))
    log(f"  classes: {', '.join(f'{k} {v}' for k, v in sorted(counts.items(), key=lambda t: -t[1]))}")
    return summary


def _class_plot(rows: list[dict], path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    lat = np.array([float(r["latitude"]) for r in rows])
    for cls in sorted({r["benthic_class"] for r in rows}):
        sel = [r for r in rows if r["benthic_class"] == cls]
        ax.scatter([float(r["longitude"]) for r in sel], [float(r["latitude"]) for r in sel],
                   s=8, label=f"{cls} ({len(sel)})")
    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(lat))))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Benthic class of geotagged frames"); ax.legend(loc="best", markerscale=2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path
