"""YOLO benthic classification.

classify_frames(): the step after process_survey. Reads output_dir/frame_data.csv, classifies every geotagged
frame with a YOLO classification model (e.g. the yolo11l-cls model trained in notebooks/YOLO_classification.ipynb)
and writes:
  * the class and class probabilities into each geotagged JPG's EXIF ImageDescription. Only the EXIF block is
    rewritten, the image is not re-encoded, so GPS, DateTimeOriginal and depth stay exactly as geotagging wrote them;
  * classified/<class>/<class>_frame_*.jpg: the frames sorted into one folder per class, named after their
    class, with the probability of every class drawn on the image and all EXIF (GPS, time, depth) kept;
  * benthic_class, class_conf, prob_<class> ... columns in frame_data.csv and frames.geojson;
  * qa_class_map.png and a "classification" section in report.json.

classify_images(): the same for any folder of images (notebooks/YOLO_inference.ipynb), keeping its sub-folders.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .pipeline import _frame_rows_ok

UNCERTAIN = "uncertain"
COLUMNS = ("benthic_class", "class_conf", "class_margin", "raw_class", "classified_path")   # added to frame_data.csv
_CLASS_NOTE = "; benthic class "
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


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


# Test-time augmentation: the probabilities are averaged over these views of each image. Benthic frames are
# looked at from above, so a mirrored frame shows the same bottom type; averaging makes the result steadier.
# "flips" also uses vertical flips: best with a model trained with flipud (notebooks/YOLO_classification.ipynb).
TTA_VIEWS = {"none": (None,), "hflip": (None, "FLIP_LEFT_RIGHT"),
             "flips": (None, "FLIP_LEFT_RIGHT", "FLIP_TOP_BOTTOM", "ROTATE_180")}


def _view(path: str, op: str):
    from PIL import Image
    with Image.open(path) as im:
        out = im.convert("RGB").transpose(getattr(Image.Transpose, op))
    out.info["path"] = str(path)                   # where the view came from (logging / tests)
    return out


def predict(model, paths: list[str], imgsz: int = 640, batch: int = 16, device: str | None = None,
            tta: str = "hflip"):
    """(class names, probability matrix [n_images × n_classes]) for a list of image paths.
    tta: "none", "hflip" (default: original + mirrored) or "flips" (+ vertical flip and 180° rotation)."""
    if tta not in TTA_VIEWS:
        raise ValueError(f"tta must be one of {', '.join(TTA_VIEWS)}")
    kw = {"imgsz": imgsz, "verbose": False}
    if device:
        kw["device"] = device
    names, probs = None, []
    for i in range(0, len(paths), batch):
        chunk, acc = paths[i:i + batch], None
        for op in TTA_VIEWS[tta]:
            src = chunk if op is None else [_view(p, op) for p in chunk]
            results = model.predict(src, **kw)
            if names is None:
                names = [results[0].names[k] for k in sorted(results[0].names)]
            p = np.stack([_probs(r) for r in results])
            acc = p if acc is None else acc + p
        probs.extend(acc / len(TTA_VIEWS[tta]))
    return names or [], np.array(probs)


def smooth_over_time(probs: np.ndarray, times: list[float], groups: list[str], window_s: float) -> np.ndarray:
    """Average each frame's probabilities with those of the frames within ±window_s/2 seconds of the same
    recording. The bottom changes slowly along a transect, so this removes single-frame flips between classes
    (a passing fish, a blurred frame) without blurring real boundaries much. window_s <= 0: unchanged."""
    if window_s <= 0 or len(probs) == 0:
        return probs
    out = np.empty_like(probs, dtype=float)
    times, groups = np.asarray(times, dtype=float), np.asarray(groups)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        idx = idx[np.argsort(times[idx], kind="stable")]
        t = times[idx]
        csum = np.vstack([np.zeros(probs.shape[1]), np.cumsum(probs[idx], axis=0)])
        lo = np.searchsorted(t, t - window_s / 2, side="left")
        hi = np.searchsorted(t, t + window_s / 2, side="right")
        out[idx] = (csum[hi] - csum[lo]) / (hi - lo)[:, None]
    return out


def margin(p) -> float:
    """Top-1 minus top-2 probability: small (< ~0.2) means the model hesitated between two classes."""
    s = np.sort(np.asarray(p, dtype=float))
    return float(s[-1] - s[-2]) if len(s) > 1 else float(s[-1])


def prob_text(names: list[str], p) -> str:
    """'seagrass 0.951, sand 0.030, ...' (highest first)."""
    return ", ".join(f"{n} {float(v):.3f}" for n, v in sorted(zip(names, p), key=lambda t: -t[1]))


def _load_exif(path: Path) -> dict:
    import piexif
    try:
        return piexif.load(str(path))
    except Exception:                               # PNG, or no EXIF at all
        return {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}


def _with_class(exif: dict, cls: str, conf: float, probs: str = "") -> dict:
    """Append (or replace) the benthic class + probabilities in ImageDescription, keeping all other EXIF."""
    import piexif
    desc = exif["0th"].get(piexif.ImageIFD.ImageDescription, b"").decode("utf-8", "replace")
    desc = desc.split(_CLASS_NOTE)[0] + f"{_CLASS_NOTE}{cls} ({conf:.2f})" + (f"; probabilities: {probs}" if probs else "")
    exif["0th"][piexif.ImageIFD.ImageDescription] = desc.encode()
    return exif


def write_class_exif(jpg: Path, cls: str, conf: float, probs: str = ""):
    """Tag a JPG in place with its class; only the EXIF block is rewritten (no re-encoding)."""
    import piexif
    piexif.insert(piexif.dump(_with_class(piexif.load(str(jpg)), cls, conf, probs)), str(jpg))


def classified_name(cls: str, src: Path) -> str:
    """seagrass_frame_2025-05-13_09-00-00.jpg: the class first, the original name (time) kept for traceability."""
    return f"{cls}_{src.stem}.jpg"


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    for name in (("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf") if bold
                 else ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size)         # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def draw_probabilities(img, label: str, names: list[str], p):
    """Draw a panel (top-left) with the predicted class and a bar + percentage for every class."""
    from PIL import Image, ImageDraw
    img = img.convert("RGB")
    w, h = img.size
    size = max(12, int(min(w, h) / 32))
    big, small = _font(int(size * 1.25), bold=True), _font(size)
    order = sorted(zip(names, (float(v) for v in p)), key=lambda t: -t[1])
    top_name, top_p = order[0]
    title = f"{label.upper()} {top_p:.1%}" if label != UNCERTAIN else f"UNCERTAIN (top: {top_name} {top_p:.1%})"
    pad, line = size // 2, int(size * 1.45)
    name_w = max(ImageDraw.Draw(img).textlength(n, font=small) for n in names)
    bar_w = max(size * 6, int(w * 0.12))
    panel_w = max(int(ImageDraw.Draw(img).textlength(title, font=big)), int(name_w + bar_w + size * 4.5)) + 2 * pad
    panel_h = pad * 2 + int(size * 1.9) + line * len(order)
    overlay = img.copy()
    d = ImageDraw.Draw(overlay)
    d.rectangle([0, 0, panel_w, panel_h], fill=(0, 0, 0))
    img = Image.blend(img, overlay, 0.6)               # 60 % black panel, the frame stays visible behind it
    d = ImageDraw.Draw(img)
    d.text((pad, pad), title, font=big, fill=(255, 255, 255))
    y = pad + int(size * 1.9)
    for n, v in order:
        hi = n == top_name and label != UNCERTAIN
        colour = (80, 220, 120) if hi else (200, 200, 200)
        d.text((pad, y), n, font=small, fill=colour)
        x0 = pad + name_w + size // 2
        d.rectangle([x0, y + size * 0.2, x0 + bar_w, y + size * 0.9], outline=(120, 120, 120))
        d.rectangle([x0, y + size * 0.2, x0 + max(1, int(bar_w * v)), y + size * 0.9], fill=colour)
        d.text((x0 + bar_w + size // 2, y), f"{v:.1%}", font=small, fill=colour)
        y += line
    return img


def write_classified(src: Path, dest: Path, label: str, conf: float, names: list[str], p,
                     annotate: bool = True) -> Path:
    """Write the classified copy of `src` to `dest`: probabilities drawn on the image (annotate=True) and all of
    src's EXIF (GPS, time, depth) kept, plus the class and probabilities in ImageDescription."""
    import piexif
    from PIL import Image
    dest.parent.mkdir(parents=True, exist_ok=True)
    exif = _with_class(_load_exif(src), label, conf, prob_text(names, p))
    exif["thumbnail"], exif["1st"] = None, {}      # the old thumbnail wouldn't show the panel
    with Image.open(src) as im:
        out = draw_probabilities(im, label, names, p) if annotate else im.convert("RGB")
        out.save(dest, "JPEG", quality=95, exif=piexif.dump(exif))
    return dest


def _label(names: list[str], p, conf: float) -> tuple[str, float]:
    top = int(np.argmax(p))
    c = float(p[top])
    return (names[top] if c >= conf else UNCERTAIN), c


def classify_frames(output_dir: str | Path, model_path: str | Path | None = None, model=None,
                    conf: float = 0.0, imgsz: int = 640, batch: int = 16, device: str | None = None,
                    organize: str = "copy", annotate: bool = True, tta: str = "hflip", smooth_s: float = 0.0,
                    log=print) -> dict:
    """Classify the geotagged frames of a process_survey output folder.

    conf: frames whose top-1 confidence is below this are labelled 'uncertain'.
    organize: 'copy' (default) writes classified/<class>/<class>_<frame>.jpg next to the untouched geotagged
    frame, 'move' removes the geotagged frame afterwards (saves disk; the CSV path column follows it),
    'none' only tags EXIF and the tables.
    annotate: draw the class probabilities on the classified copies (EXIF is kept either way).
    tta: test-time augmentation, see TTA_VIEWS ("hflip" by default).
    smooth_s: average probabilities over this many seconds of neighbouring frames of the same recording
    (e.g. 3); the per-frame (unsmoothed) class is kept in the raw_class column. 0 = off.
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
        for k in [k for k in r if k in COLUMNS or k.startswith("prob_")]:
            del r[k]

    log(f"  classifying {len(ok)} geotagged frames …")
    names, raw = predict(model, [r["path"] for r in ok], imgsz, batch, device, tta)
    from datetime import datetime
    probs = smooth_over_time(raw, [datetime.fromisoformat(r["frame_datetime"]).timestamp() for r in ok],
                             [r["recording"] for r in ok], smooth_s)
    counts: dict[str, int] = {}
    for r, p, p_raw in zip(ok, probs, raw):
        label, c = _label(names, p, conf)
        src = Path(r["path"])
        write_class_exif(src, label, c, prob_text(names, p))
        dest = ""
        if organize != "none":
            dest = write_classified(src, class_dir / label / classified_name(label, src), label, c, names, p, annotate)
            if organize == "move":
                src.unlink()
                r["path"] = str(dest)
        r["benthic_class"], r["class_conf"], r["class_margin"] = label, round(c, 4), round(margin(p), 4)
        if smooth_s > 0:
            r["raw_class"] = _label(names, p_raw, conf)[0]
        r["classified_path"] = str(dest)
        for n, v in zip(names, p):
            r[f"prob_{n}"] = round(float(v), 4)
        counts[label] = counts.get(label, 0) + 1

    # ---- tables: every row gets the same columns (blank for untagged frames)
    extra = [c for c in COLUMNS if c != "raw_class" or smooth_s > 0] + [f"prob_{n}" for n in names]
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
               "classified_dir": str(class_dir) if organize != "none" else None, "organize": organize,
               "tta": tta, "smooth_s": smooth_s,
               "low_margin_frames": sum(1 for r in ok if r["class_margin"] < 0.2)}
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


def classify_images(input_dir: str | Path, output_dir: str | Path, model_path: str | Path | None = None, model=None,
                    conf: float = 0.0, imgsz: int = 640, batch: int = 16, device: str | None = None,
                    annotate: bool = True, recursive: bool = True, tta: str = "hflip", log=print) -> dict:
    """Classify every image under input_dir (e.g. a folder of geotagged frames) into output_dir.

    Sub-folders are kept, each with its own class folders:
      input/13MAY2025/frame_x.jpg → output/13MAY2025/seagrass/seagrass_frame_x.jpg
    Each output image has the class probabilities drawn on it (annotate=True) and keeps the input's EXIF
    (GPS, time, depth), with the class and probabilities added to ImageDescription. Input files are not changed.
    Also writes output_dir/summary.csv and classification_results.json.
    """
    from .pipeline import read_gps_exif
    src_root, out = Path(input_dir).resolve(), Path(output_dir).resolve()
    it = src_root.rglob("*") if recursive else src_root.glob("*")
    images = sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXT and not p.name.startswith("._")
                    and out not in p.parents)               # never re-read our own output
    if not images:
        raise FileNotFoundError(f"no images under {input_dir}")
    if model is None:
        if not model_path:
            raise ValueError("pass model_path (the YOLO classification .pt, e.g. best.pt)")
        model = load_model(model_path, device)

    log(f"classifying {len(images)} images …")
    names, probs = predict(model, [str(p) for p in images], imgsz, batch, device, tta)
    rows, counts = [], {}
    for src, p in zip(images, probs):
        label, c = _label(names, p, conf)
        rel = src.parent.relative_to(src_root)
        dest = write_classified(src, out / rel / label / classified_name(label, src), label, c, names, p, annotate)
        try:
            gps = read_gps_exif(src)
        except Exception:
            gps = None
        rows.append({"filename": src.name, "folder": rel.as_posix() if rel.parts else "",
                     "predicted_class": label, "confidence": round(c, 4), "margin": round(margin(p), 4),
                     **{f"prob_{n}": round(float(v), 4) for n, v in zip(names, p)},
                     "latitude": round(gps[0], 7) if gps else "", "longitude": round(gps[1], 7) if gps else "",
                     "input_path": str(src), "output_path": str(dest)})
        counts[label] = counts.get(label, 0) + 1

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {"images": len(rows), "class_counts": counts, "classes": names, "conf_threshold": conf, "tta": tta,
               "with_gps": sum(1 for r in rows if r["latitude"] != ""), "output_dir": str(out),
               "model": str(model_path) if model_path else None,
               "files": {"csv": str(out / "summary.csv"), "json": str(out / "classification_results.json")}}
    (out / "classification_results.json").write_text(json.dumps({"summary": summary, "images": rows}, indent=2))
    log(f"  classes: {', '.join(f'{k} {v}' for k, v in sorted(counts.items(), key=lambda t: -t[1]))}")
    log(f"  GPS found in {summary['with_gps']}/{len(rows)} images; outputs in {out}")
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
