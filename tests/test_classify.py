import csv
import json
from pathlib import Path

import numpy as np
import piexif
import pytest
from PIL import Image

from opera_agent import classify as C
from opera_agent import pipeline as P
from test_pipeline import survey  # noqa: F401  (fixture)

NAMES = {0: "corals", 1: "macroalgae", 2: "rubble", 3: "sand", 4: "seagrass"}


class _Probs:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)


class _Result:
    def __init__(self, data):
        self.probs, self.names = _Probs(data), NAMES


class FakeYOLO:
    """Stands in for ultralytics.YOLO: seagrass for frames whose name ends in an even second, else sand
    (with low confidence for :05 so the threshold can be tested)."""
    def __init__(self):
        self.calls, self.views = [], 0

    def predict(self, paths, **kw):
        out = []
        if paths and not isinstance(paths[0], str):            # flipped views (test-time augmentation)
            self.views += len(paths)
        else:
            self.calls.append(len(paths))
        for p in paths:
            p = p if isinstance(p, str) else p.info["path"]
            sec = int(str(p)[-6:-4])
            if sec == 5:
                out.append(_Result([0.1, 0.2, 0.2, 0.3, 0.2]))
            elif sec % 2 == 0:
                out.append(_Result([0.01, 0.01, 0.01, 0.02, 0.95]))
            else:
                out.append(_Result([0.02, 0.02, 0.02, 0.9, 0.04]))
        return out


@pytest.fixture
def processed(survey, tmp_path):  # noqa: F811
    vids, _ = survey
    out = tmp_path / "out"
    P.process_survey(str(vids), str(out), log=lambda *_: None)
    return out


def _rows(out):
    with open(out / "frame_data.csv") as f:
        return list(csv.DictReader(f))


def test_classify_keeps_gps_and_sorts_by_class(processed):
    before = {r["frame_filename"]: P.read_gps_exif(processed / "geotagged" / r["frame_filename"])
              for r in _rows(processed)}
    model = FakeYOLO()
    s = C.classify_frames(processed, model=model, conf=0.5, batch=2, log=lambda *_: None)

    assert model.calls == [2, 2, 1]                                    # batched
    assert s["class_counts"] == {"seagrass": 3, "sand": 1, "uncertain": 1}
    for r in _rows(processed):
        cls = r["benthic_class"]
        copy = processed / "classified" / cls / f"{cls}_{r['frame_filename']}"   # named after its class
        assert r["classified_path"] == str(copy) and copy.exists()
        assert P.read_gps_exif(copy) == before[r["frame_filename"]]     # coordinates retained
        exif = piexif.load(str(copy))
        desc = exif["0th"][piexif.ImageIFD.ImageDescription].decode()
        assert "echosounder depth" in desc and f"benthic class {cls}" in desc
        assert "probabilities: " in desc and "corals 0." in desc           # every class's probability
        assert exif["Exif"][piexif.ExifIFD.DateTimeOriginal]            # time retained
        original = processed / "geotagged" / r["frame_filename"]
        assert Image.open(copy).getpixel((1, 1)) != Image.open(original).getpixel((1, 1))   # panel drawn
        assert "probabilities: " in piexif.load(str(original))["0th"][piexif.ImageIFD.ImageDescription].decode()
        assert float(r["prob_seagrass"]) >= 0 and r["latitude"]

    gj = json.loads((processed / "frames.geojson").read_text())
    assert {f["properties"]["benthic_class"] for f in gj["features"]} == {"seagrass", "sand", "uncertain"}
    assert json.loads((processed / "report.json").read_text())["classification"]["classified"] == 5


def test_reclassify_replaces_previous_result(processed):
    C.classify_frames(processed, model=FakeYOLO(), conf=0.5, log=lambda *_: None)
    C.classify_frames(processed, model=FakeYOLO(), conf=0.0, log=lambda *_: None)
    assert not (processed / "classified" / "uncertain").exists()
    descs = [piexif.load(str(p))["0th"][piexif.ImageIFD.ImageDescription].decode()
             for p in (processed / "geotagged").glob("*.jpg")]
    assert all(d.count("benthic class") == 1 for d in descs)


def test_move_updates_paths(processed):
    C.classify_frames(processed, model=FakeYOLO(), organize="move", log=lambda *_: None)
    assert not list((processed / "geotagged").glob("*.jpg"))
    for r in _rows(processed):
        assert r["path"] == r["classified_path"]


def test_process_survey_runs_classification(survey, tmp_path, monkeypatch):  # noqa: F811
    vids, _ = survey
    monkeypatch.setattr(C, "load_model", lambda path, device=None: FakeYOLO())
    (tmp_path / "best.pt").write_bytes(b"")
    rep = P.process_survey(str(vids), str(tmp_path / "out"), model_path=str(tmp_path / "best.pt"),
                           log=lambda *_: None)
    assert rep["classification"]["classified"] == 5
    assert "benthic_class" in _rows(tmp_path / "out")[0]


def test_classification_failure_keeps_geotagging(survey, tmp_path):  # noqa: F811
    vids, _ = survey
    rep = P.process_survey(str(vids), str(tmp_path / "out"), model_path=str(tmp_path / "missing.pt"),
                           log=lambda *_: None)
    assert rep["status_counts"] == {"ok": 5} and "error" in rep["classification"]
    assert "error" in json.loads((tmp_path / "out" / "report.json").read_text())["classification"]


# ---------------------------------------------------------------- any folder of images (inference notebook)
def test_classify_images_keeps_subfolders_and_metadata(processed, tmp_path):
    src = tmp_path / "input"
    for sub in ("dive1", "dive2"):
        (src / sub).mkdir(parents=True)
    frames = sorted((processed / "geotagged").glob("*.jpg"))
    for f in frames[:3]:
        (src / "dive1" / f.name).write_bytes(f.read_bytes())
    for f in frames[3:]:
        (src / "dive2" / f.name).write_bytes(f.read_bytes())
    Image.new("RGB", (40, 30), (10, 20, 30)).save(src / "dive2" / "extra_2025-05-13_09-00-10.png")   # no EXIF
    before = {p: p.read_bytes() for p in src.rglob("*") if p.is_file()}

    out = tmp_path / "output"
    s = C.classify_images(src, out, model=FakeYOLO(), conf=0.5, log=lambda *_: None)
    assert s["images"] == 6 and s["with_gps"] == 5
    for p, data in before.items():
        assert p.read_bytes() == data                                          # inputs untouched

    with open(out / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        dest = out / r["folder"] / r["predicted_class"] / f"{r['predicted_class']}_{Path(r['filename']).stem}.jpg"
        assert r["output_path"] == str(dest) and dest.exists()                # dive1/seagrass/seagrass_frame_…jpg
        assert r["folder"] in ("dive1", "dive2") and "prob_seagrass" in r
        desc = piexif.load(str(dest))["0th"][piexif.ImageIFD.ImageDescription].decode()
        assert f"benthic class {r['predicted_class']}" in desc and "probabilities: " in desc
        if r["latitude"]:
            assert P.read_gps_exif(dest) == pytest.approx((float(r["latitude"]), float(r["longitude"])), abs=1e-6)
    assert json.loads((out / "classification_results.json").read_text())["summary"]["images"] == 6


def test_classify_images_skips_its_own_output(processed):
    out = processed / "geotagged" / "_classified"
    C.classify_images(processed / "geotagged", out, model=FakeYOLO(), log=lambda *_: None)
    s = C.classify_images(processed / "geotagged", out, model=FakeYOLO(), log=lambda *_: None)
    assert s["images"] == 5


# ---------------------------------------------------------------- test-time augmentation, smoothing, margin
class MirrorDisagrees(FakeYOLO):
    """Says seagrass on the original, sand on every flipped view."""
    def predict(self, paths, **kw):
        flipped = paths and not isinstance(paths[0], str)
        self.views += len(paths) if flipped else 0
        return [_Result([0, 0, 0, 0.8, 0.2] if flipped else [0, 0, 0, 0.2, 0.8]) for _ in paths]


def test_tta_averages_views(processed):
    paths = [str(p) for p in sorted((processed / "geotagged").glob("*.jpg"))]
    m = MirrorDisagrees()
    names, p = C.predict(m, paths, batch=2, tta="hflip")
    assert m.views == 5 and np.allclose(p[:, 3], 0.5) and np.allclose(p[:, 4], 0.5)
    _, p = C.predict(MirrorDisagrees(), paths, tta="flips")
    assert np.allclose(p[:, 4], (0.8 + 3 * 0.2) / 4)
    _, p = C.predict(MirrorDisagrees(), paths, tta="none")
    assert np.allclose(p[:, 4], 0.8)
    with pytest.raises(ValueError):
        C.predict(m, paths, tta="rotate")


def test_smooth_over_time_stays_within_recording():
    probs = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1]], dtype=float)
    times = [0, 1, 2, 3, 4, 2]
    groups = ["A", "A", "A", "A", "A", "B"]
    out = C.smooth_over_time(probs, times, groups, window_s=3)              # ±1.5 s → 3 neighbouring frames
    assert out[2].tolist() == pytest.approx([2 / 3, 1 / 3])                  # the lone flip is outvoted
    assert out[0].tolist() == pytest.approx([1, 0])                          # edge: only itself + next
    assert out[5].tolist() == [0, 1]                                         # other recording untouched
    assert C.smooth_over_time(probs, times, groups, 0) is probs


def test_margin():
    assert C.margin([0.1, 0.6, 0.3]) == pytest.approx(0.3)


def test_classify_frames_with_smoothing_keeps_raw_class(processed):
    s = C.classify_frames(processed, model=FakeYOLO(), conf=0.4, smooth_s=12, log=lambda *_: None)
    rows = _rows(processed)
    assert s["smooth_s"] == 12 and s["tta"] == "hflip"
    assert {"raw_class", "class_margin"} <= set(rows[0])
    assert [r["raw_class"] for r in rows] == ["seagrass", "uncertain", "seagrass", "sand", "seagrass"]
    # ±6 s with frames 5 s apart → each frame averaged with its neighbours, e.g. :05 = mean of :00, :05, :10
    assert [r["benthic_class"] for r in rows] == ["seagrass", "seagrass", "sand", "seagrass", "seagrass"]
    assert float(rows[1]["prob_seagrass"]) == pytest.approx((0.95 + 0.2 + 0.95) / 3, abs=1e-4)
