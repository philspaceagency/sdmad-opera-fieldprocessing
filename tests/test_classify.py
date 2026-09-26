import csv
import json

import numpy as np
import piexif
import pytest

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
        self.calls = []

    def predict(self, paths, **kw):
        self.calls.append(len(paths))
        out = []
        for p in paths:
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
        copy = processed / "classified" / r["benthic_class"] / r["frame_filename"]
        assert r["classified_path"] == str(copy) and copy.exists()
        assert P.read_gps_exif(copy) == before[r["frame_filename"]]     # coordinates retained
        exif = piexif.load(str(copy))
        desc = exif["0th"][piexif.ImageIFD.ImageDescription].decode()
        assert "echosounder depth" in desc and f"benthic class {r['benthic_class']}" in desc
        assert exif["Exif"][piexif.ExifIFD.DateTimeOriginal]            # time retained
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
