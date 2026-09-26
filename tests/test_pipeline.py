import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from opera_agent import pipeline as P

GPX = """<?xml version="1.0"?>
<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>
<trkpt lat="14.500" lon="121.000"><time>2025-05-13T01:00:00Z</time></trkpt>
<trkpt lat="14.501" lon="121.001"><time>2025-05-13T01:00:10Z</time><extensions><depth>5.0</depth></extensions></trkpt>
<trkpt lat="14.502" lon="121.002"><time>2025-05-13T01:00:20Z</time><extensions><depth>6.0</depth></extensions></trkpt>
</trkseg></trk></gpx>"""


# ------------------------------------------------------------------ discovery
@pytest.mark.parametrize("stem,key", [
    ("GH010123", ("GH0123", 1)), ("GX020123", ("GX0123", 2)),
    ("GOPR0123", ("GOPR0123", 0)), ("GP010123", ("GOPR0123", 1)), ("GP020123", ("GOPR0123", 2)),
    ("clip", ("clip", 0)),
])
def test_gopro_key(stem, key):
    assert P._gopro_key(stem) == key


def test_old_style_chapters_form_one_recording(tmp_path):
    files = [tmp_path / n for n in ("GP020123.MP4", "GOPR0123.MP4", "GP010123.MP4")]
    recs = P.group_recordings(files)
    assert len(recs) == 1
    assert [c.name for c in recs[0].chapters] == ["GOPR0123.MP4", "GP010123.MP4", "GP020123.MP4"]


# ------------------------------------------------------------------ GPX + interpolation
def test_read_gpx_and_interpolate(tmp_path):
    g = tmp_path / "t.gpx"; g.write_text(GPX)
    tr = P.read_gpx(g)
    assert len(tr.t) == 3 and np.isnan(tr.depth[0])
    t = np.array([tr.t[0] - 1, tr.t[0] + 5, tr.t[1] + 5, tr.t[-1] + 1])
    lat, lon, dep, st = P.interpolate(tr, t, max_gap_s=60)
    assert list(st) == ["before_track", "ok", "ok", "after_track"]
    assert lat[1] == pytest.approx(14.5005) and dep[1] == 5.0     # depth carried from the side that has it
    assert dep[2] == pytest.approx(5.5)


def test_gps_exif_roundtrip(tmp_path):
    jpg = tmp_path / "a.jpg"; Image.new("RGB", (8, 8)).save(jpg)
    P.write_gps_exif(jpg, -14.123456, 121.654321, datetime(2025, 5, 13, 9, 0, 0), "Asia/Manila", 4.2)
    lat, lon = P.read_gps_exif(jpg)
    assert lat == pytest.approx(-14.123456, abs=1e-7) and lon == pytest.approx(121.654321, abs=1e-7)


# ------------------------------------------------------------------ output-dir safety
def test_output_inside_videos_dir_is_refused(tmp_path):
    with pytest.raises(ValueError):
        P.check_output_dir(tmp_path, tmp_path)
    with pytest.raises(ValueError):
        P.check_output_dir(tmp_path, tmp_path / "out")
    P.check_output_dir(tmp_path / "videos", tmp_path / "videos_processed")      # sibling is fine


# ------------------------------------------------------------------ process_survey without ffmpeg
@pytest.fixture
def survey(tmp_path, monkeypatch):
    """A videos folder with one fake GoPro recording; ffprobe/ffmpeg are stubbed."""
    vids = tmp_path / "13MAY2025"; vids.mkdir()
    (vids / "GH010001.MP4").write_bytes(b"")
    (vids / "track.gpx").write_text(GPX)
    state = {"start": datetime(2025, 5, 13, 9, 0, 0)}         # 01:00:00 UTC in Asia/Manila

    def probe(rec):
        rec.start_local, rec.durations = state["start"], [25.0]
        return rec

    def extract(rec, out_dir, interval_s=1.0, clock_offset_s=0.0, max_width=None, log=print):
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for i in range(0, 25, 5):
            ts = rec.start_local + timedelta(seconds=i + clock_offset_s)
            f = out_dir / P._frame_name(ts, interval_s)
            Image.new("RGB", (16, 16), (i * 10, 80, 120)).save(f)
            frames.append((f, ts))
        return frames

    monkeypatch.setattr(P, "probe_recording", probe)
    monkeypatch.setattr(P, "extract_frames", extract)
    return vids, state


def test_process_survey(survey, tmp_path):
    vids, _ = survey
    rep = P.process_survey(str(vids), str(tmp_path / "out"), log=lambda *_: None)
    assert rep["status_counts"] == {"ok": 5}
    assert len(list((tmp_path / "out" / "geotagged").glob("*.jpg"))) == 5
    assert json.loads((tmp_path / "out" / "frames.geojson").read_text())["features"]


def test_rerun_removes_stale_frames(survey, tmp_path):
    vids, state = survey
    out = tmp_path / "out"
    P.process_survey(str(vids), str(out), log=lambda *_: None)
    (out / "notes.txt").write_text("keep me")
    state["start"] = datetime(2025, 5, 13, 9, 0, 3)          # "fixed" clock → different frame names
    P.process_survey(str(vids), str(out), log=lambda *_: None)
    names = sorted(p.name for p in (out / "geotagged").glob("*.jpg"))
    with open(out / "frame_data.csv") as f:
        assert names == sorted(r["frame_filename"] for r in csv.DictReader(f) if r["status"] == "ok")
    assert "frame_2025-05-13_09-00-00.jpg" not in names
    assert (out / "notes.txt").exists()                        # only the pipeline's own outputs are cleared


def test_process_survey_refuses_output_inside_videos(survey):
    vids, _ = survey
    with pytest.raises(ValueError):
        P.process_survey(str(vids), str(vids / "processed"), log=lambda *_: None)
