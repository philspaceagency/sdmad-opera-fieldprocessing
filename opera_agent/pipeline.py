"""Portable OpERA field-processing pipeline:
GoPro videos (+ chapters) → timestamped frames → GPX (echosounder) interpolation → geotagged JPGs.

Re-implements scripts/concatenate_video.py, extract_photo.py and geotag_gpx_frames.py from
philspaceagency/sdmad-opera-fieldprocessing so they run on Colab/Linux/Mac/Windows:
  * no Windows exiftool.exe needed (reads times with ffprobe or exiftool, writes EXIF with piexif)
  * GoPro chapters grouped per recording and ordered by chapter (GH01xxxx, GH02xxxx, ...)
  * GPX points without depth are still used for position (depth is carried along when present)
  * frames outside the GPX track, or across big GPS gaps, are reported instead of silently skipped
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

LOCAL_TZ = "Asia/Manila"
VIDEO_EXT = {".mp4", ".mov"}


# =================================================================== discovery
_GOPRO_NEW = re.compile(r"^G([HXL])(\d{2})(\d{4})$", re.I)    # GH010123 → chapter 01, file 0123
_GOPRO_OLD_FIRST = re.compile(r"^GOPR(\d{4})$", re.I)          # GOPR0123 (chapter 0)
_GOPRO_OLD_NEXT = re.compile(r"^GP(\d{2})(\d{4})$", re.I)      # GP010123


@dataclass
class Recording:
    """One GoPro recording = one or more chapter files, played back to back."""
    key: str
    chapters: list[Path]
    start_local: datetime | None = None      # naive, camera wall-clock time
    durations: list[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return float(sum(self.durations))

    def summary(self) -> dict:
        end = self.start_local + timedelta(seconds=self.duration) if self.start_local else None
        return {"recording": self.key,
                "chapters": [c.name for c in self.chapters],
                "start_local": self.start_local.isoformat(sep=" ") if self.start_local else None,
                "end_local": end.isoformat(sep=" ", timespec="seconds") if end else None,
                "duration_min": round(self.duration / 60, 1)}


def _gopro_key(stem: str) -> tuple[str, int]:
    """(recording id, chapter number) from a GoPro file name; non-GoPro files are their own recording."""
    # old-style names first: GP010123 must continue GOPR0123, not start its own recording
    if m := _GOPRO_OLD_FIRST.match(stem):
        return f"GOPR{m.group(1)}", 0
    if m := _GOPRO_OLD_NEXT.match(stem):
        return f"GOPR{m.group(2)}", int(m.group(1))
    if m := _GOPRO_NEW.match(stem):
        return f"G{m.group(1).upper()}{m.group(3)}", int(m.group(2))
    return stem, 0


def find_files(folder: str | Path, exts: set[str], recursive: bool = True) -> list[Path]:
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in exts
                  and not p.name.startswith("._"))       # skip macOS resource forks on SD cards


def group_recordings(videos: list[Path]) -> list[Recording]:
    groups: dict[tuple, list[tuple[int, Path]]] = {}
    for v in videos:
        rec, chap = _gopro_key(v.stem)
        groups.setdefault((str(v.parent), rec), []).append((chap, v))
    recs = [Recording(key=rec, chapters=[p for _, p in sorted(items)]) for (_, rec), items in groups.items()]
    return recs


# =================================================================== video metadata
def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True).stdout


def find_exiftool() -> str | None:
    for cand in (os.environ.get("EXIFTOOL"), shutil.which("exiftool"), shutil.which("exiftool.exe")):
        if cand and Path(cand).exists():
            return cand
    return None


def video_create_time(path: Path) -> datetime | None:
    """Recording start as the camera's wall-clock time (naive datetime).
    GoPro writes camera-local time into the QuickTime CreateDate field; exiftool shows it raw,
    ffprobe shows the same raw value with a misleading 'Z' suffix. Both are read as-is."""
    et = find_exiftool()
    if et:
        try:
            meta = json.loads(_run([et, "-CreateDate", "-j", str(path)]))[0]
            if meta.get("CreateDate") and not meta["CreateDate"].startswith("0000"):
                return datetime.strptime(meta["CreateDate"][:19], "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass
    try:
        info = json.loads(_run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)]))
        ct = info.get("format", {}).get("tags", {}).get("creation_time")
        if ct:
            return datetime.strptime(ct[:19], "%Y-%m-%dT%H:%M:%S")
    except Exception:
        pass
    return None


def video_duration(path: Path) -> float:
    try:
        return float(_run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                           "-of", "default=nw=1:nk=1", str(path)]).strip())
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(path))
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return n / fps if fps else 0.0


def probe_recording(rec: Recording) -> Recording:
    rec.start_local = video_create_time(rec.chapters[0])
    rec.durations = [video_duration(c) for c in rec.chapters]
    return rec


# =================================================================== GPX
@dataclass
class Track:
    t: np.ndarray          # epoch seconds (UTC)
    lat: np.ndarray
    lon: np.ndarray
    depth: np.ndarray      # NaN where the echosounder gave no depth
    source: str

    def local_range(self, tz: str = LOCAL_TZ) -> tuple[datetime, datetime]:
        z = ZoneInfo(tz)
        return (datetime.fromtimestamp(self.t[0], z), datetime.fromtimestamp(self.t[-1], z))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def read_gpx(path: str | Path) -> Track:
    """All track points with a timestamp. GPX times are UTC. Depth is taken from any
    extension element named 'depth' (Garmin TrackPointExtension v1/v2, gpxx, ...)."""
    root = ET.parse(path).getroot()
    t, lat, lon, dep = [], [], [], []
    for pt in root.iter():
        if _local(pt.tag) != "trkpt":
            continue
        tm, d = None, np.nan
        for el in pt.iter():
            name = _local(el.tag)
            if name == "time" and el.text:
                tm = el.text.strip()
            elif name == "depth" and el.text:
                try:
                    d = float(el.text)
                except ValueError:
                    pass
        if tm is None:
            continue
        ts = datetime.fromisoformat(tm.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        t.append(ts.timestamp()); lat.append(float(pt.get("lat"))); lon.append(float(pt.get("lon"))); dep.append(d)
    if not t:
        raise ValueError(f"No timestamped track points in {path}")
    order = np.argsort(t)
    arr = lambda x: np.asarray(x, dtype=float)[order]
    t_, keep = arr(t), None
    keep = np.concatenate([[True], np.diff(t_) > 0])      # drop duplicate timestamps
    return Track(t_[keep], arr(lat)[keep], arr(lon)[keep], arr(dep)[keep], str(path))


def merge_tracks(tracks: list[Track]) -> Track:
    if len(tracks) == 1:
        return tracks[0]
    t = np.concatenate([x.t for x in tracks]); o = np.argsort(t)
    cat = lambda a: np.concatenate([getattr(x, a) for x in tracks])[o]
    t = t[o]; keep = np.concatenate([[True], np.diff(t) > 0])
    return Track(t[keep], cat("lat")[keep], cat("lon")[keep], cat("depth")[keep],
                 ", ".join(x.source for x in tracks))


def interpolate(track: Track, t_utc: np.ndarray, max_gap_s: float = 60.0):
    """Linear interpolation of lat/lon/depth at each time. Returns arrays plus a status per
    frame: 'ok' | 'before_track' | 'after_track' | 'gps_gap'."""
    i = np.searchsorted(track.t, t_utc, side="right")
    n = len(track.t)
    lat = np.full(len(t_utc), np.nan); lon = lat.copy(); dep = lat.copy()
    status = np.array(["ok"] * len(t_utc), dtype=object)
    for k, (ti, ii) in enumerate(zip(t_utc, i)):
        if ii == n and ti == track.t[-1]:
            ii = n - 1
        if ii == 0:
            status[k] = "before_track"; continue
        if ii >= n:
            status[k] = "after_track"; continue
        a, b = ii - 1, ii
        gap = track.t[b] - track.t[a]
        if gap > max_gap_s:
            status[k] = "gps_gap"; continue
        r = (ti - track.t[a]) / gap
        lat[k] = track.lat[a] + r * (track.lat[b] - track.lat[a])
        lon[k] = track.lon[a] + r * (track.lon[b] - track.lon[a])
        da, db = track.depth[a], track.depth[b]
        dep[k] = da + r * (db - da) if not (np.isnan(da) or np.isnan(db)) else (da if not np.isnan(da) else db)
    return lat, lon, dep, status


# =================================================================== frames
def _frame_name(ts_local: datetime, interval_s: float) -> str:
    if interval_s >= 1:
        return ts_local.strftime("frame_%Y-%m-%d_%H-%M-%S.jpg")          # same as the original scripts
    return ts_local.strftime("frame_%Y-%m-%d_%H-%M-%S_") + f"{ts_local.microsecond // 1000:03d}.jpg"


def extract_frames(rec: Recording, out_dir: Path, interval_s: float = 1.0, clock_offset_s: float = 0.0,
                   max_width: int | None = None, quality: int = 2, log=print) -> list[tuple[Path, datetime]]:
    """Extract one frame every `interval_s` seconds across all chapters of a recording, as one
    continuous timeline (equivalent to concatenating first, without writing a big video).
    Frame i gets time = CreateDate + i*interval + clock_offset_s (camera wall-clock)."""
    if rec.start_local is None:
        raise ValueError(f"{rec.key}: no CreateDate in video metadata — cannot timestamp frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"frames_{rec.key}_", dir=out_dir))
    try:
        vf = [f"fps=1/{interval_s}"]
        if max_width:
            vf.append(f"scale='min({max_width},iw)':-2")
        if len(rec.chapters) > 1:
            lst = tmp / "chapters.txt"
            lst.write_text("".join(f"file '{c.resolve().as_posix()}'\n" for c in rec.chapters))
            src = ["-f", "concat", "-safe", "0", "-i", str(lst)]
        else:
            src = ["-i", str(rec.chapters[0])]
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *src,
               "-vf", ",".join(vf), "-q:v", str(quality), str(tmp / "%07d.jpg")]
        log(f"  extracting {rec.key} ({len(rec.chapters)} chapter(s), {rec.duration/60:.1f} min) …")
        subprocess.run(cmd, check=True)
        out = []
        start = rec.start_local + timedelta(seconds=clock_offset_s)
        for f in sorted(tmp.glob("*.jpg")):
            i = int(f.stem) - 1
            ts = start + timedelta(seconds=i * interval_s)
            dest = out_dir / _frame_name(ts, interval_s)
            if dest.exists():                                   # two cameras, same second
                dest = dest.with_name(f"{dest.stem}_{rec.key}.jpg")
            shutil.move(str(f), dest)
            out.append((dest, ts))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_output_dir(videos_dir: str | Path, output_dir: str | Path):
    """The output must not be the videos folder or inside it: the video search is recursive, so
    anything written there (e.g. concatenated .mp4s) would be picked up as input on the next run."""
    v, o = Path(videos_dir).resolve(), Path(output_dir).resolve()
    if o == v or v in o.parents:
        raise ValueError(f"output_dir {output_dir} must not be the videos folder or inside it ({videos_dir})")


def concatenate_videos(rec: Recording, out_path: Path) -> Path:
    """Lossless concatenation of a recording's chapters (like scripts/concatenate_video.py)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("".join(f"file '{c.resolve().as_posix()}'\n" for c in rec.chapters))
        lst = f.name
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", lst, "-c", "copy", "-map", "0:v", "-map", "0:a?", str(out_path)], check=True)
    finally:
        os.remove(lst)
    return out_path


# =================================================================== EXIF
def _rational(x: float, den: int = 1_000_000):
    return (int(round(x * den)), den)


def _dms(deg: float):
    deg = abs(deg); d = int(deg); m_f = (deg - d) * 60; m = int(m_f); s = (m_f - m) * 60
    return ((d, 1), (m, 1), _rational(s, 10000))


def write_gps_exif(jpg: Path, lat: float, lon: float, ts_local: datetime, tz: str,
                   depth: float | None = None, extra_note: str = ""):
    import piexif
    try:
        exif = piexif.load(str(jpg))
    except Exception:
        exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    aware = ts_local.replace(tzinfo=ZoneInfo(tz))
    utc = aware.astimezone(timezone.utc)
    off = aware.strftime("%z"); off = f"{off[:3]}:{off[3:]}"
    exif["GPS"] = {
        piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
        piexif.GPSIFD.GPSLatitude: _dms(lat),
        piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
        piexif.GPSIFD.GPSLongitude: _dms(lon),
        piexif.GPSIFD.GPSDateStamp: utc.strftime("%Y:%m:%d").encode(),
        piexif.GPSIFD.GPSTimeStamp: ((utc.hour, 1), (utc.minute, 1), (utc.second, 1)),
        piexif.GPSIFD.GPSMapDatum: b"WGS-84",
    }
    stamp = ts_local.strftime("%Y:%m:%d %H:%M:%S").encode()
    exif["Exif"][piexif.ExifIFD.DateTimeOriginal] = stamp
    exif["Exif"][piexif.ExifIFD.DateTimeDigitized] = stamp
    exif["Exif"][piexif.ExifIFD.OffsetTimeOriginal] = off.encode()
    exif["0th"][piexif.ImageIFD.DateTime] = stamp
    desc = "OpERA benthic survey frame"
    if depth is not None and not np.isnan(depth):
        desc += f"; echosounder depth {depth:.2f} m"
    if extra_note:
        desc += f"; {extra_note}"
    exif["0th"][piexif.ImageIFD.ImageDescription] = desc.encode()
    piexif.insert(piexif.dump(exif), str(jpg))


def read_gps_exif(jpg: Path) -> tuple[float, float] | None:
    import piexif
    g = piexif.load(str(jpg)).get("GPS", {})
    if piexif.GPSIFD.GPSLatitude not in g:
        return None
    conv = lambda v: sum(n / d / f for (n, d), f in zip(v, (1, 60, 3600)))
    lat = conv(g[piexif.GPSIFD.GPSLatitude]) * (-1 if g[piexif.GPSIFD.GPSLatitudeRef] == b"S" else 1)
    lon = conv(g[piexif.GPSIFD.GPSLongitude]) * (-1 if g[piexif.GPSIFD.GPSLongitudeRef] == b"W" else 1)
    return lat, lon


# =================================================================== the whole thing
# Everything process_survey / classify_frames write. A re-run clears these first so frames from an
# earlier run (e.g. with the wrong time zone, hence different file names) don't linger.
MANAGED_OUTPUTS = ("geotagged", "untagged", "classified", "_frames", "frame_data.csv", "frames.geojson",
                   "qa_track_map.png", "qa_class_map.png", "report.json")


def clear_outputs(out: Path, log=print):
    removed = []
    for name in MANAGED_OUTPUTS:
        p = out / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
        else:
            continue
        removed.append(name)
    if removed:
        log(f"  cleared previous outputs in {out}: {', '.join(removed)}")


def check_overlap(recs: list[Recording], track: Track, tz: str, clock_offset_s: float = 0.0) -> list[dict]:
    """How much of each recording falls inside the GPX time range."""
    z = ZoneInfo(tz)
    g0, g1 = track.t[0], track.t[-1]
    out = []
    for r in recs:
        if r.start_local is None:
            out.append({"recording": r.key, "problem": "no CreateDate"}); continue
        v0 = r.start_local.replace(tzinfo=z).timestamp() + clock_offset_s
        v1 = v0 + r.duration
        inside = max(0.0, min(v1, g1) - max(v0, g0))
        pct = 100 * inside / r.duration if r.duration else 0
        note = "ok"
        if pct == 0:
            shift_h = round(((g0 + g1) / 2 - (v0 + v1) / 2) / 3600, 2)
            note = (f"NO OVERLAP with GPX — video centre is {-shift_h:+.2f} h from track centre. "
                    "Check the camera clock / time zone (tz='UTC' if the camera was set to UTC, or clock_offset_s for drift).")
        elif pct < 90:
            note = "partial overlap — frames outside the track will stay untagged"
        out.append({"recording": r.key, "pct_inside_gpx": round(pct, 1), "note": note})
    return out


def inspect(videos_dir: str, gpx: str | list[str] | None = None, tz: str = LOCAL_TZ,
            clock_offset_s: float = 0.0) -> dict:
    vids = find_files(videos_dir, VIDEO_EXT)
    recs = [probe_recording(r) for r in group_recordings(vids)]
    gpx_files = [Path(g) for g in ([gpx] if isinstance(gpx, str) else gpx)] if gpx else find_files(videos_dir, {".gpx"})
    rep = {"videos_dir": str(videos_dir), "n_video_files": len(vids),
           "recordings": [r.summary() for r in recs], "gpx_files": [str(g) for g in gpx_files]}
    if gpx_files:
        tr = merge_tracks([read_gpx(g) for g in gpx_files])
        a, b = tr.local_range(tz)
        rep["gpx"] = {"points": int(len(tr.t)), "start_local": a.isoformat(sep=" "), "end_local": b.isoformat(sep=" "),
                      "points_with_depth": int(np.sum(~np.isnan(tr.depth))),
                      "median_interval_s": float(np.median(np.diff(tr.t))) if len(tr.t) > 1 else None}
        rep["overlap"] = check_overlap(recs, tr, tz, clock_offset_s)
    else:
        rep["gpx"] = None
        rep["overlap"] = "no GPX file found — pass gpx=... or put the .gpx in the videos folder"
    return rep


def process_survey(videos_dir: str, output_dir: str, gpx: str | list[str] | None = None,
                   interval_s: float = 1.0, tz: str = LOCAL_TZ, clock_offset_s: float = 0.0,
                   max_gap_s: float = 60.0, max_width: int | None = None, keep_untagged: bool = True,
                   recordings: list[str] | None = None, model_path: str | None = None,
                   classify_conf: float = 0.0, log=print) -> dict:
    """Videos folder + GPX → output_dir/geotagged/*.jpg, frame_data.csv, frames.geojson, report.json.
    With model_path (a YOLO classification .pt), the geotagged frames are then classified: see
    opera_agent.classify.classify_frames."""
    check_output_dir(videos_dir, output_dir)
    vids = find_files(videos_dir, VIDEO_EXT)
    if not vids:
        raise FileNotFoundError(f"No .mp4/.mov videos under {videos_dir}")

    recs = [probe_recording(r) for r in group_recordings(vids)]
    if recordings:
        recs = [r for r in recs if r.key in recordings]
    gpx_files = [Path(g) for g in ([gpx] if isinstance(gpx, str) else gpx)] if gpx else find_files(videos_dir, {".gpx"})
    if not gpx_files:
        raise FileNotFoundError("No GPX file given or found next to the videos")
    track = merge_tracks([read_gpx(g) for g in gpx_files])
    log(f"GPX: {len(track.t)} points from {len(gpx_files)} file(s); {len(recs)} recording(s)")

    # inputs are valid: only now replace the previous run's outputs
    out = Path(output_dir)
    if out.exists():
        clear_outputs(out, log)
    tagged_dir, untag_dir, work = out / "geotagged", out / "untagged", out / "_frames"
    for d in (tagged_dir, work):
        d.mkdir(parents=True, exist_ok=True)

    z = ZoneInfo(tz)
    rows = []
    for rec in recs:
        frames = extract_frames(rec, work, interval_s, clock_offset_s, max_width, log=log)
        t_utc = np.array([ts.replace(tzinfo=z).timestamp() for _, ts in frames])
        lat, lon, dep, status = interpolate(track, t_utc, max_gap_s)
        n_ok = 0
        for (p, ts), la, lo, de, st in zip(frames, lat, lon, dep, status):
            if st == "ok":
                write_gps_exif(p, la, lo, ts, tz, de, extra_note=f"recording {rec.key}")
                dest = tagged_dir / p.name; n_ok += 1
            elif keep_untagged:
                untag_dir.mkdir(exist_ok=True); dest = untag_dir / p.name
            else:
                p.unlink(); dest = None
            if dest:
                shutil.move(str(p), dest)
            rows.append({"frame_filename": p.name, "recording": rec.key,
                         "frame_datetime": ts.replace(tzinfo=z).isoformat(),
                         "latitude": None if np.isnan(la) else round(float(la), 7),
                         "longitude": None if np.isnan(lo) else round(float(lo), 7),
                         "depth_m": None if np.isnan(de) else round(float(de), 2),
                         "status": st, "path": str(dest) if dest else ""})
        log(f"  {rec.key}: {len(frames)} frames, {n_ok} geotagged")
    shutil.rmtree(work, ignore_errors=True)

    # ---- outputs
    csv_path = out / "frame_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["frame_filename"])
        w.writeheader(); w.writerows(rows)
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["longitude"], r["latitude"]]},
         "properties": {k: v for k, v in r.items() if k not in ("latitude", "longitude", "path")}}
        for r in rows if r["status"] == "ok"]}
    (out / "frames.geojson").write_text(json.dumps(gj))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    report = {"output_dir": str(out), "geotagged_dir": str(tagged_dir),
              "frames_total": len(rows), "status_counts": counts,
              "recordings": [r.summary() for r in recs],
              "gpx_files": [str(g) for g in gpx_files],
              "settings": {"interval_s": interval_s, "tz": tz, "clock_offset_s": clock_offset_s,
                           "max_gap_s": max_gap_s, "max_width": max_width},
              "files": {"csv": str(csv_path), "geojson": str(out / "frames.geojson")}}
    try:
        report["files"]["qa_map"] = str(_qa_plot(track, rows, out / "qa_track_map.png"))
    except Exception as e:
        log(f"  (QA map skipped: {e})")
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    if model_path:
        # geotagging is done and saved; a classification problem (no ultralytics, bad weights) is reported,
        # not raised, so the geotagged result isn't lost. classify_frames can be re-run on its own.
        try:
            from .classify import classify_frames
            report["classification"] = classify_frames(out, model_path, conf=classify_conf, log=log)
        except Exception as e:
            log(f"  ! classification failed: {type(e).__name__}: {e}")
            report["classification"] = {"error": f"{type(e).__name__}: {e}"}
            (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _frame_rows_ok(out: Path) -> tuple[list[dict], list[dict]]:
    """All rows of output_dir/frame_data.csv, and the geotagged ones among them."""
    with open(Path(out) / "frame_data.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, [r for r in rows if r["status"] == "ok"]


def _qa_plot(track: Track, rows: list[dict], path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in rows if r["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(track.lon, track.lat, lw=0.8, color="0.6", label="GPX track")
    if ok:
        sc = ax.scatter([r["longitude"] for r in ok], [r["latitude"] for r in ok], s=6,
                        c=[r["depth_m"] if r["depth_m"] is not None else np.nan for r in ok], cmap="viridis_r")
        if any(r["depth_m"] is not None for r in ok):
            fig.colorbar(sc, ax=ax, shrink=0.7, label="echosounder depth (m)")
    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(track.lat))))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"{len(ok)} geotagged frames"); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path
