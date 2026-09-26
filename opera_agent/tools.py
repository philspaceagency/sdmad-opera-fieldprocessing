"""Tools the agent can call. Each returns a JSON-serialisable dict."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from . import pipeline as P

# ---------------------------------------------------------------- schemas (Gemini function-declaration format)
TOOLS = [
    {
        "name": "list_folder",
        "description": "List sub-folders of a directory with counts of video (.mp4/.mov) and .gpx files, "
                       "to locate survey data. Use when the user hasn't given exact paths.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "depth": {"type": "integer", "description": "levels to descend (default 2, max 4)"}},
            "required": ["path"]},
    },
    {
        "name": "inspect_survey",
        "description": "Read-only check BEFORE processing: groups GoPro chapters into recordings, reads each "
                       "recording's start time and duration, reads the GPX time range and depth coverage, "
                       "and reports how much of each recording overlaps the GPX track.",
        "parameters": {"type": "object", "properties": {
            "videos_dir": {"type": "string"},
            "gpx": {"type": "array", "items": {"type": "string"},
                    "description": "GPX file path(s). Omit to auto-find .gpx files under videos_dir."},
            "tz": {"type": "string", "description": "camera clock time zone (default Asia/Manila)"},
            "clock_offset_s": {"type": "number", "description": "seconds to add to camera time (default 0)"}},
            "required": ["videos_dir"]},
    },
    {
        "name": "process_survey",
        "description": "Run the full pipeline: extract frames from every GoPro recording (chapters joined in "
                       "order), timestamp them from the video CreateDate, interpolate position and echosounder "
                       "depth from the GPX, and write GPS EXIF. Produces output_dir/geotagged/*.jpg, "
                       "frame_data.csv, frames.geojson, qa_track_map.png and report.json.",
        "parameters": {"type": "object", "properties": {
            "videos_dir": {"type": "string"},
            "output_dir": {"type": "string"},
            "gpx": {"type": "array", "items": {"type": "string"}},
            "interval_s": {"type": "number", "description": "seconds between frames (default 1)"},
            "tz": {"type": "string"},
            "clock_offset_s": {"type": "number"},
            "max_gap_s": {"type": "number", "description": "don't interpolate across GPX gaps longer than this (default 60)"},
            "max_width": {"type": "integer", "description": "downscale frames to this width in px (default: full resolution)"},
            "recordings": {"type": "array", "items": {"type": "string"},
                           "description": "only these recording ids, e.g. ['GH0123']"}},
            "required": ["videos_dir", "output_dir"]},
    },
    {
        "name": "verify_geotags",
        "description": "Read back GPS EXIF from a sample of output images and compare with frame_data.csv.",
        "parameters": {"type": "object", "properties": {
            "output_dir": {"type": "string"}, "n": {"type": "integer"}}, "required": ["output_dir"]},
    },
    {
        "name": "concatenate_recordings",
        "description": "Optional: write one lossless .mp4 per recording (chapters joined in order). "
                       "Not needed for geotagging; only when the user wants the merged videos.",
        "parameters": {"type": "object", "properties": {
            "videos_dir": {"type": "string"}, "output_dir": {"type": "string"}},
            "required": ["videos_dir", "output_dir"]},
    },
    {
        "name": "classify_frames",
        "description": "Optional, only when asked: classify geotagged frames with the YOLO benthic model "
                       "(corals, macroalgae, rubble, sand, seagrass) and add class + confidence to "
                       "frame_data.csv and frames.geojson. Needs ultralytics and ideally a GPU.",
        "parameters": {"type": "object", "properties": {
            "output_dir": {"type": "string", "description": "a folder produced by process_survey"},
            "model_path": {"type": "string", "description": "path to yolo-benthic-cls.pt"},
            "conf": {"type": "number"}}, "required": ["output_dir"]},
    },
    {
        "name": "search_workflow_docs",
        "description": "Search the sdmad-opera-fieldprocessing repository (README, scripts, notebooks, workflow "
                       "slides) for how the original workflow does something. Use for questions about the "
                       "method, not to run it.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]},
    },
]


# ---------------------------------------------------------------- implementations
class ToolRunner:
    def __init__(self, rag=None, default_model_path: str | None = None, log=print):
        self.rag = rag                       # opera_rag.OperaRAG or None
        self.default_model_path = default_model_path
        self.log = log

    def __call__(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # -----------------------------------------------------------
    def t_list_folder(self, path, depth=2):
        root = Path(path)
        if not root.is_dir():
            return {"error": f"not a directory: {path}"}
        depth = max(1, min(int(depth), 4))
        rows = []
        for d, dirs, files in os.walk(root):
            rel = Path(d).relative_to(root)
            if len(rel.parts) > depth:
                dirs[:] = []; continue
            dirs[:] = sorted(x for x in dirs if not x.startswith("."))
            v = sum(Path(f).suffix.lower() in P.VIDEO_EXT for f in files)
            g = sum(Path(f).suffix.lower() == ".gpx" for f in files)
            j = sum(Path(f).suffix.lower() in (".jpg", ".jpeg") for f in files)
            rows.append({"dir": str(Path(d)), "videos": v, "gpx": g, "jpg": j})
            if len(rows) >= 200:
                break
        return {"folders": rows}

    def t_inspect_survey(self, videos_dir, gpx=None, tz=P.LOCAL_TZ, clock_offset_s=0.0):
        return P.inspect(videos_dir, gpx, tz, clock_offset_s)

    def t_process_survey(self, videos_dir, output_dir, gpx=None, interval_s=1.0, tz=P.LOCAL_TZ,
                         clock_offset_s=0.0, max_gap_s=60.0, max_width=None, recordings=None):
        if Path(output_dir).resolve() == Path(videos_dir).resolve():
            return {"error": "output_dir must differ from videos_dir"}
        return P.process_survey(videos_dir, output_dir, gpx, interval_s, tz, clock_offset_s,
                                max_gap_s, max_width, recordings=recordings, log=self.log)

    def t_verify_geotags(self, output_dir, n=5):
        out = Path(output_dir)
        rows = list(csv.DictReader(open(out / "frame_data.csv")))
        ok = [r for r in rows if r["status"] == "ok"]
        if not ok:
            return {"error": "no geotagged frames in frame_data.csv"}
        step = max(1, len(ok) // n)
        checks = []
        for r in ok[::step][:n]:
            got = P.read_gps_exif(Path(r["path"]))
            exp = (float(r["latitude"]), float(r["longitude"]))
            err_m = None if got is None else round(max(abs(got[0] - exp[0]), abs(got[1] - exp[1])) * 111_000, 2)
            checks.append({"file": r["frame_filename"], "exif": got, "csv": exp, "max_error_m": err_m})
        return {"checked": len(checks), "all_have_gps": all(c["exif"] for c in checks), "samples": checks}

    def t_concatenate_recordings(self, videos_dir, output_dir):
        recs = P.group_recordings(P.find_files(videos_dir, P.VIDEO_EXT))
        made = []
        for r in recs:
            if len(r.chapters) == 1:
                continue
            made.append(str(P.concatenate_videos(r, Path(output_dir) / f"{r.key}_concat.mp4")))
        return {"concatenated": made, "single_chapter_recordings_skipped": [r.key for r in recs if len(r.chapters) == 1]}

    def t_classify_frames(self, output_dir, model_path=None, conf=0.0):
        from ultralytics import YOLO
        model_path = model_path or self.default_model_path
        if not model_path or not Path(model_path).exists():
            return {"error": "model_path not found; pass the path to yolo-benthic-cls.pt"}
        out = Path(output_dir)
        rows = list(csv.DictReader(open(out / "frame_data.csv")))
        ok = [r for r in rows if r["status"] == "ok"]
        model = YOLO(model_path)
        by_name = {}
        for r in ok:
            res = model.predict(r["path"], imgsz=640, verbose=False)[0]
            cls, p = res.names[int(res.probs.top1)], float(res.probs.top1conf)
            by_name[r["frame_filename"]] = (cls, p) if p >= conf else ("uncertain", p)
        for r in rows:
            r["benthic_class"], r["class_conf"] = by_name.get(r["frame_filename"], ("", ""))
            if r["class_conf"] != "":
                r["class_conf"] = round(r["class_conf"], 3)
        with open(out / "frame_data.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        gj = json.loads((out / "frames.geojson").read_text())
        for feat in gj["features"]:
            c = by_name.get(feat["properties"]["frame_filename"])
            if c:
                feat["properties"]["benthic_class"], feat["properties"]["class_conf"] = c[0], round(c[1], 3)
        (out / "frames.geojson").write_text(json.dumps(gj))
        counts = {}
        for c, _ in by_name.values():
            counts[c] = counts.get(c, 0) + 1
        return {"classified": len(by_name), "class_counts": counts}

    def t_search_workflow_docs(self, query, k=5):
        if self.rag is None:
            return {"error": "workflow docs index not loaded"}
        hits = self.rag.retrieve(query, k=k)
        return {"results": [{"source": c.cite, "name": c.name, "text": c.text[:2500]} for c, _, _ in hits]}
