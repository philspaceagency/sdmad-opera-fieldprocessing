"""Tools the agent can call. Each returns a JSON-serialisable dict."""
from __future__ import annotations

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
                           "description": "only these recording ids, e.g. ['GH0123']"},
            "classify": {"type": "boolean",
                         "description": "after geotagging, classify the frames with the YOLO benthic model "
                                        "(default true when a model is configured)"}},
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
        "description": "Classify the geotagged frames of an existing process_survey output with the YOLO benthic "
                       "model (corals, macroalgae, rubble, sand, seagrass). Keeps GPS/time/depth EXIF, adds the "
                       "class to EXIF, sorts frames into classified/<class>/ and adds class + probabilities to "
                       "frame_data.csv and frames.geojson. Use to (re)classify without re-extracting frames.",
        "parameters": {"type": "object", "properties": {
            "output_dir": {"type": "string", "description": "a folder produced by process_survey"},
            "model_path": {"type": "string", "description": "YOLO classification weights (.pt); default: configured model"},
            "conf": {"type": "number", "description": "below this top-1 confidence the class is 'uncertain' (default 0)"},
            "organize": {"type": "string", "enum": ["copy", "move", "none"],
                         "description": "copy (default) or move frames into classified/<class>/, or none"}},
            "required": ["output_dir"]},
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
                         clock_offset_s=0.0, max_gap_s=60.0, max_width=None, recordings=None, classify=None):
        model = self.default_model_path if classify is not False else None
        if classify and not model:
            return {"error": "classify=true but no YOLO model is configured (--yolo-model / OPERA_YOLO_MODEL)"}
        return P.process_survey(videos_dir, output_dir, gpx, interval_s, tz, clock_offset_s,
                                max_gap_s, max_width, recordings=recordings, model_path=model, log=self.log)

    def t_verify_geotags(self, output_dir, n=5):
        _, ok = P._frame_rows_ok(Path(output_dir))
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
        P.check_output_dir(videos_dir, output_dir)
        recs = P.group_recordings(P.find_files(videos_dir, P.VIDEO_EXT))
        made = []
        for r in recs:
            if len(r.chapters) == 1:
                continue
            made.append(str(P.concatenate_videos(r, Path(output_dir) / f"{r.key}_concat.mp4")))
        return {"concatenated": made, "single_chapter_recordings_skipped": [r.key for r in recs if len(r.chapters) == 1]}

    def t_classify_frames(self, output_dir, model_path=None, conf=0.0, organize="copy"):
        from .classify import classify_frames
        model_path = model_path or self.default_model_path
        if not model_path:
            return {"error": "no YOLO model configured; pass model_path (the classification .pt)"}
        return classify_frames(output_dir, model_path, conf=conf, organize=organize, log=self.log)

    def t_search_workflow_docs(self, query, k=5):
        if self.rag is None:
            return {"error": "workflow docs index not loaded"}
        hits = self.rag.retrieve(query, k=k)
        return {"results": [{"source": c.cite, "name": c.name, "text": c.text[:2500]} for c, _, _ in hits]}
