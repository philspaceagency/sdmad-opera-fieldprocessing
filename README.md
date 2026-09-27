# OpERA field-processing agent

Ask it in plain language, for example:

> "Process the GoPro videos from the 13MAY2025 survey and give me the geotagged images"

and it produces:

```
13MAY2025_processed/
├── geotagged/            frame_2025-05-13_09-00-00.jpg …  (GPS + time + depth + benthic class in EXIF)
├── classified/           the same frames sorted by class: corals/ macroalgae/ rubble/ sand/ seagrass/ (uncertain/)
├── untagged/             frames outside the GPX track or across GPS gaps
├── frame_data.csv        filename, recording, time, lat, lon, depth_m, status, benthic_class, class_conf, prob_<class>…
├── frames.geojson        points with depth and class, ready for QGIS / GEE
├── qa_track_map.png      GPX track with the frame positions coloured by depth
├── qa_class_map.png      frame positions coloured by benthic class
└── report.json           counts, settings, time ranges, class counts
```

`classified/` and the class columns appear when a YOLO model is configured (`--yolo-model` or `OPERA_YOLO_MODEL`).

## How a request runs

1. **`list_folder`** finds the survey folder under your data root (only when you didn't give a path).
2. **`inspect_survey`** runs read-only checks:
   - groups GoPro chapters into recordings (`GH01xxxx`, `GH02xxxx` … → recording `xxxx`);
   - reads each recording's start time and duration, and the GPX time range;
   - checks how much of each recording overlaps the GPX track.

   If a recording has no overlap, the agent stops and asks. A camera set to UTC shows up as a ~8 h offset.
3. **`process_survey`** writes the outputs:
   - extracts one frame per second (configurable) across all chapters as one continuous timeline;
   - names frames from the video CreateDate;
   - interpolates position and depth from the GPX and writes EXIF (GPS, DateTimeOriginal with `+08:00`, depth in ImageDescription).
   - with a YOLO model configured, classifies every geotagged frame (see below).
4. **`verify_geotags`** reads the EXIF back from a sample of the images and compares it with the CSV.

Re-running into the same output folder first removes the previous run's outputs (only the files listed above),
so a corrected re-run never mixes with frames from the wrong one. The output folder may not be inside the
videos folder.

Optional tools:
- **`classify_frames`** re-classifies an existing output folder (another model or threshold) without re-extracting frames.
- **`concatenate_recordings`** writes the merged videos.
- **`search_workflow_docs`** answers "how does the original do X" from the sdmad-opera-fieldprocessing repo (the RAG part).

## Benthic classification

After geotagging, `opera_agent/classify.py` runs the YOLO11 classification model trained in
`notebooks/YOLO_classification.ipynb` on each geotagged frame:

- the frame keeps all its EXIF (GPS, DateTimeOriginal, depth). The class is added to ImageDescription by
  rewriting only the EXIF block, so the image is not re-encoded;
- the frame is copied to `classified/<class>/` (`--organize move` saves disk space, `none` only tags);
- frames below `--conf` are labelled `uncertain`;
- `frame_data.csv` and `frames.geojson` get `benthic_class`, `class_conf` and one `prob_<class>` column per class.

Put the weights in `models/` (see `models/README.md`).

## Run it

**Colab:** open `notebooks/OpERA_FieldAgent_Colab.ipynb`.

**Command line:**

```bash
pip install -r requirements.txt          # plus ffmpeg on the PATH
pip install -r requirements-yolo.txt     # only for benthic classification (adds ultralytics + PyTorch)
export GEMINI_API_KEY=...
export OPENROUTER_API_KEY=...             # optional: Qwen fallback when Gemini is busy
python -m opera_agent --data-root /data/surveys "process the 13MAY2025 survey"
python -m opera_agent --chat --data-root /data/surveys

# same pipeline without the LLM
python -m opera_agent inspect /data/surveys/13MAY2025
python -m opera_agent process /data/surveys/13MAY2025 /data/surveys/13MAY2025_processed --interval 1 \
    --yolo-model models/yolo11l-benthic-cls.pt
python -m opera_agent classify /data/surveys/13MAY2025_processed --yolo-model models/yolo11l-benthic-cls.pt --conf 0.5

# agent with classification
python -m opera_agent --yolo-model models/yolo11l-benthic-cls.pt --data-root /data/surveys "process the 13MAY2025 survey"
```

**Settings:**

| Option | Default | What it does |
|---|---|---|
| `--interval` | 1 s | Time between extracted frames |
| `--tz` | Asia/Manila | Time zone the camera clock was set to |
| `--clock-offset` | 0 s | Corrects camera clock drift, in seconds |
| `--max-gap` | 60 s | Longest GPX gap it will interpolate across |
| `--max-width` | full resolution | Downscales frames to this width |
| `--yolo-model` | `$OPERA_YOLO_MODEL` | YOLO classification weights; classification is skipped without one |
| `--conf` | 0 | Frames below this top-1 confidence are labelled `uncertain` |

## Repository layout

```
opera_agent/        pipeline.py (frames + geotagging), classify.py (YOLO), tools.py + agent.py (Gemini agent), CLI
opera_rag/          search over the fieldprocessing repository (used by search_workflow_docs)
notebooks/          OpERA_FieldAgent_Colab.ipynb (run the agent), YOLO_classification.ipynb (train the model)
models/             YOLO weights go here (git-ignored)
tests/              pytest; runs without ffmpeg, a GPU or ultralytics
```

Run the tests with `pip install pytest pillow && pytest`.

## Language model: Gemini with a Qwen fallback

The agent uses Gemini 3.5 Flash by default. When a model keeps returning rate-limit, overload or network
errors (after 2 retries, 2 s then 4 s apart), or is not available for your key (404, e.g. a retired model), the
same conversation continues on the next model, including the tool results so far:

`gemini-3.5-flash` → `gemini-3.5-flash-lite` → Qwen via OpenRouter (only when `OPENROUTER_API_KEY` is set)

The next request tries the first model again. To use Gemini 3.1 Pro: `--model gemini-3.1-pro`
(or `GEMINI_MODEL=gemini-3.1-pro`). To see which ids your key can use: `client.models.list()`.

| Option | Default | What it does |
|---|---|---|
| `--provider` | `gemini` | `openrouter` uses Qwen only (needs just `OPENROUTER_API_KEY`) |
| `--model` | `gemini-3.5-flash` | Primary model id (`GEMINI_MODEL` also works) |
| `--gemini-fallback` | `gemini-3.5-flash-lite` | Gemini model(s) tried before Qwen; repeatable, `none` to skip |
| `--fallback` | `auto` | `auto`: Qwen when `OPENROUTER_API_KEY` is set; `openrouter`: always; `none`: off |
| `--fallback-model` | `qwen/qwen3-235b-a22b-2507` | Any OpenRouter model with tool calling (`OPENROUTER_MODEL` also works) |

Keys are read from the environment (or Colab secrets in the notebook), never from files in the repo.

## Differences from the original scripts

| Original | Here |
|---|---|
| Windows `exiftool.exe`, `D:/` output path | Works on Colab, Linux, Mac and Windows. Uses ffprobe (or exiftool if installed) to read times and piexif to write EXIF. |
| `sorted(glob("*.MP4"))` interleaves chapters of different recordings and misses `.mp4` on Linux | Chapters are grouped per recording and ordered by chapter number. |
| One concatenated video, timed from the first clip's CreateDate | Each recording is timed from its own CreateDate. With several recordings, the gaps between them no longer shift the frame times. |
| GPX points without depth are dropped | Every point is used for position; depth is kept where present. |
| Frames outside the track are skipped silently | Frames outside the track are moved to `untagged/` with a reason (`before_track`, `after_track`, `gps_gap`). |
| No time check | The overlap check catches camera-clock and time-zone mistakes before processing. |
