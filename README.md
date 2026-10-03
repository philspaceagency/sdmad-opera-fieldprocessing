# OpERA field-processing agent

Ask it in plain language, for example:

> "Process the GoPro videos from the 13MAY2025 survey and give me the geotagged images"

and it produces:

```
13MAY2025_processed/
├── geotagged/            frame_2025-05-13_09-00-00.jpg …  (GPS + time + depth + benthic class in EXIF)
├── classified/           per-class folders: seagrass/seagrass_frame_….jpg … (class probabilities drawn on each image)
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

- the geotagged frame keeps all its EXIF (GPS, DateTimeOriginal, depth). The class and the probability of every
  class are added to ImageDescription by rewriting only the EXIF block, so the image is not re-encoded;
- a classified copy is written to `classified/<class>/<class>_<frame>.jpg`, e.g.
  `classified/seagrass/seagrass_frame_2025-05-13_09-00-00.jpg`, with a panel showing every class's probability
  drawn on the image and the same EXIF (`--no-annotate` skips the panel; `--organize move` removes the
  geotagged frame afterwards to save disk space, `none` only tags);
- predictions are averaged over the frame and its mirror image (test-time augmentation, `--tta hflip`, the
  default; `flips` adds vertical flips for models trained with them, `none` turns it off);
- `--smooth 3` averages each frame's probabilities with the frames within ±1.5 s of the same recording. The bottom
  changes slowly along a transect, so this removes single-frame flips (a fish, a blurred frame). The unsmoothed
  class is kept in `raw_class`. Off by default;
- frames below `--conf` are labelled `uncertain`; `class_margin` (top-1 minus top-2 probability) flags frames where
  the model hesitated between two classes (below ~0.2: worth a look);
- `frame_data.csv` and `frames.geojson` get `benthic_class`, `class_conf` and one `prob_<class>` column per class.

Put the weights in `models/` (see `models/README.md`).

**Training** (`notebooks/YOLO_classification.ipynb`) checks the dataset for near-identical frames that sit in
different splits (they make the test score optimistic: the first model scored 100 %), trains with augmentation
suited to top-down underwater frames (vertical + horizontal flips, colour/brightness, randaugment, erasing,
dropout), reports per-class precision / recall / F1 with a confusion matrix and the misclassified images, picks the
best `--tta`, and saves `best.pt` with a JSON model card. The helpers are in `opera_agent/yolo_train.py`.

To classify any folder of images (e.g. frames geotagged earlier), use `notebooks/YOLO_inference.ipynb` or
`python -m opera_agent classify-images <input_dir> <output_dir> --yolo-model …`. The output has the same
class-named, annotated images and keeps the input's sub-folders: `input/dive1/frame_x.jpg` →
`output/dive1/seagrass/seagrass_frame_x.jpg`, plus `summary.csv` and `classification_results.json`.

## Run it

**Colab:** open `notebooks/OpERA_FieldAgent_Colab.ipynb`.

**Command line:**

```bash
pip install -r requirements.txt          # plus ffmpeg on the PATH
pip install -r requirements-yolo.txt     # only for benthic classification (adds ultralytics + PyTorch)
export GEMINI_API_KEY=...                 # or put the keys in a file, see "API keys" below
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
| `--tta` | `hflip` | Test-time augmentation: `none`, `hflip`, `flips` |
| `--smooth` | 0 (off) | Average class probabilities over this many seconds of neighbouring frames |
| `--keys-file` | see "API keys" | Text file with API keys and settings |

## Repository layout

```
opera_agent/        pipeline.py (frames + geotagging), classify.py (YOLO), yolo_train.py (training checks),
                    llm.py + tools.py + agent.py (Gemini / Qwen agent), keys.py (key file), CLI
opera_rag/          search over the fieldprocessing repository (used by search_workflow_docs)
notebooks/          OpERA_FieldAgent_Colab.ipynb (run the agent), YOLO_classification.ipynb (train the model),
                    YOLO_inference.ipynb (classify a folder of images)
models/             YOLO weights go here (git-ignored)
hpc/                example SLURM job; environment.yml: conda env with ffmpeg
api_keys.example.txt  template for the key file (copy to api_keys.txt, which is git-ignored)
tests/              pytest; runs without ffmpeg, a GPU or ultralytics
```

Run the tests with `pip install pytest pillow && pytest`.

## API keys (local and HPC)

Colab reads the keys from its secrets. Elsewhere, set environment variables, or keep them in a text file:

```bash
cp api_keys.example.txt api_keys.txt      # git-ignored; never commit real keys
chmod 600 api_keys.txt                    # only you can read it (a warning is printed otherwise)
```

```
GEMINI API KEY: AIza...
OPENROUTER API KEY: sk-or-...
OPERA_YOLO_MODEL=/path/to/yolo11l-benthic-cls.pt     # optional settings work too
```

Both `LABEL: value` and `NAME=value` lines work. The file is found at, first match wins: `--keys-file PATH`
(any command), `$OPERA_KEYS_FILE`, `./api_keys.txt`, `~/.config/opera/api_keys.txt`. Environment variables
override the file. In Python: `OperaAgent(keys_file="...")`. Only the names of the settings loaded are printed,
never the values.

## Running on an HPC cluster

```bash
conda env create -f environment.yml && conda activate opera     # includes ffmpeg
mkdir -p ~/.config/opera && cp api_keys.example.txt ~/.config/opera/api_keys.txt   # then edit + chmod 600
mkdir -p logs && sbatch hpc/process_survey.slurm /data/surveys/13MAY2025 /data/surveys/13MAY2025_processed
```

- The direct pipeline (`process`, `classify`, `classify-images`) needs no internet, so it runs on compute nodes.
  Put the YOLO weights on the cluster and set `OPERA_YOLO_MODEL` in the key file (or pass `--yolo-model`).
- The natural-language agent calls Gemini / OpenRouter, so it needs outbound internet: run it on a login or
  interactive node if compute nodes are offline.
- For a GPU, install the PyTorch build that matches the cluster's CUDA driver
  (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu124`), then check with
  `python -c "import torch; print(torch.cuda.is_available())"`.

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
