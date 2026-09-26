# OpERA field-processing agent

Ask it in plain language, for example:

> "Process the GoPro videos from the 13MAY2025 survey and give me the geotagged images"

and it produces:

```
13MAY2025_processed/
├── geotagged/            frame_2025-05-13_09-00-00.jpg …  (GPS + time + depth in EXIF)
├── untagged/             frames outside the GPX track or across GPS gaps
├── frame_data.csv        filename, recording, time, lat, lon, depth_m, status
├── frames.geojson        points, ready for QGIS / GEE
├── qa_track_map.png      GPX track with the frame positions coloured by depth
└── report.json           counts, settings, time ranges
```

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
4. **`verify_geotags`** reads the EXIF back from a sample of the images and compares it with the CSV.

Optional tools:
- **`classify_frames`** adds the YOLO benthic class to the CSV and GeoJSON.
- **`concatenate_recordings`** writes the merged videos.
- **`search_workflow_docs`** answers "how does the original do X" from the sdmad-opera-fieldprocessing repo (the RAG part).

## Run it

**Colab:** open `OpERA_FieldAgent_Colab.ipynb`.

**Command line:**

```bash
pip install -r requirements.txt          # plus ffmpeg on the PATH
export GEMINI_API_KEY=...
python -m opera_agent --data-root /data/surveys "process the 13MAY2025 survey"
python -m opera_agent --chat --data-root /data/surveys

# same pipeline without the LLM
python -m opera_agent inspect /data/surveys/13MAY2025
python -m opera_agent process /data/surveys/13MAY2025 /data/surveys/13MAY2025_processed --interval 1
```

**Settings:**

| Option | Default | What it does |
|---|---|---|
| `--interval` | 1 s | Time between extracted frames |
| `--tz` | Asia/Manila | Time zone the camera clock was set to |
| `--clock-offset` | 0 s | Corrects camera clock drift, in seconds |
| `--max-gap` | 60 s | Longest GPX gap it will interpolate across |
| `--max-width` | full resolution | Downscales frames to this width |

## Differences from the original scripts

| Original | Here |
|---|---|
| Windows `exiftool.exe`, `D:/` output path | Works on Colab, Linux, Mac and Windows. Uses ffprobe (or exiftool if installed) to read times and piexif to write EXIF. |
| `sorted(glob("*.MP4"))` interleaves chapters of different recordings and misses `.mp4` on Linux | Chapters are grouped per recording and ordered by chapter number. |
| One concatenated video, timed from the first clip's CreateDate | Each recording is timed from its own CreateDate. With several recordings, the gaps between them no longer shift the frame times. |
| GPX points without depth are dropped | Every point is used for position; depth is kept where present. |
| Frames outside the track are skipped silently | Frames outside the track are moved to `untagged/` with a reason (`before_track`, `after_track`, `gps_gap`). |
| No time check | The overlap check catches camera-clock and time-zone mistakes before processing. |
