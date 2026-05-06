# OpERA Field Processing

This repository contains tools and notebooks for processing underwater benthic survey imagery using YOLO (You Only Look Once) deep learning models. The workflow extracts frames from videos, geotags them with GPS data, and classifies benthic features (corals, seagrass, sand, etc.).

## Project Structure

```
FieldProcessing/
├── notebooks/
│   ├── YOLO_classification.ipynb    # Train YOLO classification model
│   ├── YOLO_inference.ipynb           # Run inference with metadata preservation
│   └── rename_photos.ipynb            # Utility for batch renaming photos
├── scripts/
│   ├── extract_photo.py               # Extract frames from GoPro videos with timestamps
│   ├── geotag_gpx_frames.py           # Geotag images using GPX track data
│   └── concatenate_video.py           # Concatenate multiple MP4 files using ffmpeg
├── models/
│   └── yolo-benthic-cls.pt           # Pre-trained benthic classification model
└── scripts/exiftool/                  # ExifTool for metadata extraction
```

## Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- FFmpeg installed and in PATH

### Required Python Packages
```bash
pip install torch torchvision
pip install ultralytics roboflow rasterio geopandas piexif
pip install pandas numpy opencv-python pillow tqdm
```

## Complete Workflow

### Step 1: Extract Frames from Video

Extract timestamped frames from GoPro video files. The script uses ExifTool to read the video creation date and names each frame accordingly.

```bash
python scripts/extract_photo.py \
    -s /path/to/original_video.mp4 \
    -o ./OutputFrames \
    -p /path/to/processed_video.mp4
```

**Arguments:**
| Flag | Description |
|------|-------------|
| `-s, --source_video` | Original video file for metadata extraction |
| `-o, --output_dir` | Directory to save extracted frames |
| `-p, --processed_video` | Actual video file to process (can be different from source) |

**Output:** Frames saved as `frame_YYYY-MM-DD_HH-MM-SS.jpg`

### Step 2: Geotag Frames with GPX Data

Associate GPS coordinates with each frame using a GPX track file.

```bash
python scripts/geotag_gpx_frames.py
```

**Configuration (edit script):**
```python
gpx_file_path = './GPX/13MAY2025.GPX'  # Path to your GPX file
frame_dir = './Output'                  # Directory containing extracted frames
```

The script:
1. Reads GPX track with timestamps and depth
2. Converts UTC to PST (Asia/Manila timezone)
3. Interpolates GPS coordinates for each frame timestamp
4. Writes GPS data to image EXIF using ExifTool

**Output:** CSV file with frame filenames, coordinates, and timestamps

### Step 3A: Train Classification Model (Optional)

If you need to retrain the model on new data:

```bash
jupyter notebook notebooks/YOLO_classification.ipynb
```

**Training Parameters:**
- Model: YOLO11-large classification (`yolo11l-cls.pt`)
- Epochs: 100
- Batch Size: 16
- Image Size: 640x640
- Early Stopping Patience: 20

**Dataset Structure:**
```
dataset/
├── train/
│   ├── corals/
│   ├── macroalgae/
│   ├── rubble/
│   ├── sand/
│   └── seagrass/
├── val/
│   └── ... (same structure)
└── test/
    └── ... (same structure)
```

### Step 3B: Run Inference on Images

Classify benthic features in images while preserving GPS metadata.

```python
from ultralytics import YOLO

model = YOLO('./models/yolo-benthic-cls.pt')

results = model.predict(
    source='./InputImages',
    imgsz=640,
    conf=0.25,
    save=True,
    project='./Output',
    name='predictions'
)
```

**Inference Function (from notebook):**
```python
results = classify_benthic(
    input_dir='./InputImages',
    output_dir='./Output',
    img_size=640,
    conf=0.20
)
```

**Output Files:**
- Classified images renamed by predicted class (e.g., `seagrass.jpg`)
- `classification_results_with_gps.json` - Full results with GPS coordinates
- `summary.csv` - CSV summary with predictions and coordinates

## Model Classes

The benthic classification model predicts 5 classes:

| Class | Description |
|-------|-------------|
| corals | Hard and soft coral formations |
| macroalgae | Large seaweeds and algae |
| rubble | Broken coral fragments and debris |
| sand | Sandy substrate |
| seagrass | Seagrass beds |

## Utility Scripts

### Rename Photos
Batch rename photos for dataset organization:

```python
folder_path = r'./dataset/val/seagrass'
# Renames to: seagrass.1.jpg, seagrass.2.jpg, etc.
```

### Concatenate Videos
Merge multiple MP4 files without re-encoding:

```bash
python scripts/concatenate_video.py -f ./VideoClips -o output.mp4
```

## ExifTool Setup

The bundled ExifTool is located at:
```
scripts/exiftool/exiftool-12.92_64/exiftool.exe
```

Ensure this path is correct in the scripts before running.

## Output Examples

### Training Output
```
📊 Dataset Statistics:
TRAIN: 1146 images (corals: 240, macroalgae: 171, rubble: 162, sand: 246, seagrass: 327)
VAL:   112 images

📈 Validation Metrics:
  Top-1 Accuracy: 1.0000 (100.00%)
  Top-5 Accuracy: 1.0000 (100.00%)
```

### Inference Output
```
✅ Processed 28 images in 58.53s (2.09s per image)
📍 GPS data found in 28/28 images
📋 Metadata preserved in 28/28 output images
```

## Troubleshooting

**GPU Out of Memory:**
- Reduce batch size: `BATCH_SIZE = 8`
- Use smaller model: `yolo11s-cls.pt` instead of `yolo11l-cls.pt`

**EXIF/GPS Not Found:**
- Verify ExifTool path is correct
- Check input images have EXIF data
- Ensure GPX timestamps match frame timestamps

**Import Errors:**
- Install missing packages from requirements
- Use Python 3.8+ virtual environment