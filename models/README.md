# models/

Put the YOLO benthic classification weights here, e.g. `models/yolo11l-benthic-cls.pt`.
`.pt` files are git-ignored: they are large, so keep the master copy on Drive.

The weights come from `notebooks/YOLO_classification.ipynb` (`<run>/weights/best.pt`,
YOLO11-large classification, classes: corals, macroalgae, rubble, sand, seagrass).

Classification needs `pip install -r requirements-yolo.txt` (ultralytics + PyTorch). Use the weights with:

```bash
python -m opera_agent process <videos_dir> <output_dir> --yolo-model models/yolo11l-benthic-cls.pt
python -m opera_agent classify <output_dir> --yolo-model models/yolo11l-benthic-cls.pt   # re-classify only
export OPERA_YOLO_MODEL=models/yolo11l-benthic-cls.pt                                      # or set it once
```
