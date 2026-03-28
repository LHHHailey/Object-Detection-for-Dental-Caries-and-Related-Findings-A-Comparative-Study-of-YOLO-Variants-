# Dental caries detection YOLOv5nu/ YOLOv8m/ YOLO11m/ YOLO12m
Findings:A Comparative Study of YOLO Variants (20-Epoch Benchmark)

Object detection for caries, cavities, cracks, and teeth in intraoral-style images: data conversion, training, inference, and a minimal web UI. Based on [AndreyGermanov/yolov8_caries_detector](https://github.com/AndreyGermanov/yolov8_caries_detector).

## What this repository contains (and does not)

To keep the repo small, it **usually only includes `yolo_dataset/data.yaml`** (keep this **relative path**; do **not** put `data.yaml` at the repository root). It does **not** include `yolo_dataset/**/images` or `labels`, the raw **`dataset/`** tree, or large **`.tar`** archives.

**You cannot train immediately after clone.** Prepare the raw DatasetNinja/Supervisely-style data locally, then run the conversion script to build a full `yolo_dataset/`.

**Typical workflow:**

1. Obtain DentalAI-style data (see **Data source** below). You should have `dataset/train|valid|test/{img,ann}` and `dataset/meta.json` under the project root.  
2. Install dependencies, then run: `python run_convert.py`  
3. The script **deletes and recreates** `yolo_dataset/`, writing `images/`, `labels/`, and a new `data.yaml` (same class names as the checked-in `data.yaml`; paths are written by the script).  
4. Run `run_yolo_compare.ps1` or call Ultralytics training yourself.

If you use `run_all.ps1` with a `.tar`, it performs extract → `dataset/` → `run_convert.py` automatically.

## Repository layout

| Path | Description |
|------|-------------|
| `dataset/` | Raw data (`train` / `valid` / `test` with `img` + `ann`, plus `meta.json`) — **usually not in the minimal repo; provide locally** |
| `yolo_dataset/` | YOLO layout and `data.yaml` produced by `run_convert.py`; **minimal repo may only ship `data.yaml` as reference** |
| `run_convert.py` | Supervisely-style JSON → YOLO labels (script counterpart of `convert.ipynb`) |
| `run_all.ps1` | One-shot: extract tar → install deps → convert → optional smoke train / predict / web |
| `train_yolo_compare.py` | Train several YOLO variants with shared hyperparameters |
| `run_yolo_compare.ps1` | Wrapper for compare training (logging) |
| `object_detector.py` + `index.html` | Local web UI at `http://127.0.0.1:8080` |
| `best.pt` | Default weights for inference (replace with your `weights/best.pt`) |

## Requirements

- Python 3.10+ recommended (match your PyTorch wheels).  
- Windows **tar** for `.tar` archives (used by `run_all.ps1`).

```powershell
cd <REPO_ROOT>   # same folder as run_convert.py
python -m pip install -r requirements.txt
```

## One-click pipeline (recommended)

Place your full archive (top-level `train/`, `valid/`, `test/`, `meta.json`, etc.) anywhere on disk, e.g. `C:\data\dentalai_sample.tar`:

```powershell
cd <REPO_ROOT>
.\run_all.ps1 -TarPath "C:\data\dentalai_sample.tar"
```

If `dataset/` already exists:

```powershell
.\run_all.ps1 -SkipExtract
```

Flags: `-SkipTrain`, `-SkipPredict`, `-SkipWeb`, `-Device 0` (GPU).  
Logs: `logs\run_all_*.log`

## Manual steps

1. Prepare `dataset/` as in the table above. **Required if you only cloned `yolo_dataset/data.yaml` — do not train before conversion.**  
2. `python run_convert.py` → full `yolo_dataset/` (`images/`, `labels/`, `data.yaml`).  
3. Train (same idea as `train.ipynb`), for example:

```python
from ultralytics import YOLO
import os
model = YOLO("yolov8m.pt")
model.train(
    data=os.path.join(os.getcwd(), "yolo_dataset", "data.yaml"),
    epochs=30,
    imgsz=640,
    batch=16,
    device="cpu",  # or 0 for GPU
)
```

4. Best weights are typically under `runs/detect/<run>/weights/best.pt` (exact path depends on `project` / `name`). Copy to repo-root `best.pt` for the default web demo.

## Web service

```powershell
cd <REPO_ROOT>
python object_detector.py
```

Open `http://127.0.0.1:8080` and upload an image.  
Custom weights:

```powershell
$env:YOLO_MODEL_PATH = "runs\detect\train\weights\best.pt"
python object_detector.py
```
<img src="https://github.com/LHHHailey/Object-Detection-for-Dental-Caries-and-Related-Findings-A-Comparative-Study-of-YOLO-Variants-/blob/main/caries_demo.png" width="600" alt="caries_demo">

**API:** `POST /detect`, form field **`image_file`**, JSON array of boxes. The sample server **returns only class id 0 (Caries)** for the demo.

## Multi-model comparison

Requires a valid `yolo_dataset\data.yaml` (and full dataset after conversion):

```powershell
.\run_yolo_compare.ps1 -Device cpu -Epochs 100 -Batch 16
```

Summary: `runs\compare\_compare_logs\summary_all.json`.  
TensorBoard: `tensorboard --logdir runs` (separate terminal).

## Data source

DentalAI and related links: [datasetninja.com/dentalai](https://datasetninja.com/dentalai) (see upstream README).

## License

See `LICENSE` / `LICENSE.md` in this repository.
