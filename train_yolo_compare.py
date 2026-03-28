"""
Compare training across Ultralytics YOLO model checkpoints with unified hyperparameters.

Uses one data.yaml (default: ./yolo_dataset/data.yaml). Each run saves under runs/compare/<run_name>/.

Note on versions:
  - v5 / v8 / v11: official *.pt names from Ultralytics (auto-download).
  - v12: enabled if your ultralytics version provides weights (e.g. yolo12m.pt); otherwise skipped.
  - v26: not an official Ultralytics release name; script tries yolov26m.pt if present locally, else skips.

Unified settings (CLI overridable):
  epochs=100, batch=16, imgsz=640, lr0=0.01, cos_lr=True, lrf=0.01, optimizer=SGD,
  basic aug: fliplr + HSV + mosaic; mixup/copy_paste off.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Default model ids: (run_folder_name, weights_file)
DEFAULT_MODELS = [
    ("yolov5nu", "yolov5nu.pt"),
    ("yolov8m", "yolov8m.pt"),
    ("yolo11m", "yolo11m.pt"),
    ("yolo12m", "yolo12m.pt"),
    ("yolov26m", "yolov26m.pt"),
]


def build_train_kwargs(args: argparse.Namespace) -> dict:
    return {
        "data": str(Path(args.data).resolve()),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "cos_lr": True,
        "optimizer": "SGD",
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "mosaic": 1.0,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "device": args.device,
        "project": str(Path(args.project).resolve()),
        "exist_ok": True,
        "pretrained": True,
        "plots": True,
        "verbose": True,
        "save": True,
        "val": True,
        "patience": 0,
    }


def run_one(name: str, weights: str, train_kw: dict, log_dir: Path) -> dict:
    from ultralytics import YOLO

    row = {
        "name": name,
        "weights": weights,
        "status": "unknown",
        "best": None,
        "error": None,
        "seconds": None,
    }
    wpath = Path(weights)
    if not wpath.is_file() and not str(weights).startswith("http"):
        # let YOLO download official pt; if fails, skip
        pass

    t0 = time.perf_counter()
    try:
        model = YOLO(weights)
        train_kw = {**train_kw, "name": name}
        model.train(**train_kw)
        row["status"] = "ok"
        save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
        if save_dir:
            cand = Path(save_dir) / "weights" / "best.pt"
            if cand.is_file():
                row["best"] = str(cand.resolve())
        if row["best"] is None:
            fallback = Path(train_kw["project"]) / name / "weights" / "best.pt"
            if fallback.is_file():
                row["best"] = str(fallback.resolve())
    except Exception as e:
        row["status"] = "failed"
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()
    row["seconds"] = round(time.perf_counter() - t0, 2)

    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{name}_result.json", "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="yolo_dataset/data.yaml", help="YOLO data.yaml path")
    p.add_argument("--project", default="runs/compare", help="Ultralytics project root")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01, help="Final lr = lr0 * lrf (cosine end)")
    p.add_argument("--device", default="cpu", help="cpu or GPU id e.g. 0")
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help='Optional pairs: name weights, e.g. yolov8m yolov8m.pt yolo11m yolo11m.pt',
    )
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"ERROR: data yaml not found: {data_path.resolve()}", file=sys.stderr)
        return 2

    if args.models:
        if len(args.models) % 2 != 0:
            print("ERROR: --models must be name weights name weights ...", file=sys.stderr)
            return 2
        pairs = list(zip(args.models[0::2], args.models[1::2]))
    else:
        pairs = DEFAULT_MODELS

    train_kw = build_train_kwargs(args)
    log_dir = Path(args.project) / "_compare_logs"
    summary = []

    print("Unified train kwargs:", {k: train_kw[k] for k in ("epochs", "batch", "imgsz", "lr0", "lrf", "device")})
    for name, w in pairs:
        print(f"\n>>> Training {name} ({w})")
        row = run_one(name, w, dict(train_kw), log_dir)
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    out = log_dir / "summary_all.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    failed = [r for r in summary if r["status"] != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
