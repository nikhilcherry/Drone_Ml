"""
Drone V2: run several training configs, pick the winner on VAL, then evaluate
that winner ONCE on TEST.

Nothing is tuned against the test set: selection uses val only.
Each run lands in runs/detect/<name>/ and all results are compared at the end.

    python experiment.py              # run every config (~25-35 min total)
    python experiment.py --only v2_hires
    python experiment.py --list
"""
import argparse, json, time
from pathlib import Path

DATA = "dataset/data.yaml"

# Shared augmentation: moderate, keeps drones realistic.
BASE_AUG = dict(
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,     # more colour/brightness variation
    degrees=6.0, translate=0.12, scale=0.5, # scale=0.5: the distance proxy
    shear=0.0, perspective=0.0005,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, close_mosaic=15, mixup=0.0, copy_paste=0.0,
)

CONFIGS = {
    # the V1 recipe plus the negative images - the control
    "v2_base":   dict(model="yolo11n.pt", imgsz=640,  batch=16, extra={}),

    # higher input resolution: small/distant drones survive downsampling better,
    # and your frames are mixed portrait/landscape so letterboxing costs pixels
    "v2_hires":  dict(model="yolo11n.pt", imgsz=960,  batch=8,  extra={}),

    # a bigger backbone: more capacity, more overfitting risk on ~300 images
    "v2_small":  dict(model="yolo11s.pt", imgsz=640,  batch=12, extra={}),

    # freeze the first 10 backbone layers: keeps generic COCO features intact,
    # a classic small-dataset defence against overfitting
    "v2_freeze": dict(model="yolo11n.pt", imgsz=640,  batch=16,
                      extra=dict(freeze=10)),

    # hi-res + multi_scale: strongest scale robustness we can get from config
    "v2_hires_ms": dict(model="yolo11n.pt", imgsz=960, batch=8,
                        extra=dict(multi_scale=True)),
}


def check_gpu():
    import torch
    ok = torch.cuda.is_available()
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  available={ok}")
    if ok:
        print(f"device: {torch.cuda.get_device_name(0)}")
    return ok


def run_one(name, cfg, epochs, patience):
    from ultralytics import YOLO
    print(f"\n{'='*62}\n  {name}   model={cfg['model']}  imgsz={cfg['imgsz']}  "
          f"batch={cfg['batch']}  {cfg['extra']}\n{'='*62}")
    t0 = time.time()
    model = YOLO(cfg["model"])
    model.train(
        data=DATA, epochs=epochs, imgsz=cfg["imgsz"], batch=cfg["batch"],
        patience=patience, workers=4, device=0, seed=0, pretrained=True,
        optimizer="auto", project="runs/detect", name=name, exist_ok=True,
        plots=True, val=True, deterministic=True, **BASE_AUG, **cfg["extra"],
    )
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    # validation metrics for the best checkpoint (selection happens here)
    m = YOLO(str(best)).val(data=DATA, split="val", imgsz=cfg["imgsz"],
                            device=0, verbose=False)
    r = dict(name=name, weights=str(best), imgsz=cfg["imgsz"],
             model=cfg["model"], minutes=round((time.time()-t0)/60, 1),
             val_P=round(float(m.box.mp), 4), val_R=round(float(m.box.mr), 4),
             val_mAP50=round(float(m.box.map50), 4),
             val_mAP5095=round(float(m.box.map), 4))
    print(f"\n  {name}: val mAP50 {r['val_mAP50']:.3f}  "
          f"mAP50-95 {r['val_mAP5095']:.3f}  P {r['val_P']:.3f}  R {r['val_R']:.3f}"
          f"  ({r['minutes']} min)")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run just one config")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in CONFIGS.items():
            print(f"  {k:<14} {v}")
        return
    if not check_gpu():
        raise SystemExit("CUDA not available - refusing to run on CPU.")

    todo = {args.only: CONFIGS[args.only]} if args.only else CONFIGS
    results = []
    for name, cfg in todo.items():
        try:
            results.append(run_one(name, cfg, args.epochs, args.patience))
        except Exception as e:
            print(f"\n  {name} FAILED: {type(e).__name__}: {e}")
            if "out of memory" in str(e).lower():
                print("  -> lower `batch` for this config and re-run "
                      f"with --only {name}")

    if not results:
        raise SystemExit("no runs completed")

    results.sort(key=lambda r: r["val_mAP5095"], reverse=True)
    Path("experiment_results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'='*72}\nVAL COMPARISON (selection metric: mAP50-95)\n{'='*72}")
    print(f"{'config':<14}{'model':<12}{'imgsz':>6}{'P':>8}{'R':>8}"
          f"{'mAP50':>8}{'mAP50-95':>10}{'min':>6}")
    for r in results:
        print(f"{r['name']:<14}{r['model']:<12}{r['imgsz']:>6}{r['val_P']:>8.3f}"
              f"{r['val_R']:>8.3f}{r['val_mAP50']:>8.3f}"
              f"{r['val_mAP5095']:>10.3f}{r['minutes']:>6.1f}")

    win = results[0]
    print(f"\nWINNER on val: {win['name']}")
    print(f"weights: {win['weights']}")

    # ---- the test set is touched exactly once, by the winner only ----
    from ultralytics import YOLO
    print(f"\n{'='*72}\nTEST SET - winner only, evaluated once\n{'='*72}")
    m = YOLO(win["weights"]).val(data=DATA, split="test", imgsz=win["imgsz"],
                                 device=0, plots=True,
                                 name=f"{win['name']}_test")
    print(f"  Precision: {m.box.mp:.4f}")
    print(f"  Recall:    {m.box.mr:.4f}")
    print(f"  mAP50:     {m.box.map50:.4f}")
    print(f"  mAP50-95:  {m.box.map:.4f}")
    print(f"\nComparison written to experiment_results.json")
    print(f"Use this model live:\n  python realtime_track.py "
          f"--weights \"{win['weights']}\" --imgsz {win['imgsz']}")


if __name__ == "__main__":
    main()
