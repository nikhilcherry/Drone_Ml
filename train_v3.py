"""
V3: two-stage training on local footage + public drone data.

Stage A - pretrain on dataset_combined/ (local train + everything under
          external/). Thousands of drones over grass, trees, roads and open
          sky teach the detector what a drone *is*, which 198 courtyard frames
          cannot.
Stage B - fine-tune stage A on dataset/ alone at a low learning rate, so the
          model specialises back onto the camera and scene it will actually
          run against without forgetting stage A.

Selection is on the LOCAL val set, same as V2, so V3 numbers are directly
comparable to the README table. The test set is evaluated once, by the winner
only. If dataset_combined/data_extval.yaml exists it is also reported - that
is the unseen-background score, the one the README calls the biggest gap - but
it is never used to choose a model.

    python train_v3.py
    python train_v3.py --model yolo11s.pt --imgsz 1280 --batch 4
    python train_v3.py --skip-a --stage-a-weights runs/detect/v3_pretrain/weights/best.pt
    python train_v3.py --skip-b            # combined-data model only

Start with the defaults. yolo11s at 960 is the right size once the training
set is in the thousands; yolo11n was only chosen because 294 images cannot
feed anything bigger.
"""
import argparse, json, time
from pathlib import Path

COMBINED = Path("dataset_combined/data.yaml")
EXTVAL   = Path("dataset_combined/data_extval.yaml")
LOCAL    = Path("dataset/data.yaml")
PROJECT  = "runs/detect"
RESULTS  = Path("experiment_results_v3.json")

# Stage A: lots of varied data, so augment for scale and colour and let it run.
AUG_A = dict(
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
    degrees=6.0, translate=0.12, scale=0.5,
    shear=0.0, perspective=0.0005,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, close_mosaic=15, mixup=0.0, copy_paste=0.0,
)
# Stage B: the V2 recipe, which is what actually won on this footage. Heavier
# augmentation cost ~0.20 mAP50 here (README, "what did not work") - do not
# reintroduce it just because stage A uses it on a much larger set.
AUG_B = dict(
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
    degrees=5.0, translate=0.1, scale=0.4,
    shear=0.0, perspective=0.0005,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0,
)


def check_gpu():
    import torch
    ok = torch.cuda.is_available()
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  available={ok}")
    if ok:
        print(f"device: {torch.cuda.get_device_name(0)}")
    return ok


def val_metrics(weights, data, split, imgsz, device):
    from ultralytics import YOLO
    m = YOLO(str(weights)).val(data=str(data), split=split, imgsz=imgsz,
                               device=device, verbose=False)
    return dict(P=round(float(m.box.mp), 4), R=round(float(m.box.mr), 4),
                mAP50=round(float(m.box.map50), 4),
                mAP5095=round(float(m.box.map), 4))


def show(tag, m):
    print(f"  {tag:<26} P {m['P']:.3f}  R {m['R']:.3f}  "
          f"mAP50 {m['mAP50']:.3f}  mAP50-95 {m['mAP5095']:.3f}")


def train(name, weights, data, epochs, patience, imgsz, batch, device, aug, **extra):
    from ultralytics import YOLO
    print(f"\n{'='*66}\n  {name}: {weights} on {data}  imgsz={imgsz} batch={batch}"
          f"\n{'='*66}")
    t0 = time.time()
    model = YOLO(str(weights))
    model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch,
                patience=patience, workers=4, device=device, seed=0,
                pretrained=True, optimizer="auto", project=PROJECT, name=name,
                exist_ok=True, plots=True, val=True, deterministic=True,
                **aug, **extra)
    # Ultralytics may nest the run under its own runs_dir - always ask.
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"best.pt not found at {best}")
    print(f"  {name} done in {(time.time()-t0)/60:.1f} min -> {best}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt",
                    help="stage A starting weights (yolo11n.pt if VRAM is tight)")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="960 was worth ~0.13 mAP50 over 640; 1280 is worth "
                         "trying once the public data supplies genuinely small drones")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs-a", type=int, default=100)
    ap.add_argument("--epochs-b", type=int, default=80)
    ap.add_argument("--patience-a", type=int, default=20)
    ap.add_argument("--patience-b", type=int, default=25)
    ap.add_argument("--lr-b", type=float, default=0.0005,
                    help="stage B learning rate: low on purpose, so fine-tuning "
                         "adapts rather than overwrites stage A")
    ap.add_argument("--freeze-b", type=int, default=0,
                    help="freeze N backbone layers in stage B (try 10 if stage B "
                         "overfits the 294 local images)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--stage-a-weights", default=None)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()

    if not check_gpu():
        raise SystemExit("CUDA not available - refusing to train on CPU.")
    if not LOCAL.exists():
        raise SystemExit(f"{LOCAL} not found.")
    if not args.skip_a and not COMBINED.exists():
        raise SystemExit(f"{COMBINED} not found - run:\n"
                         "  python fetch_public_data.py --list\n"
                         "  python build_combined_dataset.py --apply")

    runs = {}

    # ---------------------------- stage A ----------------------------
    a_weights = Path(args.stage_a_weights) if args.stage_a_weights else None
    if not args.skip_a:
        a_weights = train("v3_pretrain", args.model, COMBINED,
                          args.epochs_a, args.patience_a, args.imgsz,
                          args.batch, args.device, AUG_A)
    if a_weights:
        print("\nstage A on the LOCAL splits:")
        runs["v3_pretrain"] = dict(weights=str(a_weights), imgsz=args.imgsz,
                                   val=val_metrics(a_weights, LOCAL, "val",
                                                   args.imgsz, args.device))
        show("stage A / local val", runs["v3_pretrain"]["val"])

    # ---------------------------- stage B ----------------------------
    if not args.skip_b:
        if not a_weights:
            raise SystemExit("--skip-a needs --stage-a-weights.")
        b_weights = train("v3_finetune", a_weights, LOCAL,
                          args.epochs_b, args.patience_b, args.imgsz,
                          args.batch, args.device, AUG_B,
                          lr0=args.lr_b, **({"freeze": args.freeze_b}
                                            if args.freeze_b else {}))
        runs["v3_finetune"] = dict(weights=str(b_weights), imgsz=args.imgsz,
                                   val=val_metrics(b_weights, LOCAL, "val",
                                                   args.imgsz, args.device))
        show("stage B / local val", runs["v3_finetune"]["val"])

    if not runs:
        raise SystemExit("nothing trained")

    # ------------- selection on local val, exactly as V2 did -------------
    print(f"\n{'='*66}\nSELECTION (local val, metric mAP50-95)\n{'='*66}")
    for k, v in runs.items():
        show(k, v["val"])
    win_name = max(runs, key=lambda k: runs[k]["val"]["mAP5095"])
    win = runs[win_name]
    print(f"\nWINNER on val: {win_name}\n  {win['weights']}")

    # unseen-background read, reported but never selected on
    if EXTVAL.exists():
        print("\nUnseen-background check (held-out public images):")
        for k, v in runs.items():
            v["ext_val"] = val_metrics(v["weights"], EXTVAL, "val",
                                       args.imgsz, args.device)
            show(k, v["ext_val"])
        print("  This is the number the README's limitation #1 is about. A model "
              "that scores here has learned drones, not one courtyard.")

    # ------------------- test set: once, winner only -------------------
    print(f"\n{'='*66}\nLOCAL TEST SET - winner only, evaluated once\n{'='*66}")
    win["test"] = val_metrics(win["weights"], LOCAL, "test", args.imgsz, args.device)
    show("test", win["test"])
    print("\n  V2_hires for reference: P 0.776  R 0.667  mAP50 0.771  mAP50-95 0.295")
    print("  With ~42 test boxes, differences under ~0.1 mAP50 are noise.")

    RESULTS.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"\nwritten to {RESULTS}")
    print(f"\nRun it live:\n  python realtime_track.py --weights \"{win['weights']}\" "
          f"--imgsz {args.imgsz}")
    print(f"Squeeze more out of it without retraining:\n"
          f"  python evaluate.py --weights \"{win['weights']}\" --imgsz {args.imgsz} --tta\n"
          f"  python evaluate.py --weights \"{win['weights']}\" --imgsz {args.imgsz} --tile")


if __name__ == "__main__":     # required on Windows when workers > 0
    main()
