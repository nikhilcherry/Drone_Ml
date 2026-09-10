"""
Experiment trainer: same data, same augmentation, same schedule as the V4
baseline - only the thing under test changes.

Differs from train_public.py in two ways that matter:
  * builds from a .yaml and then explicitly .load()s pretrained weights, so an
    architecture change keeps the backbone instead of training from scratch
    (train_public.py's YOLO(model) path would silently discard it)
  * never touches the test split. Selection is on val; test is spent once, at
    the end of the whole investigation, by the winner only.
"""
import argparse, json, time
from pathlib import Path

DATA = "dataset_public/data.yaml"

# identical to train_public.py's AUG - the baseline recipe, held fixed
AUG = dict(
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
    degrees=6.0, translate=0.12, scale=0.5,
    shear=0.0, perspective=0.0005,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, close_mosaic=15, mixup=0.0, copy_paste=0.0,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=DATA,
                    help="dataset yaml. dataset_tiled/data.yaml trains on 320px "
                         "crops while still validating on the ORIGINAL whole "
                         "val images, so the number stays comparable")
    ap.add_argument("--load", default=None, help="pretrained .pt to transfer")
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="0")
    ap.add_argument("--multi-scale", action="store_true")
    ap.add_argument("--optimizer", default="auto",
                    help="'auto' lets Ultralytics pick and SILENTLY IGNORES "
                         "lr0 - name an optimizer (e.g. AdamW) when fine-tuning "
                         "at a deliberately lower learning rate")
    ap.add_argument("--extra", default="{}", help="JSON of extra train kwargs")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    assert torch.cuda.is_available(), "refusing to train on CPU"
    free, total = torch.cuda.mem_get_info()
    print(f"GPU free {free/1e9:.2f}/{total/1e9:.2f} GB", flush=True)

    model = YOLO(args.model, task="detect")
    if args.load:
        model = model.load(args.load)
    det = model.model.model[-1]
    print(f"detect strides: {getattr(det,'stride',None)}", flush=True)
    print(f"params: {sum(p.numel() for p in model.model.parameters()):,}", flush=True)

    extra = json.loads(args.extra)
    t0 = time.time()
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, patience=args.patience, workers=args.workers,
                device=args.device, seed=0, pretrained=bool(args.load),
                optimizer=args.optimizer, project="runs/detect", name=args.name,
                exist_ok=True, plots=True, val=True, cos_lr=True,
                multi_scale=args.multi_scale, **AUG, **extra)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    mins = (time.time() - t0) / 60
    print(f"\ntrained {mins:.1f} min -> {best}", flush=True)

    m = YOLO(str(best)).val(data=args.data, split="val", imgsz=args.imgsz,
                            device=args.device, verbose=False, plots=False)
    out = dict(name=args.name, model=args.model, data=args.data, imgsz=args.imgsz,
               batch=args.batch, epochs=args.epochs, minutes=round(mins, 1),
               weights=str(best),
               val=dict(P=round(float(m.box.mp), 4), R=round(float(m.box.mr), 4),
                        mAP50=round(float(m.box.map50), 4),
                        mAP5095=round(float(m.box.map), 4)))
    print(json.dumps(out, indent=2), flush=True)
    Path(f"exp_{args.name}.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
