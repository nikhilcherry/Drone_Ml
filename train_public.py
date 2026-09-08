"""
Train on the public drone data, on a split that has not been leaked into.

This is stage A of `train_v3.py` with an honest yardstick attached. It trains on
`dataset_public/` (built by `split_public.py`), selects on that split's val, and
touches its test set exactly once, at the end, with the selected model.

Why it exists as its own script: `train_v3.py` requires `dataset/` - the local
courtyard footage - and refuses to run without it. The public data is useful on
its own, both as a detector that has actually seen grass, trees and 20-pixel
drones, and as the starting weights for a later local fine-tune:

    python train_public.py
    python train_v3.py --skip-a --stage-a-weights runs/detect/public_s/weights/best.pt

Reported at the end, in this order:

    val              selection metric, watched every epoch
    test             evaluated once, by the winner only
    test per source  which dataset the score comes from, because a studio
                     catalogue shot and a 20-pixel drone over a field are not
                     the same problem and averaging them hides that
    test by size     small (<32x32) / medium / large recall

Anything that changes the weights is a separate run with a separate name.
--tta does not, and is reported as a separate row on the same model.
"""
import argparse, json, time
from collections import defaultdict
from pathlib import Path

DATA = Path("dataset_public/data.yaml")
PROJECT = "runs/detect"
RESULTS = Path("public_results.json")

# Public data is varied enough to earn real augmentation - unlike the 294 local
# frames, where heavier augmentation cost ~0.20 mAP50 (see README).
AUG = dict(
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
    degrees=6.0, translate=0.12, scale=0.5,
    shear=0.0, perspective=0.0005,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, close_mosaic=15, mixup=0.0, copy_paste=0.0,
)


def check_gpu():
    import torch
    ok = torch.cuda.is_available()
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  available={ok}")
    if ok:
        print(f"device: {torch.cuda.get_device_name(0)}")
    return ok


def metrics(m):
    return dict(P=round(float(m.box.mp), 4), R=round(float(m.box.mr), 4),
                mAP50=round(float(m.box.map50), 4),
                mAP5095=round(float(m.box.map), 4))


def show(tag, d):
    print(f"  {tag:<28} P {d['P']:.3f}  R {d['R']:.3f}  "
          f"mAP50 {d['mAP50']:.3f}  mAP50-95 {d['mAP5095']:.3f}")


def evaluate(weights, split, imgsz, device, augment=False):
    from ultralytics import YOLO
    return metrics(YOLO(str(weights)).val(
        data=str(DATA), split=split, imgsz=imgsz, device=device,
        augment=augment, verbose=False, plots=False))


def per_source(weights, imgsz, device, conf=0.25, iou_thr=0.5):
    """Recall and false positives on the test split, broken out by source
    dataset and by GT box size. Detection counts, not mAP - this answers 'does
    it find the drone', which is the question the live tracker actually asks."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    imgs = sorted(Path("dataset_public/images/test").glob("*"))
    stats = defaultdict(lambda: dict(gt=0, hit=0, fp=0, imgs=0))
    size_stats = defaultdict(lambda: dict(gt=0, hit=0))

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    for i in range(0, len(imgs), 16):
        batch = imgs[i:i + 16]
        preds_batch = model.predict([str(x) for x in batch], imgsz=imgsz,
                                    conf=conf, device=device, verbose=False)
        for p, r in zip(batch, preds_batch):
            # source name is the prefix fetch_public_data.py stamped on the file
            name = p.name[4:] if p.name.startswith("neg_") else p.name
            src = "_".join(name.split("_")[:3])
            H, W = r.orig_shape
            lf = Path("dataset_public/labels/test") / (p.stem + ".txt")
            gts = []
            for line in lf.read_text().splitlines():
                if not line.strip():
                    continue
                _, cx, cy, w, h = (float(v) for v in line.split())
                gts.append([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H])
            preds = r.boxes.xyxy.cpu().numpy().tolist() if len(r.boxes) else []
            used = set()
            stats[src]["imgs"] += 1
            for g in gts:
                area = (g[2]-g[0]) * (g[3]-g[1])
                band = "small" if area < 32*32 else ("medium" if area < 96*96 else "large")
                stats[src]["gt"] += 1
                size_stats[band]["gt"] += 1
                best, bi = 0.0, -1
                for k, pr in enumerate(preds):
                    if k in used:
                        continue
                    v = iou(g, pr)
                    if v > best:
                        best, bi = v, k
                if best >= iou_thr:
                    used.add(bi)
                    stats[src]["hit"] += 1
                    size_stats[band]["hit"] += 1
            stats[src]["fp"] += len(preds) - len(used)
    return stats, size_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--name", default="public_s")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="the public images are natively 640; going higher "
                         "mainly buys resolution on the sub-32px drones")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4,
                    help="dataloader workers. Each one costs about 1 GB of RAM; "
                         "8 of them will OOM a 24 GB machine that is doing anything else")
    ap.add_argument("--device", default="0")
    ap.add_argument("--weights", default=None,
                    help="skip training and just evaluate these weights")
    ap.add_argument("--tta", action="store_true", help="also report a TTA pass")
    args = ap.parse_args()

    if not check_gpu():
        raise SystemExit("CUDA not available - refusing to train on CPU.")
    if not DATA.exists():
        raise SystemExit(f"{DATA} not found - run:\n"
                         "  python fetch_public_data.py --source rf_anti_uav --rf-key KEY\n"
                         "  python split_public.py --src external/rf_anti_uav")

    out = {}
    if args.weights:
        best = Path(args.weights)
    else:
        from ultralytics import YOLO
        print(f"\n{'='*70}\n  {args.name}: {args.model} on {DATA}  "
              f"imgsz={args.imgsz} batch={args.batch}\n{'='*70}")
        t0 = time.time()
        model = YOLO(args.model)
        model.train(data=str(DATA), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, patience=args.patience, workers=args.workers,
                    device=args.device, seed=0, pretrained=True,
                    optimizer="auto", project=PROJECT, name=args.name,
                    exist_ok=True, plots=True, val=True, cos_lr=True, **AUG)
        # Ultralytics may nest the run under its own runs_dir - always ask.
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        out["train_minutes"] = round((time.time() - t0) / 60, 1)
        print(f"  trained in {out['train_minutes']} min -> {best}")
    if not best.exists():
        raise SystemExit(f"weights not found: {best}")
    out["weights"] = str(best)
    out["imgsz"] = args.imgsz

    print(f"\n{'='*70}\nVAL (selection set)\n{'='*70}")
    out["val"] = evaluate(best, "val", args.imgsz, args.device)
    show("val", out["val"])

    print(f"\n{'='*70}\nTEST - evaluated once, by the selected model only\n{'='*70}")
    out["test"] = evaluate(best, "test", args.imgsz, args.device)
    show("test", out["test"])
    if args.tta:
        out["test_tta"] = evaluate(best, "test", args.imgsz, args.device, augment=True)
        show("test + TTA", out["test_tta"])
        print("  TTA changes no weights; it is the same model evaluated differently.")

    print(f"\n{'='*70}\nTEST, broken out (conf 0.25, IoU 0.5)\n{'='*70}")
    src, size = per_source(best, args.imgsz, args.device)
    print(f"  {'source':<22}{'images':>8}{'boxes':>8}{'recall':>9}{'FP/img':>9}")
    for k in sorted(src):
        s = src[k]
        rec = s["hit"] / s["gt"] if s["gt"] else float("nan")
        print(f"  {k:<22}{s['imgs']:>8}{s['gt']:>8}{rec:>9.3f}"
              f"{s['fp']/max(s['imgs'],1):>9.3f}")
    print(f"\n  {'GT box size':<22}{'boxes':>8}{'recall':>9}")
    for band in ("small", "medium", "large"):
        s = size[band]
        if s["gt"]:
            label = "small (<32x32)" if band == "small" else band
            print(f"  {label:<22}{s['gt']:>8}{s['hit']/s['gt']:>9.3f}")
    out["test_per_source"] = {k: dict(v) for k, v in src.items()}
    out["test_by_size"] = {k: dict(v) for k, v in size.items()}

    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten to {RESULTS}")
    print(f"\nFine-tune onto the local footage when it exists:\n"
          f"  python train_v3.py --skip-a --stage-a-weights \"{best}\"")


if __name__ == "__main__":
    main()
