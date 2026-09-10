"""
Evaluate an already-trained drone model. Does NOT retrain.

  1. locates best.pt (wherever Ultralytics actually put it)
  2. runs the held-out TEST set once
  3. saves annotated predictions on every test image
  4. breaks recall down by drone size: small / medium / large

Two switches buy accuracy with no retraining:
  --tta    test-time augmentation (flips/scales, merged). Measured a LOSS on
           the public data - see the README's V4 section and docs/EXPERIMENTS.md.
           Do not enable it on that model.
  --tile   sliced inference (see tiled_infer.py). This is the one that moves the
           small-drone band, because a crop run at a LARGER network size shows
           the drone magnified. Note the tile must be smaller than the image:
           the public frames are natively 640x640, so --tile-size 640 is one
           tile covering the whole frame and does nothing at all.

    python evaluate.py
    python evaluate.py --weights path/to/best.pt --conf 0.25
    python evaluate.py --weights best.pt --tile --tile-size 320
"""
import argparse
from pathlib import Path

import numpy as np

# GT box area thresholds in pixels (COCO-style)
SMALL_MAX  = 32 * 32
MEDIUM_MAX = 96 * 96


def find_weights():
    hits = sorted(Path(".").rglob("drone_v1/weights/best.pt"))
    if not hits:
        hits = sorted(Path(".").rglob("best.pt"))
    if not hits:
        raise SystemExit("Could not find best.pt - pass --weights explicitly.")
    return hits[0]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--dataset", default="dataset",
                    help="dataset folder (default: dataset)")
    ap.add_argument("--tta", action="store_true",
                    help="test-time augmentation (measured as a LOSS on the "
                         "public data - see docs/EXPERIMENTS.md)")
    ap.add_argument("--tile", action="store_true",
                    help="sliced inference for the size breakdown")
    ap.add_argument("--tile-size", type=int, default=320,
                    help="crop size in source pixels; MUST be smaller than the "
                         "image or slicing is a no-op. Crops are run at 2x this "
                         "size, which is where the magnification comes from")
    ap.add_argument("--overlap", type=float, default=0.25)
    args = ap.parse_args()

    data = str(Path(args.dataset) / "data.yaml")
    images = Path(args.dataset) / "images" / "test"
    labels = Path(args.dataset) / "labels" / "test"

    from ultralytics import YOLO
    w = Path(args.weights) if args.weights else find_weights()
    print(f"weights: {w.resolve()}\n")
    model = YOLO(str(w))

    print("=== TEST SET (held out, evaluated once) ===")
    if args.tile:
        print("  (mAP below is the standard full-frame val; Ultralytics' val\n"
              "   cannot slice. The tiled gain shows in the size breakdown.)")
    m = model.val(data=data, split="test", imgsz=args.imgsz, augment=args.tta,
                  device=args.device, plots=True, name="drone_v1_test")
    print(f"  Precision: {m.box.mp:.4f}")
    print(f"  Recall:    {m.box.mr:.4f}")
    print(f"  mAP50:     {m.box.map50:.4f}")
    print(f"  mAP50-95:  {m.box.map:.4f}")

    print("\n=== annotated predictions ===")
    model.predict(source=str(images), imgsz=args.imgsz, device=args.device,
                  conf=args.conf, augment=args.tta, save=True,
                  name="drone_v1_test_predictions", exist_ok=True, verbose=False)

    mode = ("tiled" if args.tile else "full-frame") + (" + TTA" if args.tta else "")
    print(f"\n=== SIZE BREAKDOWN (recall at IoU>=0.5, conf>={args.conf}, "
          f"{mode}) ===")
    import cv2
    bands = {"small (<32x32)": [], "medium (32-96)": [], "large (>96x96)": []}
    ious_by_band = {k: [] for k in bands}
    if args.tile:
        from tiled_infer import predict_tiled
    for img_path in sorted(images.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lf = labels / (img_path.stem + ".txt")
        if not lf.exists():
            continue
        lines = [l for l in lf.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        im = cv2.imread(str(img_path))
        H, W = im.shape[:2]
        gts = []
        for line in lines:
            _, cx, cy, nw, nh = (float(v) for v in line.split())
            gts.append(((cx-nw/2)*W, (cy-nh/2)*H, (cx+nw/2)*W, (cy+nh/2)*H))

        if args.tile:
            preds = [tuple(b[:4]) for b in predict_tiled(
                model, im, tile=args.tile_size, overlap=args.overlap,
                conf=args.conf, device=args.device, full_imgsz=args.imgsz,
                tta=args.tta)]
        else:
            r = model.predict(source=str(img_path), imgsz=args.imgsz,
                              device=args.device, conf=args.conf,
                              augment=args.tta, verbose=False)[0]
            preds = ([tuple(b) for b in r.boxes.xyxy.cpu().numpy()]
                     if r.boxes is not None else [])

        for g in gts:
            area = (g[2]-g[0]) * (g[3]-g[1])
            band = ("small (<32x32)" if area < SMALL_MAX
                    else "medium (32-96)" if area < MEDIUM_MAX
                    else "large (>96x96)")
            best_iou = max((iou(g, p) for p in preds), default=0.0)
            bands[band].append(1 if best_iou >= 0.5 else 0)
            ious_by_band[band].append(best_iou)

    print(f"  {'band':<18}{'GT boxes':>9}{'detected':>10}{'recall':>9}{'mean IoU':>10}")
    for k, v in bands.items():
        if not v:
            print(f"  {k:<18}{0:>9}{'-':>10}{'-':>9}{'-':>10}")
            continue
        print(f"  {k:<18}{len(v):>9}{sum(v):>10}{sum(v)/len(v):>9.3f}"
              f"{np.mean(ious_by_band[k]):>10.3f}")
    total = [x for v in bands.values() for x in v]
    if total:
        print(f"  {'ALL':<18}{len(total):>9}{sum(total):>10}"
              f"{sum(total)/len(total):>9.3f}")


if __name__ == "__main__":
    main()
