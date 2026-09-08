"""
Sliced (tiled) inference: run the detector on overlapping crops instead of the
whole downscaled frame, then merge the boxes back with NMS.

Why it matters here: the drone is roughly 30x15 px in an 848x480 frame. Feeding
that frame to the network at 960 barely helps, because the network still sees
the drone at its original scale. A 640 px crop of the same frame is upscaled to
the network's input, so the drone arrives several times larger. This is the
standard fix for small-object detection and it costs no retraining - only
inference time.

Use it when accuracy matters more than frame rate (offline analysis, evaluation,
recorded footage). It is roughly `tiles + 1` forward passes per frame, so a 2x2
grid is ~5x slower than a plain pass.

    python tiled_infer.py --weights best.pt --source dataset/images/test --save
    python tiled_infer.py --weights best.pt --source clip.mp4 --tile 512 --overlap 0.3

Importable:
    from tiled_infer import predict_tiled
    boxes = predict_tiled(model, frame, tile=640, overlap=0.25, conf=0.25)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def tile_origins(W, H, tile, overlap):
    """Top-left corners of overlapping tiles covering the whole frame."""
    step = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, W - tile + 1), step))
    ys = list(range(0, max(1, H - tile + 1), step))
    if not xs or xs[-1] + tile < W:
        xs.append(max(0, W - tile))
    if not ys or ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    return [(x, y) for y in sorted(set(ys)) for x in sorted(set(xs))]


def nms(boxes, iou_thr=0.5):
    """boxes: Nx5 (x1,y1,x2,y2,conf). Plain numpy - no torch needed."""
    if len(boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    b = np.asarray(boxes, dtype=np.float32)
    x1, y1, x2, y2, sc = b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4]
    area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = sc.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest]); iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest]); iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        union = area[i] + area[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou < iou_thr]
    return b[keep]


def predict_tiled(model, image, tile=640, overlap=0.25, conf=0.25, device="0",
                  iou_merge=0.5, full_pass=True, full_imgsz=960, tta=False):
    """Detect on overlapping tiles + (optionally) the whole frame.

    Returns Nx5 float32 (x1, y1, x2, y2, conf) in full-image pixel coordinates.
    """
    img = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
    if img is None:
        raise ValueError(f"cannot read image: {image}")
    H, W = img.shape[:2]
    tile = min(tile, max(H, W))
    out = []

    crops, offsets = [], []
    for (x, y) in tile_origins(W, H, tile, overlap):
        x2, y2 = min(W, x + tile), min(H, y + tile)
        crops.append(img[y:y2, x:x2])
        offsets.append((x, y))

    # one batched call per frame beats one call per tile
    if crops:
        res = model.predict(crops, imgsz=tile, device=device, conf=conf,
                            augment=tta, verbose=False)
        for r, (ox, oy) in zip(res, offsets):
            if r.boxes is None or not len(r.boxes):
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            xyxy[:, [0, 2]] += ox
            xyxy[:, [1, 3]] += oy
            out.append(np.column_stack([xyxy, cf]))

    # the whole frame too: a drone straddling a tile seam, or one large enough
    # that a crop clips it, is only caught here
    if full_pass:
        r = model.predict(img, imgsz=full_imgsz, device=device, conf=conf,
                          augment=tta, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            out.append(np.column_stack([r.boxes.xyxy.cpu().numpy(),
                                        r.boxes.conf.cpu().numpy()]))

    if not out:
        return np.zeros((0, 5), dtype=np.float32)
    merged = nms(np.vstack(out), iou_merge)
    merged[:, [0, 2]] = merged[:, [0, 2]].clip(0, W)
    merged[:, [1, 3]] = merged[:, [1, 3]].clip(0, H)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image, folder, or video")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960, help="full-frame pass size")
    ap.add_argument("--device", default="0")
    ap.add_argument("--tta", action="store_true", help="also flip/scale augment")
    ap.add_argument("--no-full-pass", action="store_true")
    ap.add_argument("--save", action="store_true", help="write annotated output")
    ap.add_argument("--out", default="tiled_output")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    src = Path(args.source)
    outdir = Path(args.out)
    if args.save:
        outdir.mkdir(parents=True, exist_ok=True)

    def draw(img, boxes):
        for x1, y1, x2, y2, c in boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img, f"drone {c:.2f}", (int(x1), max(14, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return img

    kw = dict(tile=args.tile, overlap=args.overlap, conf=args.conf,
              device=args.device, full_pass=not args.no_full_pass,
              full_imgsz=args.imgsz, tta=args.tta)

    if src.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {src}")
        writer, n, hits = None, 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes = predict_tiled(model, frame, **kw)
            n += 1
            hits += 1 if len(boxes) else 0
            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(outdir / (src.stem + "_tiled.mp4")),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                writer.write(draw(frame, boxes))
        cap.release()
        if writer:
            writer.release()
        print(f"{n} frames, {hits} with a detection ({hits/max(1,n):.0%})")
        return

    imgs = ([src] if src.is_file()
            else sorted(p for p in src.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")))
    tiles = None
    total = 0
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if tiles is None:
            H, W = img.shape[:2]
            tiles = len(tile_origins(W, H, min(args.tile, max(H, W)), args.overlap))
            print(f"{W}x{H} -> {tiles} tiles"
                  f"{' + full frame' if not args.no_full_pass else ''} per image\n")
        boxes = predict_tiled(model, img, **kw)
        total += len(boxes)
        print(f"  {p.name:<44} {len(boxes)} detection(s)"
              + (f"  best conf {boxes[:,4].max():.2f}" if len(boxes) else ""))
        if args.save:
            cv2.imwrite(str(outdir / p.name), draw(img, boxes))
    print(f"\n{len(imgs)} images, {total} detections")
    if args.save:
        print(f"annotated output: {outdir.resolve()}")


if __name__ == "__main__":
    main()
