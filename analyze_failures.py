"""
Diagnose V1's misses. Chooses a confidence threshold on VAL, then applies it
once to TEST. Does not retrain and does not tune against the test set.

Outputs:
  - F1 vs confidence sweep on val -> recommended threshold
  - test recall at that threshold, overall and per size band
  - failure_montage.jpg: crops of the GT boxes the model missed

    python analyze_failures.py
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

DATA = "dataset/data.yaml"
SMALL_MAX, MEDIUM_MAX = 32*32, 96*96


def find_weights():
    hits = sorted(Path(".").rglob("drone_v1/weights/best.pt")) or \
           sorted(Path(".").rglob("best.pt"))
    if not hits:
        raise SystemExit("best.pt not found - pass --weights")
    return hits[0]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def load_split(split):
    idir, ldir = Path(f"dataset/images/{split}"), Path(f"dataset/labels/{split}")
    out = []
    for p in sorted(idir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lf = ldir / (p.stem + ".txt")
        gts = []
        if lf.exists():
            im = cv2.imread(str(p)); H, W = im.shape[:2]
            for line in lf.read_text().splitlines():
                if not line.strip():
                    continue
                _, cx, cy, nw, nh = (float(v) for v in line.split())
                gts.append(((cx-nw/2)*W, (cy-nh/2)*H, (cx+nw/2)*W, (cy+nh/2)*H))
        out.append((p, gts))
    return out


def run(model, items, imgsz, device, conf):
    """Return list of (path, gts, preds) with preds = [(box, conf)]."""
    res = []
    for p, gts in items:
        r = model.predict(source=str(p), imgsz=imgsz, device=device,
                          conf=conf, verbose=False)[0]
        preds = []
        if r.boxes is not None and len(r.boxes):
            for b, c in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.conf.cpu().numpy()):
                preds.append((tuple(b), float(c)))
        res.append((p, gts, preds))
    return res


def score(res, thr):
    tp = fp = fn = 0
    for _, gts, preds in res:
        kept = [b for b, c in preds if c >= thr]
        matched = set()
        for g in gts:
            best, bi = 0.0, -1
            for i, b in enumerate(kept):
                if i in matched:
                    continue
                v = iou(g, b)
                if v > best:
                    best, bi = v, i
            if best >= 0.5:
                tp += 1; matched.add(bi)
            else:
                fn += 1
        fp += len(kept) - len(matched)
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    w = Path(args.weights) if args.weights else find_weights()
    print(f"weights: {w.resolve()}\n")
    model = YOLO(str(w))

    # ---- sweep on VAL only ----
    val = run(model, load_split("val"), args.imgsz, args.device, conf=0.01)
    print("=== VAL: threshold sweep (choose here, not on test) ===")
    print(f"  {'conf':>6}{'precision':>11}{'recall':>9}{'F1':>8}")
    best_thr, best_f1 = 0.25, -1.0
    for thr in [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        p_, r_, f_ = score(val, thr)
        star = ""
        if f_ > best_f1:
            best_f1, best_thr, star = f_, thr, ""
        print(f"  {thr:>6.2f}{p_:>11.3f}{r_:>9.3f}{f_:>8.3f}")
    print(f"\n  best F1 on val at conf = {best_thr:.2f}  (F1 {best_f1:.3f})")

    # ---- apply once to TEST ----
    test = run(model, load_split("test"), args.imgsz, args.device, conf=0.01)
    p_, r_, f_ = score(test, best_thr)
    p25, r25, f25 = score(test, 0.25)
    print(f"\n=== TEST at the val-chosen threshold {best_thr:.2f} ===")
    print(f"  precision {p_:.3f}   recall {r_:.3f}   F1 {f_:.3f}")
    print(f"  (for comparison, at conf 0.25: "
          f"precision {p25:.3f}  recall {r25:.3f}  F1 {f25:.3f})")

    # ---- per band at the chosen threshold + collect misses ----
    bands = {"small (<32x32)": [0, 0], "medium (32-96)": [0, 0],
             "large (>96x96)": [0, 0]}
    misses = []
    for p, gts, preds in test:
        kept = [b for b, c in preds if c >= best_thr]
        for g in gts:
            area = (g[2]-g[0])*(g[3]-g[1])
            k = ("small (<32x32)" if area < SMALL_MAX
                 else "medium (32-96)" if area < MEDIUM_MAX else "large (>96x96)")
            best = max((iou(g, b) for b in kept), default=0.0)
            bands[k][1] += 1
            if best >= 0.5:
                bands[k][0] += 1
            else:
                misses.append((p, g, best, int(area)))
    print(f"\n=== TEST size breakdown at conf {best_thr:.2f} ===")
    print(f"  {'band':<18}{'GT':>5}{'found':>7}{'recall':>9}")
    for k, (hit, tot) in bands.items():
        print(f"  {k:<18}{tot:>5}{hit:>7}"
              f"{(hit/tot if tot else 0):>9.3f}")

    # ---- montage of misses ----
    if misses:
        tiles = []
        for p, g, best, area in misses[:12]:
            im = cv2.imread(str(p))
            H, W = im.shape[:2]
            x1, y1, x2, y2 = (int(v) for v in g)
            pad = int(max(x2-x1, y2-y1)*0.8)+10
            X1, Y1 = max(0, x1-pad), max(0, y1-pad)
            X2, Y2 = min(W, x2+pad), min(H, y2+pad)
            crop = im[Y1:Y2, X1:X2].copy()
            if crop.size == 0:
                continue
            cv2.rectangle(crop, (x1-X1, y1-Y1), (x2-X1, y2-Y1), (0, 255, 0), 2)
            crop = cv2.resize(crop, (260, 260), interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop, f"missed {int(np.sqrt(area))}px IoU{best:.2f}",
                        (5, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(crop)
        while len(tiles) % 4:
            tiles.append(np.zeros((260, 260, 3), np.uint8))
        rows = [np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]
        cv2.imwrite("failure_montage.jpg", np.vstack(rows))
        print(f"\n{len(misses)} missed GT box(es). "
              f"First {min(12,len(misses))} written to failure_montage.jpg "
              f"(green = the drone the model did not find)")


if __name__ == "__main__":
    main()
