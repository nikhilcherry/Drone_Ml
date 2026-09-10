"""
Evaluation harness for the public drone dataset.

Why this exists: evaluate.py runs one model.predict() per image plus a full
save-annotated pass, which is far too slow to sweep inference configs over
3,088 images. This batches, caches ground truth, and computes AP itself so a
tiled or merged prediction set can be scored with the same metric as a plain
one.

Correctness is checked against Ultralytics' own val() - see --verify.
"""
import argparse, json, time, collections
from pathlib import Path

import numpy as np
import cv2

ROOT = Path("dataset_public")
SOURCES = ["rf_anti_uav", "rf_anti_drone", "rf_drone_yolov7"]
SMALL_MAX, MEDIUM_MAX = 32 * 32, 96 * 96


def source_of(name):
    n = name[4:] if name.startswith("neg_") else name
    for s in SOURCES:
        if n.startswith(s):
            return s
    return "unknown"


def band_of(area):
    return "small" if area < SMALL_MAX else "medium" if area < MEDIUM_MAX else "large"


def load_gt(split, cache=True):
    """{stem: dict(path, wh, boxes Nx4 xyxy px, src)}. Caches image dims."""
    imgs, labs = ROOT / "images" / split, ROOT / "labels" / split
    cpath = Path(f".cache_gt_{split}.json")
    dims = {}
    if cache and cpath.exists():
        dims = json.loads(cpath.read_text())
    gt, new = {}, False
    for ip in sorted(imgs.iterdir()):
        if ip.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if ip.name in dims:
            W, H = dims[ip.name]
        else:
            im = cv2.imread(str(ip))
            if im is None:
                continue
            H, W = im.shape[:2]
            dims[ip.name] = [W, H]
            new = True
        lf = labs / (ip.stem + ".txt")
        boxes = []
        if lf.exists():
            for line in lf.read_text().split("\n"):
                if not line.strip():
                    continue
                p = line.split()
                cx, cy, nw, nh = (float(v) for v in p[1:5])
                boxes.append([(cx-nw/2)*W, (cy-nh/2)*H, (cx+nw/2)*W, (cy+nh/2)*H])
        gt[ip.stem] = dict(path=ip, wh=(W, H), src=source_of(ip.name),
                           boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4))
    if cache and new:
        cpath.write_text(json.dumps(dims))
    return gt


def iou_matrix(a, b):
    """a: Nx4, b: Mx4 -> NxM."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0).astype(np.float32)


def compute_ap(gt, preds, iou_thrs=None):
    """COCO-style AP. preds: {stem: Nx5 xyxy+conf}. Returns metrics dict."""
    if iou_thrs is None:
        iou_thrs = np.arange(0.5, 1.0, 0.05)
    n_gt = sum(len(g["boxes"]) for g in gt.values())
    rows = []          # (conf, stem, pred_idx)
    for stem, p in preds.items():
        for i, b in enumerate(p):
            rows.append((float(b[4]), stem, i))
    rows.sort(key=lambda r: -r[0])
    n_p = len(rows)
    tp = np.zeros((n_p, len(iou_thrs)), dtype=bool)
    fp = np.zeros((n_p, len(iou_thrs)), dtype=bool)
    ious = {stem: iou_matrix(preds.get(stem, np.zeros((0, 5)))[:, :4], g["boxes"])
            for stem, g in gt.items()}
    matched = {stem: np.zeros((len(iou_thrs), len(g["boxes"])), dtype=bool)
               for stem, g in gt.items()}
    for k, (conf, stem, pi) in enumerate(rows):
        M = ious.get(stem)
        g = gt[stem]
        if M is None or M.shape[1] == 0:
            fp[k, :] = True
            continue
        order = np.argsort(-M[pi])
        for t, thr in enumerate(iou_thrs):
            hit = False
            for gi in order:
                if M[pi, gi] < thr:
                    break
                if not matched[stem][t, gi]:
                    matched[stem][t, gi] = True
                    tp[k, t] = True
                    hit = True
                    break
            if not hit:
                fp[k, t] = True
    aps = []
    for t in range(len(iou_thrs)):
        ctp = np.cumsum(tp[:, t]); cfp = np.cumsum(fp[:, t])
        rec = ctp / max(n_gt, 1)
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        # 101-point interpolation
        mprec = np.concatenate([[1.0], prec, [0.0]])
        mrec = np.concatenate([[0.0], rec, [1.0]])
        for i in range(len(mprec) - 2, -1, -1):
            mprec[i] = max(mprec[i], mprec[i + 1])
        x = np.linspace(0, 1, 101)
        aps.append(np.trapezoid(np.interp(x, mrec, mprec), x))
    return dict(mAP50=float(aps[0]), mAP5095=float(np.mean(aps)), n_gt=n_gt, n_pred=n_p)


def operating_point(gt, preds, conf_thr=0.25, iou_thr=0.5):
    """P/R plus per-size and per-source recall at a fixed conf."""
    TP = FP = 0
    band = collections.defaultdict(lambda: [0, 0])   # hit, total
    src = collections.defaultdict(lambda: [0, 0, 0])  # hit, total, fp
    for stem, g in gt.items():
        p = preds.get(stem, np.zeros((0, 5), dtype=np.float32))
        p = p[p[:, 4] >= conf_thr] if len(p) else p
        M = iou_matrix(p[:, :4], g["boxes"])
        used = np.zeros(len(g["boxes"]), dtype=bool)
        order = np.argsort(-p[:, 4]) if len(p) else []
        pred_hit = np.zeros(len(p), dtype=bool)
        for pi in order:
            if M.shape[1] == 0:
                continue
            cand = np.argsort(-M[pi])
            for gi in cand:
                if M[pi, gi] < iou_thr:
                    break
                if not used[gi]:
                    used[gi] = True
                    pred_hit[pi] = True
                    break
        TP += int(used.sum()); FP += int((~pred_hit).sum())
        s = g["src"]
        src[s][1] += len(g["boxes"]); src[s][0] += int(used.sum())
        src[s][2] += int((~pred_hit).sum())
        for gi, b in enumerate(g["boxes"]):
            a = (b[2]-b[0]) * (b[3]-b[1])
            k = band_of(a)
            band[k][1] += 1; band[k][0] += int(used[gi])
    n_gt = sum(len(g["boxes"]) for g in gt.values())
    return dict(P=TP / max(TP + FP, 1), R=TP / max(n_gt, 1), TP=TP, FP=FP,
                by_size={k: dict(gt=v[1], hit=v[0], recall=v[0]/max(v[1],1))
                         for k, v in sorted(band.items())},
                by_source={k: dict(gt=v[1], hit=v[0], recall=v[0]/max(v[1],1),
                                   fp=v[2]) for k, v in sorted(src.items())})


def report(name, gt, preds, conf_thr=0.25, secs=None):
    ap = compute_ap(gt, preds)
    op = operating_point(gt, preds, conf_thr)
    print(f"\n===== {name} =====")
    if secs: print(f"  time {secs:.0f}s")
    print(f"  mAP50 {ap['mAP50']:.4f}   mAP50-95 {ap['mAP5095']:.4f}"
          f"   (gt={ap['n_gt']}, preds={ap['n_pred']})")
    print(f"  @conf{conf_thr}: P {op['P']:.4f}  R {op['R']:.4f}  "
          f"TP {op['TP']}  FP {op['FP']}")
    print(f"  {'band':<8}{'gt':>7}{'hit':>7}{'recall':>9}")
    for k in ("small", "medium", "large"):
        if k in op["by_size"]:
            v = op["by_size"][k]
            print(f"  {k:<8}{v['gt']:>7}{v['hit']:>7}{v['recall']:>9.3f}")
    print(f"  {'source':<18}{'gt':>7}{'hit':>7}{'recall':>9}{'fp':>7}")
    for k, v in op["by_source"].items():
        print(f"  {k:<18}{v['gt']:>7}{v['hit']:>7}{v['recall']:>9.3f}{v['fp']:>7}")
    return dict(name=name, **ap, **{k: v for k, v in op.items()})
