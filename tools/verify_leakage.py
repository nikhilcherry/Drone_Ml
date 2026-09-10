"""Independent check that val/test are not leaked into by train.

split_public.py measures its own leakage and writes split_report.json. A tool
grading its own homework is exactly the thing this project's README is careful
about elsewhere, and every number in docs/ACCURACY_INVESTIGATION.md rests on
this split being clean. So: recompute it from scratch, different code path.

Nearest-neighbour RMSE from each val/test image to any training image, on
grayscale thumbnails. Thumbnail RMSE rather than a perceptual hash for the
reason split_public.py already documents: drone imagery is mostly smooth sky,
and average-hash thresholds smooth gradients into identical bit patterns.
"""
import sys, argparse
sys.path.insert(0,"tools")
from pathlib import Path
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--train-sample", type=int, default=12000)
ap.add_argument("--size", type=int, default=16)
ap.add_argument("--out", default=None)
a = ap.parse_args()
R = Path("dataset_public/images")

def thumbs(paths, n):
    out = np.zeros((len(paths), n*n), np.float32)
    keep = []
    for i, p in enumerate(paths):
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        out[len(keep)] = cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA).ravel()
        keep.append(p)
    return out[:len(keep)], keep

tr = sorted((R/"train").iterdir())
rng = np.random.default_rng(0)
if len(tr) > a.train_sample:
    tr = [tr[i] for i in rng.choice(len(tr), a.train_sample, replace=False)]
ev = sorted((R/a.split).iterdir())
print(f"train sample {len(tr)}  vs  {a.split} {len(ev)}  ({a.size}x{a.size} thumbs)", flush=True)
T, tr = thumbs(tr, a.size)
E, ev = thumbs(ev, a.size)
d = a.size*a.size
Tn = (T*T).sum(1)
best = np.full(len(E), np.inf, np.float32); arg = np.zeros(len(E), int)
B = 256
for i in range(0, len(E), B):
    e = E[i:i+B]
    # squared euclidean -> RMSE
    sq = (e*e).sum(1)[:,None] + Tn[None,:] - 2.0 * (e @ T.T)
    np.maximum(sq, 0, out=sq)
    j = sq.argmin(1)
    v = np.sqrt(sq[np.arange(len(e)), j] / d)
    best[i:i+len(e)] = v; arg[i:i+len(e)] = j
    if (i//B) % 4 == 0: print(f"  {i+len(e)}/{len(E)}", flush=True)

print(f"\n=== {a.split} vs train, nearest-neighbour RMSE ===")
print(f"  median {np.median(best):.1f}   p05 {np.percentile(best,5):.1f}   min {best.min():.1f}")
for t in (1, 3, 6, 10):
    print(f"  within RMSE {t:>2}: {(best<=t).mean():>6.2%}  ({(best<=t).sum()} images)")
if a.out:
    order = np.argsort(best)[:12]
    C = 150; canvas = np.full((2*C, len(order)*C, 3), 25, np.uint8)
    for k, idx in enumerate(order):
        for row, p in ((0, ev[idx]), (1, tr[arg[idx]])):
            im = cv2.imread(str(p))
            if im is None: continue
            s = C/max(im.shape[:2]); im = cv2.resize(im, (int(im.shape[1]*s), int(im.shape[0]*s)))
            h, w = im.shape[:2]; canvas[row*C:row*C+h, k*C:k*C+w] = im
        cv2.putText(canvas, f"{best[idx]:.1f}", (k*C+4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
    cv2.putText(canvas, a.split.upper(), (4, C-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(canvas, "NEAREST TRAIN", (4, 2*C-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.imwrite(a.out, canvas); print(f"  wrote {a.out}")
