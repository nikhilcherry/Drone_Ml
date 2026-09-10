"""What does the recall recovered below conf 0.25 actually cost in precision?

'90.7% of the conf-0.25 misses are recoverable by lowering the threshold' is
true and, on its own, misleading: recall bought at conf 0.001 is worthless if
precision has collapsed to single digits by then. This prices it.
"""
import sys, os
sys.path.insert(0,"tools"); sys.path.insert(0,"."); os.environ["YOLO_VERBOSE"]="False"
import numpy as np
from ultralytics import YOLO
from evalkit import load_gt, iou_matrix, band_of

m = YOLO("runs/detect/runs/detect/public_s/weights/best.pt")
gt = load_gt("val")
stems = [s for s in gt][:500]          # includes negatives, so FPs are counted fairly
ngt = sum(len(gt[s]["boxes"]) for s in stems)
print(f"{len(stems)} images, {ngt} GT boxes", flush=True)

rows = []   # (conf, is_tp)
for i, s in enumerate(stems):
    g = gt[s]["boxes"]
    r = m.predict(str(gt[s]["path"]), imgsz=640, conf=0.001, device="cpu",
                  verbose=False, max_det=300)[0]
    if r.boxes is None or not len(r.boxes):
        continue
    p = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
    M = iou_matrix(p, g)
    used = np.zeros(len(g), bool)
    for pi in np.argsort(-cf):
        tp = False
        if M.shape[1]:
            for gi in np.argsort(-M[pi]):
                if M[pi, gi] < 0.5: break
                if not used[gi]:
                    used[gi] = tp = True; break
        rows.append((float(cf[pi]), tp))
    if i % 150 == 0: print(f"  {i}/{len(stems)}", flush=True)

rows.sort(key=lambda r: -r[0])
tp = np.cumsum([r[1] for r in rows]); n = np.arange(1, len(rows)+1)
prec = tp / n; rec = tp / ngt
confs = np.array([r[0] for r in rows])
print(f"\n  {'conf':>7}{'precision':>11}{'recall':>9}{'F1':>8}")
for c in (0.5, 0.35, 0.25, 0.15, 0.10, 0.05, 0.03, 0.01, 0.001):
    k = np.searchsorted(-confs, -c)
    if k == 0 or k > len(rows): continue
    k = min(k, len(rows)) - 1
    f1 = 2*prec[k]*rec[k]/max(prec[k]+rec[k], 1e-9)
    print(f"  {c:>7.3f}{prec[k]:>11.3f}{rec[k]:>9.3f}{f1:>8.3f}")
best = int(np.argmax(2*prec*rec/np.maximum(prec+rec, 1e-9)))
print(f"\n  F1 optimum at conf {confs[best]:.3f}: P {prec[best]:.3f} R {rec[best]:.3f}")
for target in (0.9, 0.8, 0.7, 0.5):
    idx = np.where(prec >= target)[0]
    if len(idx):
        k = idx[-1]
        print(f"  max recall at precision >= {target}: {rec[k]:.3f} (conf {confs[k]:.3f})")
