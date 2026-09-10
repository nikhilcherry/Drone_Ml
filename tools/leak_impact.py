"""Does the residual train/test similarity inflate the headline number?

Leakage only matters if it makes the score better. Split test by nearest-
neighbour distance to any training image and evaluate the SAME model on each
half. If the near-duplicate half scores much higher, the benchmark is optimistic
by roughly that much, weighted by how big the half is.

Confounders are reported alongside, because the two halves are not automatically
comparable: if the near-duplicate half happened to be mostly large studio drones
it would score higher for reasons that have nothing to do with leakage.
"""
import sys, os, json
sys.path.insert(0,"tools"); sys.path.insert(0,"."); os.environ["YOLO_VERBOSE"]="False"
import numpy as np, cv2
from pathlib import Path
from ultralytics import YOLO
from evalkit import load_gt, iou_matrix, compute_ap, band_of

R = Path("dataset_public/images")
def thumbs(paths, n=32):
    X = np.zeros((len(paths), n*n), np.float32); k = 0; keep = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        X[k] = cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA).ravel(); k += 1
        keep.append(p)
    return X[:k], keep

cache = Path("test_nn_rmse.json")
if cache.exists():
    nn = {k: v for k, v in json.load(open(cache)).items()}
else:
    tr, _ = thumbs(sorted((R/"train").iterdir()))
    E, ev = thumbs(sorted((R/"test").iterdir()))
    Tn = (tr*tr).sum(1); best = np.full(len(E), np.inf, np.float32)
    for i in range(0, len(E), 256):
        e = E[i:i+256]
        sq = (e*e).sum(1)[:,None] + Tn[None,:] - 2.0*(e @ tr.T)
        np.maximum(sq, 0, out=sq)
        best[i:i+len(e)] = np.sqrt(sq.min(1)/tr.shape[1])
    nn = {p.stem: float(b) for p, b in zip(ev, best)}
    json.dump(nn, open(cache, "w"))
print(f"nearest-RMSE cached for {len(nn)} test images", flush=True)

gt = load_gt("test")
near = [s for s in gt if nn.get(s, 99) <= 1.0 and len(gt[s]["boxes"])]
far  = [s for s in gt if nn.get(s, 0) >= 10.0 and len(gt[s]["boxes"])]
print(f"near-duplicate (RMSE<=1): {len(near)} imgs   clean (RMSE>=10): {len(far)} imgs")

m = YOLO("runs/detect/runs/detect/public_s/weights/best.pt")
def evaluate(stems, tag):
    sub = {s: gt[s] for s in stems}
    preds = {}
    for i, s in enumerate(stems):
        r = m.predict(str(gt[s]["path"]), imgsz=640, conf=0.001, device="cpu",
                      verbose=False, max_det=300)[0]
        preds[s] = (np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()])
                    .astype(np.float32) if r.boxes is not None and len(r.boxes)
                    else np.zeros((0,5), np.float32))
    ap = compute_ap(sub, preds)
    tp = 0; ngt = 0
    bands = {"small":0,"medium":0,"large":0}; srcs = {}
    for s, g in sub.items():
        p = preds[s]; p = p[p[:,4] >= 0.25]
        M = iou_matrix(p[:,:4], g["boxes"]); used = np.zeros(len(g["boxes"]), bool)
        for pi in np.argsort(-p[:,4]) if len(p) else []:
            if M.shape[1]==0: continue
            for gi in np.argsort(-M[pi]):
                if M[pi,gi] < 0.5: break
                if not used[gi]: used[gi]=True; break
        tp += int(used.sum()); ngt += len(g["boxes"])
        srcs[g["src"]] = srcs.get(g["src"],0)+len(g["boxes"])
        for b in g["boxes"]: bands[band_of((b[2]-b[0])*(b[3]-b[1]))] += 1
    print(f"\n  {tag}: {len(stems)} imgs, {ngt} boxes")
    print(f"    mAP50 {ap['mAP50']:.4f}   recall@.25 {tp/max(ngt,1):.4f}")
    print(f"    size mix: " + "  ".join(f"{k} {v/max(ngt,1):.0%}" for k,v in bands.items()))
    print(f"    source mix: " + "  ".join(f"{k.replace('rf_','')} {v/max(ngt,1):.0%}" for k,v in srcs.items()))
    return ap['mAP50'], tp/max(ngt,1)

a = evaluate(near[:400], "NEAR-DUPLICATE of train (RMSE<=1)")
b = evaluate(far[:400],  "CLEAN (RMSE>=10)")
print(f"\n  delta mAP50 (near - clean): {a[0]-b[0]:+.4f}")
print(f"  delta recall              : {a[1]-b[1]:+.4f}")
