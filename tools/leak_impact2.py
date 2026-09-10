"""Leakage impact, controlled for source AND box size.

The first attempt compared a near-duplicate half that was 97% rf_anti_uav
against a clean half that was 100% rf_anti_drone - two sources whose recall
differs by 25 points on their own. That measured the source mix, not leakage.

This stratifies: within ONE source, and within the small-box band, compare
images that have a near-identical twin in train against images that do not.
"""
import sys, os, json, random
sys.path.insert(0,"tools"); sys.path.insert(0,"."); os.environ["YOLO_VERBOSE"]="False"
import numpy as np
from ultralytics import YOLO
from evalkit import load_gt, iou_matrix, compute_ap

nn = json.load(open("test_nn_rmse.json"))
gt = load_gt("test")
m = YOLO("runs/detect/runs/detect/public_s/weights/best.pt")

def evaluate(stems, tag):
    sub = {s: gt[s] for s in stems}; preds = {}
    for s in stems:
        r = m.predict(str(gt[s]["path"]), imgsz=640, conf=0.001, device="cpu",
                      verbose=False, max_det=300)[0]
        preds[s] = (np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()])
                    .astype(np.float32) if r.boxes is not None and len(r.boxes)
                    else np.zeros((0,5), np.float32))
    ap = compute_ap(sub, preds); tp = ngt = 0; sides = []
    for s, g in sub.items():
        p = preds[s]; p = p[p[:,4] >= 0.25]
        M = iou_matrix(p[:,:4], g["boxes"]); used = np.zeros(len(g["boxes"]), bool)
        for pi in np.argsort(-p[:,4]) if len(p) else []:
            if M.shape[1]==0: continue
            for gi in np.argsort(-M[pi]):
                if M[pi,gi] < 0.5: break
                if not used[gi]: used[gi]=True; break
        tp += int(used.sum()); ngt += len(g["boxes"])
        for b in g["boxes"]: sides.append(float(np.sqrt((b[2]-b[0])*(b[3]-b[1]))))
    print(f"    {tag:<34} n={len(stems):>4} boxes={ngt:>4}  "
          f"mAP50 {ap['mAP50']:.4f}  R {tp/max(ngt,1):.4f}  "
          f"median side {np.median(sides):.1f}px", flush=True)
    return ap['mAP50'], tp/max(ngt,1)

random.seed(0)
for src in ("rf_anti_uav", "rf_anti_drone"):
    pool = [s for s, g in gt.items() if g["src"] == src and len(g["boxes"])]
    # restrict to the small band so box size cannot drive the difference
    def small_only(s):
        b = gt[s]["boxes"]
        sides = np.sqrt((b[:,2]-b[:,0])*(b[:,3]-b[:,1]))
        return sides.max() <= 32
    pool = [s for s in pool if small_only(s)]
    near = [s for s in pool if nn.get(s, 99) <= 1.0]
    far  = [s for s in pool if nn.get(s, 0) >= 10.0]
    print(f"\n  === {src}, small boxes only ===")
    print(f"    near-dup {len(near)}   clean {len(far)}")
    if len(near) < 40 or len(far) < 40:
        print("    too few in one bucket to compare"); continue
    random.shuffle(near); random.shuffle(far)
    k = min(len(near), len(far), 350)
    a = evaluate(near[:k], "near-duplicate of train (RMSE<=1)")
    b = evaluate(far[:k],  "clean (RMSE>=10)")
    print(f"    -> delta mAP50 {a[0]-b[0]:+.4f}   delta recall {a[1]-b[1]:+.4f}")
