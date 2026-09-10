"""Full-val sweep of every accepted change, on GPU, in one pass.

The tiling and operating-point numbers so far are on subsets - chosen hard on
purpose, because an average subset saturates and measures nothing, but a subset
recall is not comparable to a full-split mAP50. This runs the same configs
across the whole split so they can be quoted together.

Still val. Test stays untouched until there is a single winner to spend it on.

    python tools/final_sweep.py --weights <best.pt> --tag p2
"""
import sys, os, json, time, argparse
sys.path.insert(0,"tools"); sys.path.insert(0,"."); os.environ["YOLO_VERBOSE"]="False"
import numpy as np, cv2
from ultralytics import YOLO
from evalkit import load_gt, report, compute_ap, iou_matrix
from infer import run_plain
from tiled_infer import predict_tiled

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="runs/detect/runs/detect/public_s/weights/best.pt")
ap.add_argument("--split", default="val")
ap.add_argument("--device", default="0")
ap.add_argument("--tag", default="v4")
ap.add_argument("--skip-tiled", action="store_true")
a = ap.parse_args()

model = YOLO(a.weights)
gt = load_gt(a.split)
print(f"weights={a.weights}\nsplit={a.split} images={len(gt)}", flush=True)
out = []

def pr_table(preds, tag):
    """Precision/recall across the confidence range, on the full split."""
    ngt = sum(len(g["boxes"]) for g in gt.values())
    rows = []
    for s, g in gt.items():
        p = preds[s]
        if not len(p):
            continue
        M = iou_matrix(p[:, :4], g["boxes"])
        used = np.zeros(len(g["boxes"]), bool)
        for pi in np.argsort(-p[:, 4]):
            tp = False
            if M.shape[1]:
                for gi in np.argsort(-M[pi]):
                    if M[pi, gi] < 0.5: break
                    if not used[gi]:
                        used[gi] = tp = True; break
            rows.append((float(p[pi, 4]), tp))
    rows.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in rows]); n = np.arange(1, len(rows)+1)
    prec, rec = tp/n, tp/ngt
    confs = np.array([r[0] for r in rows])
    print(f"\n  --- operating points, {tag} ---")
    print(f"  {'constraint':<22}{'recall':>9}{'conf':>8}")
    best = {}
    for t in (0.95, 0.90, 0.85, 0.80, 0.70):
        idx = np.where(prec >= t)[0]
        if len(idx):
            k = idx[-1]
            print(f"  precision >= {t:<10.2f}{rec[k]:>9.3f}{confs[k]:>8.3f}")
            best[f"P>={t}"] = dict(recall=float(rec[k]), conf=float(confs[k]))
    f1 = 2*prec*rec/np.maximum(prec+rec, 1e-9)
    b = int(np.argmax(f1))
    print(f"  F1 optimum: conf {confs[b]:.3f}  P {prec[b]:.3f}  R {rec[b]:.3f}")
    best["f1_opt"] = dict(conf=float(confs[b]), P=float(prec[b]), R=float(rec[b]))
    return best

t = time.time()
p = run_plain(model, gt, imgsz=640, conf=0.001, iou=0.7, batch=32, device=a.device)
r = report("plain 640", gt, p, secs=time.time()-t)
r["operating_points"] = pr_table(p, "plain 640")
out.append(r)

if not a.skip_tiled:
    t = time.time(); preds = {}
    for i, (s, g) in enumerate(gt.items()):
        im = cv2.imread(str(g["path"]))
        b = predict_tiled(model, im, tile=320, net=640, overlap=0.25, conf=0.01,
                          device=a.device, full_imgsz=640, iou_merge=0.6,
                          full_pass=True, drop_edge=True)
        preds[s] = b.astype(np.float32) if len(b) else np.zeros((0,5), np.float32)
        if i % 500 == 0: print(f"  tiled {i}/{len(gt)}", flush=True)
    print("  (tiled pass uses conf floor 0.01, not 0.001: it makes ~10 forward\n"
          "   passes per image and the full box list will not fit comfortably in\n"
          "   RAM here. mAP50 is essentially unaffected at this floor; mAP50-95\n"
          "   may read a hair low.)")
    r = report("tiled 2x + drop_edge", gt, preds, secs=time.time()-t)
    r["operating_points"] = pr_table(preds, "tiled 2x")
    out.append(r)

fn = f"final_sweep_{a.tag}_{a.split}.json"
json.dump(out, open(fn, "w"), indent=1, default=float)
print(f"\nwrote {fn}")
