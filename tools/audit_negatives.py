"""
Model-assisted audit of the images the dataset calls negatives.

Circularity warning, stated up front: using the detector to find labelling
errors and then reporting the detector's score on the corrected labels would
be self-serving. This script only RANKS negatives by detection confidence so a
human can look at them; nothing is relabelled automatically, and the visual
check is the evidence, not the confidence.
"""
import sys, json, os, argparse
sys.path.insert(0, "tools")
os.environ["YOLO_VERBOSE"] = "False"
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--conf", type=float, default=0.30)
ap.add_argument("--weights", default="runs/detect/runs/detect/public_s/weights/best.pt")
ap.add_argument("--out", default="negative_audit.json")
a = ap.parse_args()

from ultralytics import YOLO
model = YOLO(a.weights)
root = Path("dataset_public")
negs = []
for sp in ("train", "val", "test"):
    for p in sorted((root/"images"/sp).iterdir()):
        lf = root/"labels"/sp/(p.stem+".txt")
        nb = len([l for l in lf.read_text().split("\n") if l.strip()]) if lf.exists() else 0
        if nb == 0:
            negs.append((sp, p))
print(f"{len(negs)} negative images across all splits", flush=True)

rows = []
B = 16
for i in range(0, len(negs), B):
    chunk = negs[i:i+B]
    res = model.predict([str(p) for _, p in chunk], imgsz=640, conf=a.conf,
                        device=a.device, verbose=False)
    for (sp, p), r in zip(chunk, res):
        if r.boxes is None or not len(r.boxes):
            continue
        cf = r.boxes.conf.cpu().numpy()
        xy = r.boxes.xyxy.cpu().numpy()
        H, W = r.orig_shape
        k = int(cf.argmax())
        area = (xy[k,2]-xy[k,0])*(xy[k,3]-xy[k,1])
        rows.append(dict(split=sp, path=str(p), n=len(cf),
                         max_conf=float(cf.max()),
                         side=float(np.sqrt(area)),
                         frac=float(area/(W*H))))
    if (i//B) % 8 == 0:
        print(f"  {i+len(chunk)}/{len(negs)}", flush=True)

rows.sort(key=lambda r: -r["max_conf"])
by_split = {}
for sp in ("train","val","test"):
    tot = sum(1 for s,_ in negs if s==sp)
    hi  = sum(1 for r in rows if r["split"]==sp and r["max_conf"]>=0.5)
    any_ = sum(1 for r in rows if r["split"]==sp)
    by_split[sp] = dict(total=tot, any_det=any_, conf50=hi)
print(json.dumps(by_split, indent=1))
json.dump(dict(by_split=by_split, rows=rows), open(a.out,"w"), indent=1)
print(f"wrote {a.out}; {len(rows)} negatives have a detection >= {a.conf}")
