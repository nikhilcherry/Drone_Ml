"""Separate 'cannot find it' from 'cannot localise it'.

For every GT box, the best IoU against any prediction. Recall as a function of
the IoU threshold then splits the deficit: recall at IoU 0.1 is essentially
'did the detector fire on the right object at all', recall at 0.5 is what mAP50
credits. The gap between them is pure localisation.
"""
import sys, os, json, argparse
sys.path.insert(0,"tools"); os.environ["YOLO_VERBOSE"]="False"
import numpy as np
from evalkit import load_gt, iou_matrix, band_of

ap=argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--device", default="cpu")
ap.add_argument("--weights", default="runs/detect/runs/detect/public_s/weights/best.pt")
ap.add_argument("--out", default="loc_analysis.json")
a=ap.parse_args()
from ultralytics import YOLO
model=YOLO(a.weights)
gt=load_gt(a.split)
stems=[s for s,g in gt.items() if len(g["boxes"])]
print(f"{len(stems)} images with boxes",flush=True)
rows=[]
for i in range(0,len(stems),16):
    ch=stems[i:i+16]
    res=model.predict([str(gt[s]["path"]) for s in ch], imgsz=640, conf=a.conf,
                      device=a.device, verbose=False)
    for s,r in zip(ch,res):
        g=gt[s]["boxes"]
        p=(r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes)
           else np.zeros((0,4),np.float32))
        M=iou_matrix(p,g)   # preds x gts
        for gi in range(len(g)):
            b=g[gi]; area=(b[2]-b[0])*(b[3]-b[1])
            rows.append(dict(src=gt[s]["src"], band=band_of(area),
                             side=float(np.sqrt(area)),
                             best=float(M[:,gi].max()) if M.shape[0] else 0.0))
    if (i//16)%20==0: print(f"  {i+len(ch)}/{len(stems)}",flush=True)

best=np.array([r["best"] for r in rows]); side=np.array([r["side"] for r in rows])
thrs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7]
print(f"\n=== recall vs IoU threshold ({len(rows)} GT boxes, conf>={a.conf}) ===")
print(f"  {'IoU':>5}" + "".join(f"{t:>8.1f}" for t in thrs))
print(f"  {'ALL':>5}" + "".join(f"{(best>=t).mean():>8.3f}" for t in thrs))
for band in ("small","medium","large"):
    m=np.array([r["band"]==band for r in rows])
    if m.sum(): print(f"  {band:>5}" + "".join(f"{(best[m]>=t).mean():>8.3f}" for t in thrs))
print()
for src in ("rf_anti_uav","rf_anti_drone","rf_drone_yolov7"):
    m=np.array([r["src"]==src for r in rows])
    if m.sum(): print(f"  {src:<16}" + "".join(f"{(best[m]>=t).mean():>8.3f}" for t in thrs))
print(f"\n  mean IoU of matched(>=0.1) boxes: {best[best>=0.1].mean():.3f}")
print(f"  GT boxes with NO overlapping prediction at all: {(best<0.01).mean():.1%}")
json.dump(rows, open(a.out,"w"))
print(f"wrote {a.out}")
