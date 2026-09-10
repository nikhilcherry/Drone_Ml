"""Tiling A/B across BOTH size regimes, with and without edge-fragment dropping.

Dropping edge boxes must be checked on the small band too: if it also removed
genuine small detections near tile seams it would give back the +8.9 points that
made tiling worth doing.
"""
import sys, os
sys.path.insert(0,"tools"); sys.path.insert(0,"."); os.environ["YOLO_VERBOSE"]="False"
import numpy as np, cv2
from ultralytics import YOLO
from evalkit import load_gt, iou_matrix
from tiled_infer import predict_tiled

m = YOLO("runs/detect/runs/detect/public_s/weights/best.pt")
gt = load_gt("val")

def subset(pred, n):
    out=[]
    for s,g in gt.items():
        b=g["boxes"]
        if not len(b): continue
        sides=np.sqrt((b[:,2]-b[:,0])*(b[:,3]-b[:,1]))
        if pred(sides): out.append(s)
        if len(out)>=n: break
    return out

def score(sel, fn, tag):
    ngt=sum(len(gt[s]["boxes"]) for s in sel); hit=fp=0
    for s in sel:
        g=gt[s]["boxes"]; im=cv2.imread(str(gt[s]["path"])); p=fn(im)
        M=iou_matrix(p[:,:4],g) if len(p) else np.zeros((0,len(g)))
        if M.shape[0]:
            used=np.zeros(len(g),bool); ph=np.zeros(len(p),bool)
            for pi in np.argsort(-p[:,4]):
                for gi in np.argsort(-M[pi]):
                    if M[pi,gi]<0.5: break
                    if not used[gi]: used[gi]=ph[pi]=True; break
            hit+=int(used.sum()); fp+=int((~ph).sum())
    print(f"  {tag:<30} recall {hit}/{ngt} = {hit/ngt:.3f}   FP {fp}", flush=True)

def plain(im):
    r=m.predict(im, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
    return (np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()])
            if r.boxes is not None and len(r.boxes) else np.zeros((0,5),np.float32))
def tiled(drop):
    return lambda im: predict_tiled(m, im, tile=320, net=640, conf=0.25, device="cpu",
                                    full_imgsz=640, iou_merge=0.6, full_pass=True,
                                    drop_edge=drop)

for name, pred, n in (("SMALL (all boxes <=16px)", lambda s: s.max()<=16, 150),
                      ("LARGE (all boxes >96px)",  lambda s: s.min()>96,  120)):
    sel = subset(pred, n)
    print(f"\n=== {name}: {len(sel)} images ===", flush=True)
    score(sel, plain, "plain 640")
    score(sel, tiled(False), "tiled 2x, keep edge boxes")
    score(sel, tiled(True),  "tiled 2x, DROP edge boxes")
