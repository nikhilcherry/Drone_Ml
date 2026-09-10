"""Does magnified tiling COST anything on large drones?

Slicing helps the small band (see tile_hard.py). The risk is the mirror image
of the resolution sweep: a 380 px studio drone does not fit in a 320 px crop,
so if the merged output leaned on tiles it would wreck the large band. The
full-frame pass inside predict_tiled exists to prevent that; this checks it.
"""
import sys, os
sys.path.insert(0,"tools"); sys.path.insert(0,"."); os.environ["YOLO_VERBOSE"]="False"
import numpy as np, cv2
from ultralytics import YOLO
from evalkit import load_gt, iou_matrix
from tiled_infer import predict_tiled

m = YOLO("runs/detect/runs/detect/public_s/weights/best.pt")
gt = load_gt("val")
sel = []
for s, g in gt.items():
    b = g["boxes"]
    if not len(b): continue
    sides = np.sqrt((b[:,2]-b[:,0])*(b[:,3]-b[:,1]))
    if sides.min() > 96: sel.append(s)
sel = sel[:120]
ngt = sum(len(gt[s]["boxes"]) for s in sel)
print(f"{len(sel)} images, {ngt} GT boxes, all >96px", flush=True)

def score(fn, tag):
    hit = fp = 0
    for s in sel:
        g = gt[s]["boxes"]; im = cv2.imread(str(gt[s]["path"]))
        p = fn(im)
        M = iou_matrix(p[:,:4], g) if len(p) else np.zeros((0,len(g)))
        if M.shape[0]:
            used=np.zeros(len(g),bool); ph=np.zeros(len(p),bool)
            for pi in np.argsort(-p[:,4]):
                for gi in np.argsort(-M[pi]):
                    if M[pi,gi]<0.5: break
                    if not used[gi]: used[gi]=ph[pi]=True; break
            hit+=int(used.sum()); fp+=int((~ph).sum())
    print(f"  {tag:<26} recall {hit}/{ngt} = {hit/ngt:.3f}   FP {fp}", flush=True)

def plain(im):
    r=m.predict(im, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
    return (np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()])
            if r.boxes is not None and len(r.boxes) else np.zeros((0,5),np.float32))

score(plain, "plain 640")
score(lambda im: predict_tiled(m, im, tile=320, net=640, conf=0.25, device="cpu",
      full_imgsz=640, iou_merge=0.6, full_pass=True), "tiled 2x + full frame")
score(lambda im: predict_tiled(m, im, tile=320, net=640, conf=0.25, device="cpu",
      full_imgsz=640, iou_merge=0.6, full_pass=False), "tiled 2x, NO full frame")
score(lambda im: predict_tiled(m, im, tile=320, net=640, conf=0.25, device="cpu",
      full_imgsz=640, iou_merge=0.6, full_pass=True, drop_edge=True),
      "tiled 2x + full + dropedge")
