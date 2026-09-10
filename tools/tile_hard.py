"""Does magnified tiling help where it should - the sub-16px band?

The earlier 40-image check saturated at 1.000 recall for every config, which
measures nothing. This selects only images whose GT boxes are ALL small, so
there is headroom for a difference to show up.
"""
import sys, os
sys.path.insert(0,"tools"); sys.path.insert(0,"."); os.environ["YOLO_VERBOSE"]="False"
import numpy as np, cv2
from ultralytics import YOLO
from evalkit import load_gt, iou_matrix
from tiled_infer import predict_tiled

m = YOLO("runs/detect/runs/detect/public_s/weights/best.pt")
gt = load_gt("val")
hard = []
for s, g in gt.items():
    b = g["boxes"]
    if not len(b): continue
    sides = np.sqrt((b[:,2]-b[:,0])*(b[:,3]-b[:,1]))
    if sides.max() <= 16: hard.append(s)
hard = hard[:180]
ngt = sum(len(gt[s]["boxes"]) for s in hard)
print(f"{len(hard)} images, {ngt} GT boxes, all <=16px", flush=True)

def score(fn, tag):
    hit = fp = 0
    for s in hard:
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

def plain(imgsz):
    def f(im):
        r=m.predict(im, imgsz=imgsz, conf=0.25, device="cpu", verbose=False)[0]
        return (np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()])
                if r.boxes is not None and len(r.boxes) else np.zeros((0,5),np.float32))
    return f

score(plain(640), "plain 640")
for tile, net in ((320,640),(320,960)):
    score(lambda im,t=tile,n=net: predict_tiled(m, im, tile=t, net=n, conf=0.25,
          device="cpu", full_imgsz=640, iou_merge=0.6), f"tiled {tile}->{net} ({net//tile}x)")
