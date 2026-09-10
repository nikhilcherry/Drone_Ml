"""Does removing the speckle help the EXISTING model?

Caveat stated up front: this model was trained on speckled images, so feeding
it clean ones is a train/test mismatch that should, if anything, hurt. If it
helps anyway, the noise is costing real accuracy and a retrain on despeckled
data is worth the GPU time. If it merely breaks even, the honest reading is
'inconclusive until retrained', not 'denoising does not work'.
"""
import sys, os, argparse
sys.path.insert(0,"tools"); os.environ["YOLO_VERBOSE"]="False"
import numpy as np, cv2
from evalkit import load_gt, iou_matrix, compute_ap
from despeckle import despeckle

ap=argparse.ArgumentParser()
ap.add_argument("--src", default="rf_anti_drone")
ap.add_argument("--split", default="val")
ap.add_argument("--device", default="cpu")
ap.add_argument("--weights", default="runs/detect/runs/detect/public_s/weights/best.pt")
a=ap.parse_args()
from ultralytics import YOLO
model=YOLO(a.weights)
gt={k:v for k,v in load_gt(a.split).items() if (not a.src or v["src"]==a.src)}
stems=list(gt)
print(f"{len(stems)} images ({a.src or 'all'}, {a.split})",flush=True)

def run(mode):
    out={}
    for i in range(0,len(stems),16):
        ch=stems[i:i+16]
        ims=[cv2.imread(str(gt[s]["path"])) for s in ch]
        if mode=="despeckle":
            ims=[despeckle(im)[0] if im is not None else im for im in ims]
        res=model.predict(ims, imgsz=640, conf=0.001, iou=0.7, device=a.device, verbose=False)
        for s,r in zip(ch,res):
            out[s]=(np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()])
                    .astype(np.float32) if r.boxes is not None and len(r.boxes)
                    else np.zeros((0,5),np.float32))
        if (i//16)%15==0: print(f"  {mode} {i+len(ch)}/{len(stems)}",flush=True)
    return out

def score(preds, tag):
    ap_=compute_ap(gt,preds)
    # operating point + no-overlap rate
    tp=fp=0; nomatch=0; ngt=0
    for s,g in gt.items():
        p=preds[s]; p=p[p[:,4]>=0.25]
        M=iou_matrix(p[:,:4],g["boxes"]); ngt+=len(g["boxes"])
        used=np.zeros(len(g["boxes"]),bool); hit=np.zeros(len(p),bool)
        for pi in np.argsort(-p[:,4]) if len(p) else []:
            if M.shape[1]==0: continue
            for gi in np.argsort(-M[pi]):
                if M[pi,gi]<0.5: break
                if not used[gi]: used[gi]=hit[pi]=True; break
        tp+=int(used.sum()); fp+=int((~hit).sum())
        if M.shape[0]:
            nomatch+=int((M.max(axis=0)<0.01).sum())
        else:
            nomatch+=len(g["boxes"])
    print(f"  {tag:<12} mAP50 {ap_['mAP50']:.4f}  R {tp/max(ngt,1):.4f}  "
          f"FP {fp}  never-seen {nomatch/max(ngt,1):.1%}")
    return ap_['mAP50']

b=score(run("raw"),"raw")
d=score(run("despeckle"),"despeckle")
print(f"\n  delta mAP50: {d-b:+.4f}")
