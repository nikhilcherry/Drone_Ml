"""Where do the false positives on a given source actually come from?"""
import sys, os, json, argparse
sys.path.insert(0,"tools"); os.environ["YOLO_VERBOSE"]="False"
from pathlib import Path
import numpy as np
from evalkit import load_gt, iou_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="rf_anti_drone")
ap.add_argument("--split", default="val")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--device", default="cpu")
ap.add_argument("--weights", default="runs/detect/runs/detect/public_s/weights/best.pt")
a = ap.parse_args()

from ultralytics import YOLO
model = YOLO(a.weights)
gt = {k:v for k,v in load_gt(a.split).items() if v["src"]==a.src}
print(f"{len(gt)} {a.src} images in {a.split}", flush=True)

stems=list(gt); recs=[]; tot_fp=0; tot_gt=0; tot_tp=0
neg_fp=0; pos_fp=0
for i in range(0,len(stems),16):
    ch=stems[i:i+16]
    res=model.predict([str(gt[s]["path"]) for s in ch], imgsz=640, conf=a.conf,
                      device=a.device, verbose=False)
    for s,r in zip(ch,res):
        g=gt[s]["boxes"]; tot_gt+=len(g)
        p=(np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()])
           if r.boxes is not None and len(r.boxes) else np.zeros((0,5),np.float32))
        M=iou_matrix(p[:,:4],g)
        used=np.zeros(len(g),bool); hit=np.zeros(len(p),bool)
        for pi in np.argsort(-p[:,4]) if len(p) else []:
            if M.shape[1]==0: continue
            for gi in np.argsort(-M[pi]):
                if M[pi,gi]<0.5: break
                if not used[gi]: used[gi]=hit[pi]=True; break
        nfp=int((~hit).sum()); tot_fp+=nfp; tot_tp+=int(used.sum())
        if nfp:
            (neg_fp if len(g)==0 else pos_fp).__class__  # noop
            if len(g)==0: neg_fp+=nfp
            else: pos_fp+=nfp
            for pi in np.where(~hit)[0]:
                x1,y1,x2,y2,c=p[pi]
                recs.append(dict(stem=s, path=str(gt[s]["path"]), conf=float(c),
                                 side=float(np.sqrt(max(0,(x2-x1)*(y2-y1)))),
                                 n_gt=len(g),
                                 best_iou=float(M[pi].max()) if M.shape[1] else 0.0))
    if (i//16)%12==0: print(f"  {i+len(ch)}/{len(stems)}",flush=True)

recs.sort(key=lambda r:-r["conf"])
print(f"\nGT {tot_gt}  TP {tot_tp}  recall {tot_tp/max(tot_gt,1):.3f}  FP {tot_fp}")
print(f"  FP on images WITH gt boxes : {pos_fp}")
print(f"  FP on negative images      : {neg_fp}")
imgs_with_fp = len({r['stem'] for r in recs})
print(f"  FPs spread over {imgs_with_fp} images")
near = sum(1 for r in recs if 0.1 <= r['best_iou'] < 0.5)
print(f"  FPs that overlap a GT box but miss IoU 0.5 (localisation, not hallucination): {near}")
sides=np.array([r['side'] for r in recs]) if recs else np.zeros(1)
print(f"  FP box side px: med {np.median(sides):.1f}  p90 {np.percentile(sides,90):.1f}")
json.dump(recs, open(f"fp_{a.src}_{a.split}.json","w"), indent=1)
print(f"wrote fp_{a.src}_{a.split}.json")
