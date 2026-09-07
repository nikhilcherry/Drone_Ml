"""
Split TEST recall by background type (green field vs sky/building).
Confirms whether V1's misses are explained by unseen backgrounds.

    python recall_by_background.py
"""
from pathlib import Path
import cv2, numpy as np

def find_weights():
    h = sorted(Path(".").rglob("drone_v1/weights/best.pt")) or sorted(Path(".").rglob("best.pt"))
    if not h: raise SystemExit("best.pt not found")
    return h[0]

def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0.,ix2-ix1),max(0.,iy2-iy1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.

def green_frac(im,box):
    x1,y1,x2,y2=[int(v) for v in box]; H,W=im.shape[:2]
    pad=int(max(x2-x1,y2-y1)*1.2)
    X1,Y1=max(0,x1-pad),max(0,y1-pad); X2,Y2=min(W,x2+pad),min(H,y2+pad)
    ring=im[Y1:Y2,X1:X2]
    if ring.size==0: return 0.
    m=np.ones(ring.shape[:2],bool); m[max(0,y1-Y1):y2-Y1,max(0,x1-X1):x2-X1]=False
    hsv=cv2.cvtColor(ring,cv2.COLOR_BGR2HSV)
    h,s=hsv[...,0][m],hsv[...,1][m]
    return float(np.mean((h>=35)&(h<=85)&(s>60))) if h.size else 0.

from ultralytics import YOLO
model=YOLO(str(find_weights()))
stats={"green field":[0,0,[]], "sky / building":[0,0,[]]}
for p in sorted(Path("dataset/images/test").iterdir()):
    lf=Path("dataset/labels/test")/(p.stem+".txt")
    if not lf.exists() or not lf.read_text().strip(): continue
    im=cv2.imread(str(p)); H,W=im.shape[:2]
    r=model.predict(source=str(p),imgsz=640,device="0",conf=0.25,verbose=False)[0]
    preds=[tuple(b) for b in r.boxes.xyxy.cpu().numpy()] if r.boxes is not None else []
    for line in lf.read_text().splitlines():
        if not line.strip(): continue
        _,cx,cy,nw,nh=[float(v) for v in line.split()]
        g=((cx-nw/2)*W,(cy-nh/2)*H,(cx+nw/2)*W,(cy+nh/2)*H)
        k="green field" if green_frac(im,g)>0.4 else "sky / building"
        best=max((iou(g,q) for q in preds),default=0.)
        stats[k][1]+=1; stats[k][2].append(best)
        if best>=0.5: stats[k][0]+=1
print(f"{'background':<18}{'GT':>5}{'found':>7}{'recall':>9}{'mean IoU':>10}")
for k,(hit,tot,ious) in stats.items():
    print(f"{k:<18}{tot:>5}{hit:>7}{(hit/tot if tot else 0):>9.3f}{np.mean(ious) if ious else 0:>10.3f}")
