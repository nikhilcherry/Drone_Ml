"""Crop around GT boxes and blow them up, to judge whether the labels are tight."""
import sys, random, argparse
sys.path.insert(0,"tools")
from pathlib import Path
import cv2, numpy as np
from evalkit import load_gt

ap=argparse.ArgumentParser()
ap.add_argument("--split",default="val"); ap.add_argument("--src",default="rf_anti_drone")
ap.add_argument("--n",type=int,default=30); ap.add_argument("--cell",type=int,default=200)
ap.add_argument("--pad",type=float,default=2.5); ap.add_argument("--maxside",type=float,default=40)
ap.add_argument("--out",required=True); ap.add_argument("--seed",type=int,default=1)
a=ap.parse_args()
gt=load_gt(a.split)
items=[]
for s,g in gt.items():
    if a.src and g["src"]!=a.src: continue
    for b in g["boxes"]:
        side=float(np.sqrt(max(1,(b[2]-b[0])*(b[3]-b[1]))))
        if side<=a.maxside: items.append((g["path"],b,side))
random.seed(a.seed); random.shuffle(items)
items=items[:a.n]
cols=6; rows=(len(items)+cols-1)//cols; C=a.cell
canvas=np.full((rows*C,cols*C,3),25,np.uint8)
for i,(p,b,side) in enumerate(items):
    im=cv2.imread(str(p))
    if im is None: continue
    H,W=im.shape[:2]
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
    half=max(12, side*a.pad)
    x1,y1=int(max(0,cx-half)),int(max(0,cy-half))
    x2,y2=int(min(W,cx+half)),int(min(H,cy+half))
    crop=im[y1:y2,x1:x2].copy()
    if crop.size==0: continue
    sc=C/max(crop.shape[:2])
    crop=cv2.resize(crop,(int(crop.shape[1]*sc),int(crop.shape[0]*sc)),interpolation=cv2.INTER_NEAREST)
    bx1,by1=int((b[0]-x1)*sc),int((b[1]-y1)*sc)
    bx2,by2=int((b[2]-x1)*sc),int((b[3]-y1)*sc)
    cv2.rectangle(crop,(bx1,by1),(bx2,by2),(0,0,255),1)
    r,c=divmod(i,cols); h,w=crop.shape[:2]
    canvas[r*C:r*C+h,c*C:c*C+w]=crop
    cv2.putText(canvas,f"{side:.0f}px",(c*C+3,r*C+14),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)
cv2.imwrite(a.out,canvas); print(f"wrote {a.out} ({len(items)} boxes)")
