"""Grid-render images (optionally with GT boxes) so they can actually be looked at."""
import sys, argparse, random
from pathlib import Path
import cv2, numpy as np

def build(paths, labels_dir, out, cols=6, cell=260, draw_gt=True, title=True):
    rows = (len(paths)+cols-1)//cols
    canvas = np.full((rows*cell, cols*cell, 3), 30, np.uint8)
    for i,p in enumerate(paths):
        im = cv2.imread(str(p))
        if im is None: continue
        H,W = im.shape[:2]
        if draw_gt and labels_dir:
            lf = Path(labels_dir)/(p.stem+".txt")
            if lf.exists():
                for line in lf.read_text().split("\n"):
                    if not line.strip(): continue
                    _,cx,cy,nw,nh = (float(v) for v in line.split()[:5])
                    x1,y1 = int((cx-nw/2)*W), int((cy-nh/2)*H)
                    x2,y2 = int((cx+nw/2)*W), int((cy+nh/2)*H)
                    cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),max(2,W//320))
        s = cell/max(H,W)
        im = cv2.resize(im,(int(W*s),int(H*s)))
        h,w = im.shape[:2]
        r,c = divmod(i,cols)
        canvas[r*cell:r*cell+h, c*cell:c*cell+w] = im
        if title:
            cv2.putText(canvas,str(i),(c*cell+4,r*cell+16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
    cv2.imwrite(out, canvas)
    print(f"wrote {out}  ({len(paths)} images, {rows}x{cols})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--filter", default="")
    ap.add_argument("--negatives", action="store_true")
    ap.add_argument("--positives", action="store_true")
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path("dataset_public")
    imgs = root/"images"/a.split; labs = root/"labels"/a.split
    sel = []
    for p in sorted(imgs.iterdir()):
        if a.filter and a.filter not in p.name: continue
        lf = labs/(p.stem+".txt")
        nb = len([l for l in lf.read_text().split("\n") if l.strip()]) if lf.exists() else 0
        if a.negatives and nb>0: continue
        if a.positives and nb==0: continue
        sel.append(p)
    random.seed(a.seed); random.shuffle(sel)
    build(sel[:a.n], labs, a.out)
