"""GT box size distribution per split and per source, in pixels."""
import sys, collections
from pathlib import Path
import numpy as np
import cv2

ROOT = Path("dataset_public")
SOURCES = ["rf_anti_uav", "rf_anti_drone", "rf_drone_yolov7"]

def source_of(name):
    n = name[4:] if name.startswith("neg_") else name
    for s in SOURCES:
        if n.startswith(s):
            return s
    return "unknown"

def main(split):
    imgs = ROOT / "images" / split
    labs = ROOT / "labels" / split
    # image size cache by (source) - but sizes vary, so read each label's image dims lazily
    per_src = collections.defaultdict(list)
    allw = []
    n_img = n_box = 0
    for lf in sorted(labs.glob("*.txt")):
        lines = [l for l in lf.read_text().split("\n") if l.strip()]
        if not lines:
            continue
        ip = None
        for ext in (".jpg", ".jpeg", ".png"):
            c = imgs / (lf.stem + ext)
            if c.exists():
                ip = c; break
        if ip is None:
            continue
        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]
        n_img += 1
        src = source_of(ip.name)
        for line in lines:
            parts = line.split()
            _, cx, cy, nw, nh = (float(v) for v in parts[:5])
            w, h = nw * W, nh * H
            per_src[src].append((w, h))
            allw.append((w, h))
            n_box += 1
    print(f"=== split={split}  images_with_boxes={n_img}  boxes={n_box} ===")
    def report(tag, arr):
        a = np.array(arr)
        if not len(a): return
        w, h = a[:, 0], a[:, 1]
        side = np.sqrt(w * h)           # equivalent square side
        area = w * h
        small = (area < 32*32).mean()
        med   = ((area >= 32*32) & (area < 96*96)).mean()
        large = (area >= 96*96).mean()
        tiny  = (area < 16*16).mean()
        print(f"  {tag:<18} n={len(a):>6}  side px: p10={np.percentile(side,10):5.1f} "
              f"med={np.median(side):5.1f} p90={np.percentile(side,90):6.1f}  "
              f"| <16px={tiny:5.1%} small={small:5.1%} med={med:5.1%} large={large:5.1%}")
    report("ALL", allw)
    for s in SOURCES:
        report(s, per_src[s])

for sp in (sys.argv[1:] or ["train", "val", "test"]):
    main(sp)
