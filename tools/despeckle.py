"""
Targeted salt-and-pepper removal that keeps 1-2 px drones.

A blanket 3x3 median erases distant drones along with the noise (verified
visually: at <=14 px the drone disappears). Salt-and-pepper here is different
from a small drone in two ways a filter can exploit:
  * it saturates - the specks are near 0 or near 255, drones are mid-tone
  * it is achromatic and isolated - a speck's colour channels move together to
    an extreme, and its neighbours are untouched background
So: replace a pixel with the local median ONLY where it is both an extreme
value and a large deviation from that median. Everything else is left alone.
"""
import numpy as np, cv2

def despeckle(img, dev=55, lo=42, hi=213):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(img, 3)
    gmed = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    d = cv2.absdiff(g, gmed)
    extreme = (g <= lo) | (g >= hi)
    mask = (d > dev) & extreme
    out = img.copy()
    out[mask] = med[mask]
    return out, mask

if __name__ == "__main__":
    import sys, random
    sys.path.insert(0, "tools")
    from evalkit import load_gt
    gt = load_gt("val")
    # how much speck is removed vs how much drone signal is touched?
    items = [(g["path"], g["boxes"]) for s, g in gt.items()
             if g["src"] == "rf_anti_drone" and len(g["boxes"])]
    random.seed(0); random.shuffle(items)
    tot_mask = tot_px = in_box = box_px = 0
    for p, boxes in items[:200]:
        im = cv2.imread(str(p))
        if im is None: continue
        _, mask = despeckle(im)
        H, W = mask.shape
        tot_mask += int(mask.sum()); tot_px += mask.size
        for b in boxes:
            x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
            x2, y2 = min(W, int(b[2])), min(H, int(b[3]))
            if x2 > x1 and y2 > y1:
                in_box += int(mask[y1:y2, x1:x2].sum())
                box_px += (x2-x1)*(y2-y1)
    print(f"pixels altered overall : {tot_mask/tot_px:.4%}")
    print(f"pixels altered inside GT boxes: {in_box/max(box_px,1):.4%}")
    print(f"  -> a filter that hit drones and background equally would show"
          f" the same rate in both rows")
