"""Exact duplicate check between splits - no thumbnails, no thresholds.

Thumbnail RMSE can be argued with. A hash of the decoded pixel array cannot:
if the same array appears in train and in test, that image is in both splits.
"""
import sys, hashlib, collections
from pathlib import Path
import cv2, numpy as np

R = Path("dataset_public/images")
def hashes(split):
    h = {}
    for p in sorted((R/split).iterdir()):
        im = cv2.imread(str(p))
        if im is None: continue
        h.setdefault(hashlib.md5(np.ascontiguousarray(im)).hexdigest(), []).append(p)
    return h

tr = hashes("train"); print(f"train: {sum(len(v) for v in tr.values())} images, "
                            f"{len(tr)} distinct pixel arrays", flush=True)
for split in ("val", "test"):
    ev = hashes(split)
    n = sum(len(v) for v in ev.values())
    shared = set(ev) & set(tr)
    leaked = sum(len(ev[k]) for k in shared)
    print(f"\n{split}: {n} images, {len(ev)} distinct")
    print(f"  pixel-identical to a TRAIN image: {leaked}/{n} = {leaked/n:.2%} "
          f"({len(shared)} distinct arrays)")
    # internal duplication within the split
    dup = sum(len(v)-1 for v in ev.values() if len(v) > 1)
    print(f"  duplicated WITHIN {split}: {dup} redundant copies")
    for k in list(shared)[:3]:
        print(f"    e.g. {ev[k][0].name[:66]}")
        print(f"      == {tr[k][0].name[:66]}")
