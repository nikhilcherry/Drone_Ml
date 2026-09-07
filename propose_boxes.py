"""
Propose ONE bounding box per frame by finding the independently-moving object.

These are PROPOSALS ONLY. They are written to dataset/labels_proposed/, never
to dataset/labels/, and must be confirmed by a human in label_tool.py before
they count as labels.

Method: the camera is handheld and tracks the drone, so the background sweeps
across the frame. We estimate that background motion between neighbouring
frames of the same video, warp the neighbour onto the current frame, and
difference. Whatever did not move with the background is the candidate.

    python propose_boxes.py            # all splits
    python propose_boxes.py --splits val --limit 20
"""
import argparse, re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DATASET  = Path("dataset")
IMAGES   = DATASET / "images"
PROPOSED = DATASET / "labels_proposed"
SPLITS   = ("train", "val", "test")
EXTS     = (".jpg", ".jpeg", ".png", ".bmp")
FRAME_RE = re.compile(r"^(?P<video>.+)_frame(?P<idx>\d+)$")

PAD          = 3      # px of breathing room around the blob
MIN_AREA     = 9
MAX_AREA     = 30000
DIFF_THRESH  = 25


def video_key(p):
    m = FRAME_RE.match(p.stem)
    return (m.group("video"), int(m.group("idx"))) if m else (p.stem, 0)


def align(prev, cur):
    """Warp prev onto cur by estimating background motion. None if it fails."""
    p0 = cv2.goodFeaturesToTrack(prev, 600, 0.01, 8)
    if p0 is None or len(p0) < 12:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None,
                                         winSize=(21, 21), maxLevel=3)
    if p1 is None:
        return None
    st = st.ravel().astype(bool)
    if st.sum() < 12:
        return None
    M, inl = cv2.estimateAffinePartial2D(p0[st], p1[st], method=cv2.RANSAC,
                                         ransacReprojThreshold=3.0)
    if M is None or inl is None or inl.sum() < 10:
        return None
    return cv2.warpAffine(prev, M, (cur.shape[1], cur.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


def propose(gray_prev, gray_cur, gray_next):
    """Return (x, y, w, h, score) of the best moving blob, or None."""
    h, w = gray_cur.shape
    acc = np.zeros((h, w), np.uint8)
    used = 0
    for nb in (gray_prev, gray_next):
        if nb is None or nb.shape != gray_cur.shape:
            continue
        warped = align(nb, gray_cur)
        if warped is None:
            continue
        used += 1
        d = cv2.absdiff(gray_cur, warped)
        _, m = cv2.threshold(d, DIFF_THRESH, 255, cv2.THRESH_BINARY)
        acc = cv2.bitwise_or(acc, m) if used == 1 else cv2.bitwise_and(acc, m)
    if used == 0:
        return None

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    acc = cv2.morphologyEx(acc, cv2.MORPH_OPEN, k)
    acc = cv2.dilate(acc, k, iterations=1)
    b = 6
    acc[:b, :] = 0; acc[-b:, :] = 0; acc[:, :b] = 0; acc[:, -b:] = 0

    n, _, stats, _ = cv2.connectedComponentsWithStats(acc, 8)
    best = None
    for j in range(1, n):
        a = stats[j, cv2.CC_STAT_AREA]
        if not (MIN_AREA <= a <= MAX_AREA):
            continue
        x, y = stats[j, cv2.CC_STAT_LEFT], stats[j, cv2.CC_STAT_TOP]
        bw, bh = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
        fill = a / max(1, bw * bh)
        aspect = min(bw, bh) / max(1, max(bw, bh))
        score = a * fill * (0.5 + aspect)      # compact + blobby + not a sliver
        if best is None or score > best[4]:
            best = (x, y, bw, bh, score)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        d = IMAGES / split
        if not d.is_dir():
            continue
        files = [p for p in d.iterdir() if p.suffix.lower() in EXTS]
        by = defaultdict(list)
        for p in files:
            by[video_key(p)[0]].append(p)

        out = PROPOSED / split
        out.mkdir(parents=True, exist_ok=True)
        made = miss = 0
        for vid, vfiles in by.items():
            vfiles.sort(key=video_key)
            if args.limit:
                vfiles = vfiles[:args.limit]
            grays = [cv2.GaussianBlur(cv2.imread(str(p), 0), (5, 5), 0)
                     for p in vfiles]
            for i, p in enumerate(vfiles):
                cur = grays[i]
                if cur is None:
                    miss += 1
                    continue
                r = propose(grays[i-1] if i > 0 else None, cur,
                            grays[i+1] if i+1 < len(vfiles) else None)
                txt = out / (p.stem + ".txt")
                if r is None:
                    txt.write_text("", encoding="utf-8")   # no proposal
                    miss += 1
                    continue
                x, y, bw, bh, _ = r
                H, W = cur.shape
                x0 = max(0, x - PAD); y0 = max(0, y - PAD)
                x1 = min(W, x + bw + PAD); y1 = min(H, y + bh + PAD)
                cx = ((x0 + x1) / 2) / W; cy = ((y0 + y1) / 2) / H
                nw = (x1 - x0) / W;       nh = (y1 - y0) / H
                txt.write_text(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n",
                               encoding="utf-8")
                made += 1
        print(f"{split}: {made} proposal(s), {miss} frame(s) with none "
              f"-> {out}")
    print("\nPROPOSALS ONLY - not labels. Confirm them in label_tool.py.")


if __name__ == "__main__":
    main()
