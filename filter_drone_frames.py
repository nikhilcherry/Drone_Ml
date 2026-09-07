"""
Filter drone dataset frames: keep frames containing a drone, quarantine the rest.

Nothing is ever deleted. Frames are moved into sibling folders:
    dataset/review/<split>/    uncertain  - look at these yourself
    dataset/removed/<split>/   no evidence of a drone

Two independent signals are combined:
  1. YOLO (COCO-pretrained) run on overlapping tiles at high resolution, with a
     low confidence threshold, looking for drone-like proxy classes.
  2. Temporal motion: frames are grouped back into their source videos using the
     filename, and consecutive frames are differenced to find a small compact
     moving blob - which is what a flying drone looks like.

A frame is only sent to removed/ when BOTH signals see nothing.

Usage:
    pip install ultralytics opencv-python
    python filter_drone_frames.py                 # dry run, prints the summary
    python filter_drone_frames.py --apply         # actually move files
    python filter_drone_frames.py --restore       # move everything back
    python filter_drone_frames.py --apply --model yolov8l.pt   # faster, less accurate
"""

import argparse
import csv
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# --------------------------- settings ---------------------------------
DATASET      = Path("dataset")
IMAGES_DIR   = DATASET / "images"
REVIEW_DIR   = DATASET / "review"
REMOVED_DIR  = DATASET / "removed"
SPLITS       = ("train", "val", "test")

# COCO classes a small quadcopter plausibly fires. There is no "drone" class.
PROXY_CLASSES = {"airplane", "bird", "kite", "frisbee", "sports ball", "boat"}

CONF_KEEP    = 0.10   # a proxy hit at/above this = keep the frame
CONF_REVIEW  = 0.04   # a weaker hit = send to review, never to removed
TILE_GRID    = (2, 3) # rows, cols of overlapping tiles per frame
TILE_OVERLAP = 0.25   # fraction of tile size shared with the neighbour
IMGSZ        = 1280   # each tile is upscaled to this before inference

# Temporal motion signal (a drone-sized moving blob)
MOTION_MIN_AREA = 12      # px^2 - deliberately tiny, far drones are a few px
MOTION_MAX_AREA = 20000   # px^2 - larger than this is camera pan / a person
MOTION_KEEP     = 1.5     # blob "droneness" score that counts as keep
MOTION_REVIEW   = 0.35    # anything above this at least goes to review

# Temporal guard: a drone does not teleport. If a neighbouring frame of the same
# video (within this many sampled frames) clearly has a drone, this frame is
# never auto-removed - the worst it can get is "review".
NEIGHBOUR_GUARD = 2
# ----------------------------------------------------------------------

FRAME_RE = re.compile(r"^(?P<video>.+)_frame(?P<idx>\d+)$")


def parse_name(path):
    """extract_frames.py wrote '<video>_frame000123.jpg'."""
    m = FRAME_RE.match(path.stem)
    if not m:
        return path.stem, 0
    return m.group("video"), int(m.group("idx"))


def tiles_for(w, h):
    rows, cols = TILE_GRID
    th, tw = h / rows, w / cols
    oh, ow = th * TILE_OVERLAP, tw * TILE_OVERLAP
    boxes = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * tw - ow))
            y1 = max(0, int(r * th - oh))
            x2 = min(w, int((c + 1) * tw + ow))
            y2 = min(h, int((r + 1) * th + oh))
            boxes.append((x1, y1, x2, y2))
    boxes.append((0, 0, w, h))   # plus the whole frame, for close-up drones
    return boxes


# ------------------------- signal 1: YOLO -----------------------------
def yolo_score(model, img, names):
    """Best confidence among drone-like proxy classes, over all tiles."""
    h, w = img.shape[:2]
    crops, best, hits = [], 0.0, []
    for (x1, y1, x2, y2) in tiles_for(w, h):
        crops.append(img[y1:y2, x1:x2])
    results = model.predict(crops, imgsz=IMGSZ, conf=CONF_REVIEW,
                            verbose=False, max_det=50)
    for res in results:
        if res.boxes is None:
            continue
        for cls_id, conf in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            label = names[int(cls_id)]
            if label in PROXY_CLASSES and conf > best:
                best, hits = conf, [label]
            elif label in PROXY_CLASSES and conf == best:
                hits.append(label)
    return best, (hits[0] if hits else "")


# ---------------------- signal 2: temporal motion ---------------------
def motion_scores(paths):
    """
    paths: frames of ONE video, in time order.
    Returns {path: score}. Score rewards a small, compact, isolated moving blob
    and punishes the whole-frame change produced by camera pans.
    """
    scores = {p: 0.0 for p in paths}
    grays = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        grays.append(None if img is None else cv2.GaussianBlur(img, (5, 5), 0))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for i, p in enumerate(paths):
        cur = grays[i]
        if cur is None:
            scores[p] = MOTION_KEEP   # unreadable -> never auto-remove
            continue
        neighbours = [g for g in (grays[i - 1] if i > 0 else None,
                                  grays[i + 1] if i + 1 < len(paths) else None)
                      if g is not None and g.shape == cur.shape]
        if not neighbours:
            scores[p] = MOTION_KEEP
            continue

        best = 0.0
        for nb in neighbours:
            diff = cv2.absdiff(cur, nb)
            _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            changed = mask.mean() / 255.0          # 0..1 whole-frame change
            if changed > 0.35:
                continue                            # camera pan, unusable
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            for j in range(1, n):
                area = stats[j, cv2.CC_STAT_AREA]
                if not (MOTION_MIN_AREA <= area <= MOTION_MAX_AREA):
                    continue
                bw, bh = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
                fill = area / max(1, bw * bh)       # compact blob, not a smear
                aspect = min(bw, bh) / max(1, max(bw, bh))
                s = 4.0 * fill * aspect * (1.0 - min(1.0, changed / 0.35))
                best = max(best, s)
        scores[p] = best
    return scores


# ------------------------------ main ----------------------------------
def decide(yolo_conf, motion, neighbour_has_drone):
    if yolo_conf >= CONF_KEEP:
        return "keep", f"yolo {yolo_conf:.2f}"
    if motion >= MOTION_KEEP:
        return "keep", f"motion {motion:.2f}"
    if yolo_conf >= CONF_REVIEW or motion >= MOTION_REVIEW:
        return "review", f"weak (yolo {yolo_conf:.2f}, motion {motion:.2f})"
    if neighbour_has_drone:
        return "review", "no evidence, but a neighbouring frame has a drone"
    return "removed", "no evidence"


def restore():
    moved = 0
    for src_root in (REVIEW_DIR, REMOVED_DIR):
        if not src_root.is_dir():
            continue
        for split in SPLITS:
            d = src_root / split
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(IMAGES_DIR / split / f.name))
                    moved += 1
    print(f"Restored {moved} file(s) to dataset/images/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually move files")
    ap.add_argument("--restore", action="store_true", help="undo a previous run")
    ap.add_argument("--model", default="yolov8x.pt", help="ultralytics weights")
    ap.add_argument("--report", default="filter_report.csv")
    ap.add_argument("--no-yolo", action="store_true",
                    help="skip YOLO entirely and use only the motion signal "
                         "(seconds instead of hours on CPU)")
    ap.add_argument("--splits", default=",".join(SPLITS),
                    help="comma-separated subset of splits to process")
    args = ap.parse_args()

    if args.restore:
        restore()
        return

    if not IMAGES_DIR.is_dir():
        sys.exit(f"Not found: {IMAGES_DIR.resolve()}  (run this from the folder "
                 f"that contains dataset/)")

    model = names = None
    if args.no_yolo:
        print("Running with the motion signal only (--no-yolo).")
    else:
        from ultralytics import YOLO
        print(f"Loading {args.model} (downloads once on first run)...")
        model = YOLO(args.model)
        names = model.names
        try:
            import torch
            dev = "GPU" if torch.cuda.is_available() else "CPU (this will be slow)"
            print(f"Inference device: {dev}")
        except Exception:
            pass

    wanted_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    rows, totals = [], {}
    for split in wanted_splits:
        split_dir = IMAGES_DIR / split
        if not split_dir.is_dir():
            continue
        files = sorted(p for p in split_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
        if not files:
            continue

        # group by source video, in frame order, for the temporal signal
        by_video = defaultdict(list)
        for p in files:
            by_video[parse_name(p)[0]].append(p)
        motion = {}
        for vid, vfiles in by_video.items():
            vfiles.sort(key=lambda p: parse_name(p)[1])
            motion.update(motion_scores(vfiles))

        counts = defaultdict(int)
        print(f"\n{split.upper()}: scanning {len(files)} frames "
              f"from {len(by_video)} video(s)")

        # --- pass A: score every frame ---
        conf_of, label_of = {}, {}
        for n, p in enumerate(files, 1):
            if model is None:
                conf_of[p], label_of[p] = 0.0, ""      # motion signal only
            else:
                img = cv2.imread(str(p))
                if img is None:
                    conf_of[p], label_of[p] = -1.0, ""  # unreadable -> review
                else:
                    conf_of[p], label_of[p] = yolo_score(model, img, names)
            if n % 25 == 0 or n == len(files):
                print(f"   {n}/{len(files)}", end="\r", flush=True)
        print(" " * 30, end="\r")

        # --- pass B: decide, using neighbouring frames of the same video ---
        strong = {p: (conf_of[p] >= CONF_KEEP or motion.get(p, 0.0) >= MOTION_KEEP)
                  for p in files}
        verdict_of, reason_of = {}, {}
        for vfiles in by_video.values():
            for i, p in enumerate(vfiles):
                lo = max(0, i - NEIGHBOUR_GUARD)
                hi = min(len(vfiles), i + NEIGHBOUR_GUARD + 1)
                near = any(strong[q] for q in vfiles[lo:hi] if q is not p)
                if conf_of[p] < 0:
                    verdict, why = "review", "unreadable file"
                else:
                    verdict, why = decide(conf_of[p], motion.get(p, 0.0), near)
                    if label_of[p] and conf_of[p] >= CONF_KEEP:
                        why += f" ({label_of[p]})"
                verdict_of[p], reason_of[p] = verdict, why
                counts[verdict] += 1

        for p in files:
            rows.append([split, p.name, verdict_of[p],
                         f"{max(conf_of[p], 0.0):.3f}",
                         f"{motion.get(p, 0.0):.2f}", reason_of[p]])

        if args.apply:
            for split_name, fname, verdict, *_ in rows:
                if split_name != split or verdict == "keep":
                    continue
                dest_root = REVIEW_DIR if verdict == "review" else REMOVED_DIR
                dest = dest_root / split
                dest.mkdir(parents=True, exist_ok=True)
                src = split_dir / fname
                if src.exists():
                    shutil.move(str(src), str(dest / fname))   # file unmodified

        totals[split] = (len(files), counts["keep"], counts["review"],
                         counts["removed"])

    with open(args.report, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "filename", "verdict", "yolo_conf",
                    "motion_score", "reason"])
        w.writerows(rows)

    print()
    for split in SPLITS:
        if split not in totals:
            continue
        orig, keep, rev, rem = totals[split]
        print(f"{split.upper()}:")
        print(f"  original: {orig}")
        print(f"  kept:     {keep}")
        print(f"  review:   {rev}")
        print(f"  removed:  {rem}")
        print()

    if args.apply:
        print(f"Uncertain frames -> {REVIEW_DIR}/<split>/")
        print(f"No-drone frames  -> {REMOVED_DIR}/<split>/   (nothing deleted)")
        print("Undo with:  python filter_drone_frames.py --restore")
    else:
        print("DRY RUN - no files were moved. Re-run with --apply to move them.")
    print(f"Per-frame detail written to {args.report}")


if __name__ == "__main__":
    main()
