"""
Fast keyboard review of drone dataset frames.

Shows every frame full-screen-ish and lets you mark it in one keypress.
Nothing is deleted: marked frames move to dataset/removed/<split>/ on quit,
and dataset/images/<split>/ keeps everything else. Progress is saved, so you
can quit half way and pick up where you left off.

    KEYS
      K  /  ->      keep this frame, next
      X  /  Del     mark as NO DRONE, next
      J  /  <-      go back one frame
      Z             undo the mark on the current frame
      +  /  -       zoom in / out (display only, files are never modified)
      F             toggle 1:1 pixel view at the cursor-free centre
      S             save progress now
      Q  /  Esc     save and quit (then choose whether to move files)

Usage:
    pip install opencv-python
    python review_frames.py                 # all splits
    python review_frames.py --splits test   # one split
    python review_frames.py --order report  # hardest-first, using filter_report.csv
"""

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import cv2

DATASET     = Path("dataset")
IMAGES_DIR  = DATASET / "images"
REMOVED_DIR = DATASET / "removed"
SPLITS      = ("train", "val", "test")
PROGRESS    = Path("review_progress.json")
EXTS        = (".jpg", ".jpeg", ".png", ".bmp")
WIN         = "drone review  |  K=keep  X=no drone  J=back  Z=undo  Q=quit"

FRAME_RE = re.compile(r"^(?P<video>.+)_frame(?P<idx>\d+)$")


def sort_key(path):
    m = FRAME_RE.match(path.stem)
    return (m.group("video"), int(m.group("idx"))) if m else (path.stem, 0)


def collect(splits, order, report):
    """Return [(split, Path)] in review order."""
    items = []
    for split in splits:
        d = IMAGES_DIR / split
        if not d.is_dir():
            print(f"  (skipping {split}, no such folder)")
            continue
        items += [(split, p) for p in d.iterdir() if p.suffix.lower() in EXTS]

    if order == "report" and Path(report).is_file():
        score = {}
        with open(report, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    score[(row["split"], row["filename"])] = float(row["motion_score"])
                except (KeyError, ValueError):
                    pass
        # lowest score first: the frames most likely to be empty
        items.sort(key=lambda sp: (score.get((sp[0], sp[1].name), 99.0),
                                   sp[0], sort_key(sp[1])))
        print(f"Ordered by {report} (most likely empty first).")
    else:
        items.sort(key=lambda sp: (sp[0], sort_key(sp[1])))
        print("Ordered by video and frame number.")
    return items


def load_progress():
    if PROGRESS.is_file():
        try:
            return json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"marks": {}, "position": 0}


def save_progress(state):
    PROGRESS.write_text(json.dumps(state, indent=1), encoding="utf-8")


def draw(img, split, name, i, total, mark, zoom, marked_count):
    h, w = img.shape[:2]
    # display-only scaling; the file on disk is never touched
    view = cv2.resize(img, (int(w * zoom), int(h * zoom)),
                      interpolation=cv2.INTER_NEAREST if zoom > 1 else cv2.INTER_AREA)
    bar = 64
    canvas = cv2.copyMakeBorder(view, bar, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    colour = {"remove": (60, 60, 235), "keep": (90, 200, 90), None: (200, 200, 200)}[mark]
    label = {"remove": "NO DRONE", "keep": "KEEP", None: "unmarked"}[mark]
    cv2.putText(canvas, f"{i+1}/{total}  [{split}]  {label}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{name[:70]}   zoom {zoom:.1f}x   marked-for-removal: {marked_count}",
                (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--order", choices=("video", "report"), default="video")
    ap.add_argument("--report", default="filter_report.csv")
    ap.add_argument("--zoom", type=float, default=1.4)
    args = ap.parse_args()

    if not IMAGES_DIR.is_dir():
        sys.exit(f"Not found: {IMAGES_DIR.resolve()} - run this from the folder "
                 f"containing dataset/")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    items = collect(splits, args.order, args.report)
    if not items:
        sys.exit("No images found.")

    state = load_progress()
    marks = state["marks"]
    i = min(state.get("position", 0), len(items) - 1)
    zoom = args.zoom

    print(f"{len(items)} frames to review. Starting at #{i+1}.")
    print("K = keep, X = no drone, J = back, Z = undo, +/- zoom, Q = save and quit\n")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while 0 <= i < len(items):
        split, path = items[i]
        key_id = f"{split}/{path.name}"
        img = cv2.imread(str(path))
        if img is None:
            i += 1
            continue
        marked = sum(1 for v in marks.values() if v == "remove")
        cv2.imshow(WIN, draw(img, split, path.name, i, len(items),
                             marks.get(key_id), zoom, marked))
        k = cv2.waitKey(0) & 0xFF

        if k in (ord('k'), ord('K'), 83, 13, 32):        # keep / right / enter / space
            marks[key_id] = "keep"; i += 1
        elif k in (ord('x'), ord('X'), 255, 8):           # no drone / del / backspace
            marks[key_id] = "remove"; i += 1
        elif k in (ord('j'), ord('J'), 81):               # back / left
            i = max(0, i - 1)
        elif k in (ord('z'), ord('Z')):
            marks.pop(key_id, None)
        elif k in (ord('+'), ord('=')):
            zoom = min(6.0, zoom + 0.2)
        elif k in (ord('-'), ord('_')):
            zoom = max(0.4, zoom - 0.2)
        elif k in (ord('f'), ord('F')):
            zoom = 1.0 if zoom != 1.0 else args.zoom
        elif k in (ord('s'), ord('S')):
            state["marks"], state["position"] = marks, i
            save_progress(state); print(f"  progress saved at #{i+1}")
        elif k in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()
    state["marks"], state["position"] = marks, min(i, len(items) - 1)
    save_progress(state)

    to_remove = [k for k, v in marks.items() if v == "remove"]
    kept = sum(1 for v in marks.values() if v == "keep")
    print(f"\nreviewed: {len(marks)} / {len(items)}")
    print(f"  keep:      {kept}")
    print(f"  no drone:  {len(to_remove)}")
    print(f"  unseen:    {len(items) - len(marks)}")

    if not to_remove:
        print("\nNothing marked for removal. Progress saved to review_progress.json.")
        return

    ans = input(f"\nMove {len(to_remove)} frame(s) to {REMOVED_DIR}/? [y/N] ").strip().lower()
    if ans != "y":
        print("Nothing moved. Your marks are saved; re-run to continue or apply later.")
        return

    moved = 0
    for key_id in to_remove:
        split, name = key_id.split("/", 1)
        src = IMAGES_DIR / split / name
        if not src.exists():
            continue
        dest = REMOVED_DIR / split
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest / name))      # file itself is unmodified
        moved += 1
    print(f"Moved {moved} frame(s). Originals are in {REMOVED_DIR}/<split>/, not deleted.")
    print("Put them back any time with:  python filter_drone_frames.py --restore")


if __name__ == "__main__":
    main()
