"""
Manual YOLO box labeling for a single class (0 = drone).

Two clicks per image, designed for a drone only ~30 px wide:
  1. LOCATE  - full frame on screen, a magnifier follows your cursor.
               Click roughly on the drone.
  2. BOX     - the tool zooms 4x into that spot. Drag a tight box.
               Enter accepts and moves on.

Writes YOLO format to dataset/labels/<split>/<same name>.txt:
    0 center_x center_y width height     (all normalised 0-1)

Images are opened read-only and are never modified.

    KEYS
      click        locate the drone, then drag to box it
      Enter/Space  accept the box, next image
      R            redraw this box
      B            back to locate mode (mis-clicked)
      N            no drone here -> writes an EMPTY .txt (negative image)
      P            previous image
      S            skip, leave unlabeled
      + / -        magnifier strength (locate) / zoom (box mode)
      Q / Esc      save and quit

Usage:
    python label_tool.py --split train
    python label_tool.py --split train --relabel      # revisit done ones
"""
import argparse, sys
from pathlib import Path

import cv2
import numpy as np

DATASET = Path("dataset")
IMAGES  = DATASET / "images"
LABELS  = DATASET / "labels"
EXTS    = (".jpg", ".jpeg", ".png", ".bmp")
WIN     = "label  |  click drone -> drag box -> Enter   (N=no drone, S=skip, Q=quit)"

FIT_W, FIT_H = 1280, 720
BANNER = 30          # px of header drawn above the image; mouse Y includes it
LOUPE = 190          # px of the source image shown in the magnifier


class State:
    def __init__(self):
        self.mode = "locate"     # locate | box
        self.mouse = (0, 0)
        self.centre = None       # click point in full-image coords
        self.drag = None         # (x0,y0,x1,y1) in display coords
        self.dragging = False
        self.zoom = 4.0
        self.loupe_zoom = 4.0


def fit_scale(w, h):
    return min(FIT_W / w, FIT_H / h, 1.0)


def draw_locate(img, st):
    h, w = img.shape[:2]
    s = fit_scale(w, h)
    view = cv2.resize(img, (int(w * s), int(h * s)))
    mx, my = st.mouse
    fx, fy = int(mx / s), int(my / s)          # cursor in full-image coords
    r = int(LOUPE / 2)
    x0, y0 = max(0, fx - r), max(0, fy - r)
    x1, y1 = min(w, fx + r), min(h, fy + r)
    if x1 > x0 and y1 > y0:
        patch = img[y0:y1, x0:x1]
        side = int(LOUPE * st.loupe_zoom / 2)
        patch = cv2.resize(patch, (side, side), interpolation=cv2.INTER_NEAREST)
        cv2.line(patch, (side // 2 - 9, side // 2), (side // 2 + 9, side // 2),
                 (0, 255, 255), 1)
        cv2.line(patch, (side // 2, side // 2 - 9), (side // 2, side // 2 + 9),
                 (0, 255, 255), 1)
        cv2.rectangle(patch, (0, 0), (side - 1, side - 1), (0, 255, 255), 1)
        ph, pw = patch.shape[:2]
        px = view.shape[1] - pw - 8
        py = 8 if my > view.shape[0] // 2 else view.shape[0] - ph - 8
        if px > 0 and py > 0:
            view[py:py + ph, px:px + pw] = patch
    return view


def draw_box_mode(img, st):
    h, w = img.shape[:2]
    cx, cy = st.centre
    half = int(180 / 2)
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, cx + half), min(h, cy + half)
    crop = img[y0:y1, x0:x1]
    view = cv2.resize(crop, (int(crop.shape[1] * st.zoom),
                             int(crop.shape[0] * st.zoom)),
                      interpolation=cv2.INTER_NEAREST)
    if st.drag:
        a, b, c, d = st.drag
        cv2.rectangle(view, (a, b - BANNER), (c, d - BANNER), (0, 0, 255), 1)
    return view, (x0, y0)


def to_yolo(box_full, W, H):
    x0, y0, x1, y1 = box_full
    x0, x1 = sorted((max(0, x0), min(W, x1)))
    y0, y1 = sorted((max(0, y0), min(H, y1)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return (0, ((x0 + x1) / 2) / W, ((y0 + y1) / 2) / H,
            (x1 - x0) / W, (y1 - y0) / H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--relabel", action="store_true",
                    help="also revisit images that already have a label")
    args = ap.parse_args()

    src = IMAGES / args.split
    dst = LABELS / args.split
    if not src.is_dir():
        sys.exit(f"Not found: {src.resolve()}")
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    if not args.relabel:
        todo = [p for p in files if not (dst / (p.stem + ".txt")).exists()]
    else:
        todo = files
    if not todo:
        print(f"All {len(files)} image(s) in {args.split} already have labels. "
              f"Use --relabel to revisit.")
        return

    print(f"{args.split}: {len(todo)} image(s) to label "
          f"({len(files) - len(todo)} already done)")
    print("Click the drone, drag a tight box, press Enter. N = no drone.\n")

    st = State()

    def on_mouse(event, x, y, flags, _):
        st.mouse = (x, y)
        if st.mode == "locate":
            if event == cv2.EVENT_LBUTTONDOWN:
                st.pending_click = (x, y)
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                st.dragging = True
                st.drag = (x, y, x, y)
            elif event == cv2.EVENT_MOUSEMOVE and st.dragging:
                a, b, _, _ = st.drag
                st.drag = (a, b, x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                st.dragging = False
                a, b, _, _ = st.drag
                st.drag = (a, b, x, y)

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)

    i = 0
    saved = neg = 0
    while 0 <= i < len(todo):
        path = todo[i]
        img = cv2.imread(str(path))
        if img is None:
            print(f"  cannot open {path.name}, skipping")
            i += 1
            continue
        H, W = img.shape[:2]
        st.mode, st.centre, st.drag = "locate", None, None
        st.pending_click = None
        origin = (0, 0)

        while True:
            if st.mode == "locate":
                view = draw_locate(img, st)
                hint = f"[{i+1}/{len(todo)}] {path.name}  - click the drone"
            else:
                view, origin = draw_box_mode(img, st)
                hint = f"[{i+1}/{len(todo)}] drag a tight box, Enter=accept, B=back"
            canvas = cv2.copyMakeBorder(view, 30, 0, 0, 0, cv2.BORDER_CONSTANT,
                                        value=(25, 25, 25))
            cv2.putText(canvas, hint, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (230, 230, 230), 1, cv2.LINE_AA)
            cv2.imshow(WIN, canvas)
            k = cv2.waitKey(20) & 0xFF

            if st.mode == "locate" and st.pending_click:
                mx, my = st.pending_click
                st.pending_click = None
                s = fit_scale(W, H)
                st.centre = (int(mx / s), int((my - BANNER) / s))
                st.mode, st.drag = "box", None
                continue

            if k in (ord('q'), ord('Q'), 27):
                cv2.destroyAllWindows()
                print(f"\nsaved {saved} box label(s), {neg} negative(s). "
                      f"Re-run to continue where you left off.")
                return
            if k in (ord('n'), ord('N')):
                (dst / (path.stem + ".txt")).write_text("", encoding="utf-8")
                neg += 1
                i += 1
                break
            if k in (ord('s'), ord('S')):
                i += 1
                break
            if k in (ord('p'), ord('P')):
                i = max(0, i - 1)
                break
            if k in (ord('b'), ord('B')):
                st.mode, st.drag = "locate", None
                continue
            if k in (ord('r'), ord('R')):
                st.drag = None
                continue
            if k in (ord('+'), ord('=')):
                if st.mode == "box":
                    st.zoom = min(12.0, st.zoom + 1)
                else:
                    st.loupe_zoom = min(10.0, st.loupe_zoom + 1)
                continue
            if k in (ord('-'), ord('_')):
                if st.mode == "box":
                    st.zoom = max(2.0, st.zoom - 1)
                else:
                    st.loupe_zoom = max(2.0, st.loupe_zoom - 1)
                continue
            if k in (13, 32) and st.mode == "box" and st.drag:
                a, b, c, d = st.drag
                b, d = b - BANNER, d - BANNER      # mouse Y includes the header
                ox, oy = origin
                full = (ox + a / st.zoom, oy + b / st.zoom,
                        ox + c / st.zoom, oy + d / st.zoom)
                y = to_yolo(full, W, H)
                if y is None:
                    print("  box too small, redraw")
                    st.drag = None
                    continue
                cls, cx, cy, nw, nh = y
                (dst / (path.stem + ".txt")).write_text(
                    f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n",
                    encoding="utf-8")
                saved += 1
                i += 1
                break

    cv2.destroyAllWindows()
    print(f"\nDone with {args.split}: {saved} box label(s), {neg} negative(s).")
    print("Next: python validate_labels.py")


if __name__ == "__main__":
    main()
