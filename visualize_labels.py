"""
Draw labels on a random sample of images so you can eyeball their quality.
Read-only with respect to dataset/: output goes to label_check/.

    python visualize_labels.py                 # 20 random images, all splits
    python visualize_labels.py --n 30 --split train
    python visualize_labels.py --zoom          # add a magnified inset per box
"""
import argparse, random
from pathlib import Path

import cv2

DATASET = Path("dataset")
IMAGES  = DATASET / "images"
LABELS  = DATASET / "labels"
OUT     = Path("label_check")
SPLITS  = ("train", "val", "test")
EXTS    = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--split", default=",".join(SPLITS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zoom", action="store_true",
                    help="paste a 4x magnified crop of each box in the corner")
    args = ap.parse_args()

    pool = []
    for split in [s.strip() for s in args.split.split(",") if s.strip()]:
        d = IMAGES / split
        if d.is_dir():
            pool += [(split, p) for p in d.iterdir() if p.suffix.lower() in EXTS]
    if not pool:
        raise SystemExit("no images found")

    random.Random(args.seed).shuffle(pool)
    sample = pool[:args.n]
    OUT.mkdir(exist_ok=True)
    drawn = empty = nolabel = 0

    for split, p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        lf = LABELS / split / (p.stem + ".txt")
        tag = "NO LABEL FILE"
        if lf.exists():
            lines = [l for l in lf.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                tag = "negative (empty label)"
                empty += 1
            else:
                tag = f"{len(lines)} box(es)"
                drawn += 1
                for k, line in enumerate(lines):
                    try:
                        _, cx, cy, nw, nh = (float(v) for v in line.split())
                    except ValueError:
                        continue
                    x1, y1 = int((cx - nw/2) * W), int((cy - nh/2) * H)
                    x2, y2 = int((cx + nw/2) * W), int((cy + nh/2) * H)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f"{int(nw*W)}x{int(nh*H)}px", (x1, max(12, y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    if args.zoom and k == 0:
                        pad = 40
                        cx0, cy0 = max(0, x1-pad), max(0, y1-pad)
                        cx1, cy1 = min(W, x2+pad), min(H, y2+pad)
                        crop = img[cy0:cy1, cx0:cx1]
                        if crop.size:
                            z = cv2.resize(crop, None, fx=4, fy=4,
                                           interpolation=cv2.INTER_NEAREST)
                            zh, zw = z.shape[:2]
                            zh, zw = min(zh, H//2), min(zw, W//2)
                            img[0:zh, W-zw:W] = z[:zh, :zw]
                            cv2.rectangle(img, (W-zw, 0), (W-1, zh), (0,255,255), 2)
        else:
            nolabel += 1
        cv2.putText(img, f"{split}  {tag}", (8, H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(OUT / f"{split}__{p.name}"), img)

    print(f"wrote {len(sample)} image(s) to {OUT.resolve()}")
    print(f"  with boxes: {drawn}   negatives: {empty}   no label file: {nolabel}")
    print("Open that folder and check every box sits tightly on a drone.")


if __name__ == "__main__":
    main()
