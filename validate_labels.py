"""
Validate a YOLO single-class dataset before training. Read-only: this script
never writes to dataset/, it only reports.

Checks, per split:
  - every image opens, and its dimensions
  - every image has a matching .txt label
  - no orphan label files (a .txt with no image)
  - every line is well formed: 5 fields, class id 0, floats
  - coordinates in [0,1], width/height > 0
  - boxes stay inside the image
  - no duplicate images across train/val/test (content hash)
  - flags multiple boxes on one image (one drone per frame was the rule for
    the local footage; public data legitimately has several, so pass
    --allow-multi when checking dataset_combined)

    python validate_labels.py
    python validate_labels.py --dataset dataset_combined --allow-multi
"""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def check_line(line):
    """Return (ok, message, box) for a single YOLO line."""
    parts = line.split()
    if len(parts) != 5:
        return False, f"expected 5 fields, got {len(parts)}", None
    try:
        cls = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
    except ValueError:
        return False, "non-numeric field", None
    if cls != 0:
        return False, f"class id {cls}, expected 0", None
    for name, v in (("center_x", cx), ("center_y", cy), ("width", w), ("height", h)):
        if not (0.0 <= v <= 1.0):
            return False, f"{name}={v:.4f} outside [0,1]", None
    if w <= 0 or h <= 0:
        return False, f"width/height must be > 0 (got {w:.4f}x{h:.4f})", None
    if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 \
       or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
        return False, "box extends outside the image", None
    return True, "", (cx, cy, w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset",
                    help="dataset folder to check (default: dataset)")
    ap.add_argument("--allow-multi", action="store_true",
                    help="do not warn about images with more than one box")
    args = ap.parse_args()

    DATASET = Path(args.dataset)
    IMAGES, LABELS = DATASET / "images", DATASET / "labels"
    SPLITS = tuple(s for s in ("train", "val", "val_ext", "test")
                   if (IMAGES / s).is_dir()) or ("train", "val", "test")
    print(f"checking {DATASET.resolve()}  splits: {', '.join(SPLITS)}\n")

    errors, warnings = [], []
    stats = {}
    hashes = defaultdict(list)
    dims = defaultdict(int)
    all_boxes = []

    for split in SPLITS:
        idir, ldir = IMAGES / split, LABELS / split
        if not idir.is_dir():
            errors.append(f"missing folder: {idir}")
            continue
        imgs = sorted(p for p in idir.iterdir() if p.suffix.lower() in EXTS)
        n_img = len(imgs)
        n_pos = n_neg = n_missing = n_box = 0
        boxes = []

        for p in imgs:
            im = cv2.imread(str(p))
            if im is None:
                errors.append(f"{split}/{p.name}: cannot be opened")
                continue
            h, w = im.shape[:2]
            dims[(w, h)] += 1
            hashes[hashlib.md5(p.read_bytes()).hexdigest()].append(f"{split}/{p.name}")

            lf = ldir / (p.stem + ".txt")
            if not lf.exists():
                n_missing += 1
                errors.append(f"{split}/{p.name}: no label file")
                continue
            lines = [l.strip() for l in lf.read_text(encoding="utf-8").splitlines()
                     if l.strip()]
            if not lines:
                n_neg += 1
                continue
            if len(lines) > 1 and not args.allow_multi:
                warnings.append(f"{split}/{lf.name}: {len(lines)} boxes "
                                f"(one drone per frame was the rule)")
            ok_any = False
            for ln, line in enumerate(lines, 1):
                ok, msg, box = check_line(line)
                if not ok:
                    errors.append(f"{split}/{lf.name} line {ln}: {msg}")
                else:
                    ok_any = True
                    n_box += 1
                    boxes.append((box[2] * w, box[3] * h))
            if ok_any:
                n_pos += 1

        # orphan labels
        if ldir.is_dir():
            stems = {p.stem for p in imgs}
            for lf in ldir.glob("*.txt"):
                if lf.stem not in stems:
                    errors.append(f"{split}/{lf.name}: label with no image")

        stats[split] = dict(images=n_img, positives=n_pos, negatives=n_neg,
                            missing=n_missing, boxes=n_box)
        all_boxes += boxes

    # cross-split duplicates
    for h, where in hashes.items():
        if len(where) > 1:
            splits_seen = {w.split("/")[0] for w in where}
            (errors if len(splits_seen) > 1 else warnings).append(
                ("DUPLICATE ACROSS SPLITS: " if len(splits_seen) > 1
                 else "duplicate image: ") + ", ".join(where))

    print("=" * 58)
    for split in SPLITS:
        s = stats.get(split)
        if not s:
            continue
        print(f"{split.upper()}")
        print(f"  Images:            {s['images']}")
        print(f"  Images with drone: {s['positives']}")
        print(f"  Negative images:   {s['negatives']}")
        print(f"  Missing labels:    {s['missing']}")
        print(f"  Bounding boxes:    {s['boxes']}")
        print()
    print("-" * 58)
    print("image dimensions:")
    for (w, h), c in sorted(dims.items(), key=lambda kv: -kv[1]):
        print(f"  {w}x{h}: {c} image(s)")
    if all_boxes:
        ws = [b[0] for b in all_boxes]; hs = [b[1] for b in all_boxes]
        print(f"\nbox width  px: min {min(ws):.0f}  max {max(ws):.0f}  "
              f"avg {sum(ws)/len(ws):.1f}")
        print(f"box height px: min {min(hs):.0f}  max {max(hs):.0f}  "
              f"avg {sum(hs)/len(hs):.1f}")
        tiny = sum(1 for w_, h_ in all_boxes if w_ * h_ < 32 * 32)
        print(f"boxes smaller than 32x32 px: {tiny}/{len(all_boxes)} "
              f"({tiny/len(all_boxes)*100:.0f}%)")
    print("-" * 58)

    if warnings:
        print(f"\n{len(warnings)} WARNING(S):")
        for w in warnings[:20]:
            print("  ! " + w)
        if len(warnings) > 20:
            print(f"  ... and {len(warnings)-20} more")
    if errors:
        print(f"\n{len(errors)} ERROR(S):")
        for e in errors[:30]:
            print("  X " + e)
        if len(errors) > 30:
            print(f"  ... and {len(errors)-30} more")
        print("\nNOT READY FOR TRAINING - fix the errors above.")
    else:
        print("\nAll checks passed. Dataset is ready for YOLO training.")


if __name__ == "__main__":
    main()
