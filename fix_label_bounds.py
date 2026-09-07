"""
Clamp YOLO boxes that extend past the image edge, back to [0,1].

Originals are copied to dataset/labels_backup_<timestamp>/ before anything is
written. Only files that actually need clamping are touched.

    python fix_label_bounds.py            # dry run, shows what would change
    python fix_label_bounds.py --apply
"""
import shutil, sys, time
from pathlib import Path
import cv2

DATASET = Path("dataset")
IMAGES, LABELS = DATASET / "images", DATASET / "labels"
SPLITS = ("train", "val", "test")


def clamp(cx, cy, w, h):
    x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    return (x1 + x2)/2, (y1 + y2)/2, x2 - x1, y2 - y1


def main():
    apply = "--apply" in sys.argv
    changes = []
    for split in SPLITS:
        for lf in sorted((LABELS / split).glob("*.txt")):
            txt = lf.read_text(encoding="utf-8")
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if not lines:
                continue
            out, touched = [], False
            for line in lines:
                p = line.split()
                if len(p) != 5:
                    out.append(line); continue
                c = int(p[0]); cx, cy, w, h = (float(v) for v in p[1:])
                ncx, ncy, nw, nh = clamp(cx, cy, w, h)
                if abs(ncx-cx) > 1e-9 or abs(ncy-cy) > 1e-9 \
                   or abs(nw-w) > 1e-9 or abs(nh-h) > 1e-9:
                    touched = True
                    out.append(f"{c} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
                else:
                    out.append(line)
            if touched:
                changes.append((lf, "\n".join(out) + "\n"))

    if not changes:
        print("No out-of-bounds boxes found. Nothing to do.")
        return
    print(f"{len(changes)} label file(s) need clamping:")
    for lf, _ in changes:
        print(f"  {lf.parent.name}/{lf.name}")
    if not apply:
        print("\nDRY RUN - nothing written. Re-run with --apply.")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = DATASET / f"labels_backup_{stamp}"
    for split in SPLITS:
        (backup / split).mkdir(parents=True, exist_ok=True)
        for lf in (LABELS / split).glob("*.txt"):
            shutil.copy2(lf, backup / split / lf.name)
    print(f"\nFull backup of all labels: {backup}")

    for lf, content in changes:
        lf.write_text(content, encoding="utf-8")
    print(f"Clamped {len(changes)} file(s).")


if __name__ == "__main__":
    main()
