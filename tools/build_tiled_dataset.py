"""
Build a crop-based TRAINING set that matches how tiled inference sees the world.

Why: magnified tiling at inference is worth +10 points of recall on sub-16px
drones (docs/EXPERIMENTS.md I8), but the detector was trained on whole 640 px
frames. At inference a 320 px crop is fed to a 640 px network, so every object
arrives 2x larger than anything the model saw in training. Closing that gap is
the standard SAHI fine-tuning recipe: train on the same crops you will infer on.

Only the TRAIN split is tiled. val and test stay exactly as they are, as whole
images with their original labels, and are scored with tiled inference - which
is the deployment path anyway. That keeps the result directly comparable to
"V4 weights + tiled inference" rather than inventing a new yardstick.

Tiles are kept only when their labels can be made clean:
  * a tile with no box at all is a negative, and a capped number are kept
  * a box at least `--min-inside` inside the tile is clipped and kept
  * a tile holding a box that is only PARTLY inside is DROPPED entirely -
    labelling half a drone as a whole one, or leaving the visible half
    unlabelled, both teach the detector something false
"""
import argparse, random, sys
from pathlib import Path
sys.path.insert(0, "tools")
import numpy as np, cv2

SRC = Path("dataset_public")

def tile_origins(W, H, tile, overlap):
    step = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, max(1, W - tile + 1), step))
    ys = list(range(0, max(1, H - tile + 1), step))
    if xs[-1] + tile < W: xs.append(max(0, W - tile))
    if ys[-1] + tile < H: ys.append(max(0, H - tile))
    return [(x, y) for y in sorted(set(ys)) for x in sorted(set(xs))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_tiled")
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--min-inside", type=float, default=0.7,
                    help="a box with less than this fraction of its area inside "
                         "the tile makes the tile unusable")
    ap.add_argument("--neg-per-image", type=int, default=1,
                    help="empty tiles kept per source image; negatives took "
                         "precision from 0.616 to 0.776 on the local set")
    ap.add_argument("--max-side", type=float, default=64,
                    help="only tile images whose smallest box is under this. "
                         "Cropping a 380px studio drone into 320px tiles just "
                         "makes fragments")
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    imgs = sorted((SRC / "images" / "train").iterdir())
    kept = neg_kept = dropped = skipped_big = 0
    n_box = 0
    for n, ip in enumerate(imgs):
        if ip.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lf = SRC / "labels" / "train" / (ip.stem + ".txt")
        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes = []
        if lf.exists():
            for line in lf.read_text().split("\n"):
                if not line.strip(): continue
                _, cx, cy, nw, nh = (float(v) for v in line.split()[:5])
                boxes.append([(cx-nw/2)*W, (cy-nh/2)*H, (cx+nw/2)*W, (cy+nh/2)*H])
        boxes = np.array(boxes, np.float32).reshape(-1, 4)
        if len(boxes):
            sides = np.sqrt((boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1]))
            if sides.min() > a.max_side:
                skipped_big += 1
                continue
        t = min(a.tile, min(H, W))
        empties = []
        for (x, y) in tile_origins(W, H, t, a.overlap):
            x2, y2 = x + t, y + t
            keep, bad = [], False
            for b in boxes:
                ix1, iy1 = max(b[0], x), max(b[1], y)
                ix2, iy2 = min(b[2], x2), min(b[3], y2)
                iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
                inter = iw * ih
                area = (b[2]-b[0]) * (b[3]-b[1])
                if inter <= 0:
                    continue
                if area <= 0 or inter / area < a.min_inside:
                    bad = True; break            # partly-visible drone: unusable
                keep.append([ix1-x, iy1-y, ix2-x, iy2-y])
            if bad:
                dropped += 1
                continue
            if not keep:
                empties.append((x, y))
                continue
            crop = im[y:y2, x:x2]
            stem = f"{ip.stem}_t{x}_{y}"
            cv2.imwrite(str(out/"images"/"train"/f"{stem}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            lines = []
            for k in keep:
                cx = (k[0]+k[2])/2 / t; cy = (k[1]+k[3])/2 / t
                bw = (k[2]-k[0]) / t;    bh = (k[3]-k[1]) / t
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (out/"labels"/"train"/f"{stem}.txt").write_text("\n".join(lines)+"\n")
            kept += 1; n_box += len(keep)
        rng.shuffle(empties)
        for (x, y) in empties[:a.neg_per_image]:
            crop = im[y:y+t, x:x+t]
            stem = f"neg_{ip.stem}_t{x}_{y}"
            cv2.imwrite(str(out/"images"/"train"/f"{stem}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            (out/"labels"/"train"/f"{stem}.txt").write_text("")
            neg_kept += 1
        if n % 2000 == 0:
            print(f"  {n}/{len(imgs)}  crops={kept} neg={neg_kept} "
                  f"dropped={dropped}", flush=True)

    # val/test stay whole - point the yaml at the ORIGINAL images
    (out / "data.yaml").write_text(
        f"path: {Path(a.out).resolve()}\n"
        f"train: images/train\n"
        f"val: {(SRC/'images'/'val').resolve()}\n"
        f"test: {(SRC/'images'/'test').resolve()}\n\n"
        f"nc: 1\nnames: ['drone']\n")
    print(f"\ncrops kept        : {kept}  ({n_box} boxes)")
    print(f"negative crops    : {neg_kept}")
    print(f"tiles dropped     : {dropped}  (a box was only partly inside)")
    print(f"images skipped    : {skipped_big}  (smallest box > {a.max_side}px)")
    print(f"\nwrote {out}/data.yaml  (val/test point at the ORIGINAL whole images)")

if __name__ == "__main__":
    main()
