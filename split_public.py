"""
Re-split public data so its test score means something.

Public drone datasets are exported by Roboflow-style tooling that (a) writes
several augmented copies of every source frame and (b) splits at random, so the
*same frame* lands in train and test. On the anti-UAV set that is not a rounding
error: 23% of official test images are bit-identical to a training image by
perceptual hash. A model that memorises training frames then scores near-
perfectly on that test split without having learned anything transferable -
exactly the mistake this project's README spends four paragraphs refusing to
make about its own footage.

Two things leak, and both are handled here:

  1. Augmented copies. `x_00579_jpg.rf.<hash>.jpg` - 2 to 5 copies per source
     frame, identical content. Grouped by source stem.
  2. Genuine duplicate frames under different indices. Caught by comparing
     32x32 greyscale thumbnails and joining anything within `--dup-rmse`
     (default 3 on a 0-255 scale, chosen from the measured distribution).

Clusters, not images, are dealt into train/val/test - the same role
`reshuffle_splits.py` gives a whole video. The script then *verifies* its own
claim by reporting how close the nearest training image is to each test image,
so the split is auditable rather than merely asserted.

    python split_public.py --src external/rf_anti_uav --out dataset_public
    python split_public.py --src external/a --src external/b --out dataset_public

Output is the layout the rest of this repo expects:

    dataset_public/images/{train,val,test}/
    dataset_public/labels/{train,val,test}/
    dataset_public/data.yaml
    dataset_public/split_report.json

Negatives (`neg_` prefix, empty label file) are carried through, because
precision on this project lives or dies on background images.

Note on perceptual hashing: an earlier version of this script used a 256-bit
average hash and reported 27% duplicates where pixel comparison finds 4.5%.
Drone images are mostly smooth sky, and aHash thresholds a smooth gradient into
near-identical bit patterns for unrelated images. Thumbnail RMSE does not have
that failure mode, which is why it is what runs here.
"""
import argparse, json, re, shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
THUMB = 32                                    # 32x32 grey thumbnail per image
RF_COPY = re.compile(r"_jpg\.rf\.[0-9a-f]+$")    # Roboflow augmented-copy suffix


def source_stem(path):
    """The original frame identity, with any Roboflow copy suffix removed."""
    return RF_COPY.sub("", path.stem)


def thumb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((THUMB, THUMB)), dtype=np.float32).ravel()


def collect(srcs):
    """(image path, label path) for every annotated image under the sources."""
    pairs = []
    for src in srcs:
        src = Path(src)
        for img in sorted(src.rglob("*")):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lf = img.with_suffix(".txt")
            if not lf.exists():
                parts = list(img.parts)
                for i, part in enumerate(parts):
                    if part == "images":
                        alt = list(parts); alt[i] = "labels"
                        lf = Path(*alt).with_suffix(".txt")
                        break
            if lf.exists():
                pairs.append((img, lf))
    return pairs


def duplicate_clusters(reps, rmse, chunk=2048):
    """Union-find over pairs closer than `rmse`. Chunked so memory stays flat."""
    X = np.stack([thumb(p) for p in reps])
    sq = (X * X).sum(1)
    parent = list(range(len(reps)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    thresh = rmse ** 2 * X.shape[1]
    for s in range(0, len(reps), chunk):
        blk = X[s:s + chunk]
        d2 = sq[s:s + chunk, None] + sq[None, :] - 2 * blk @ X.T
        i_idx, j_idx = np.where(d2 <= thresh)
        for i, j in zip(i_idx, j_idx):
            gi = s + int(i)
            if gi < j:
                ra, rb = find(gi), find(int(j))
                if ra != rb:
                    parent[rb] = ra
    groups = defaultdict(list)
    for i in range(len(reps)):
        groups[find(i)].append(i)
    return list(groups.values()), X


def assign(sizes, frac_val, frac_test):
    """Largest cluster first into whichever split has the biggest relative deficit."""
    total = sum(sizes)
    want = {"val": total * frac_val, "test": total * frac_test,
            "train": total * (1 - frac_val - frac_test)}
    have = dict.fromkeys(want, 0)
    where = {}
    for c in sorted(range(len(sizes)), key=lambda c: -sizes[c]):
        split = max(want, key=lambda k: (want[k] - have[k]) / max(want[k], 1))
        where[c] = split
        have[split] += sizes[c]
    return where


def verify(X, cluster_split, clusters, sample=400, seed=0):
    """How close is the nearest training image to each test image? Lower = leakier."""
    rng = np.random.default_rng(seed)
    tr = [i for c, members in enumerate(clusters) if cluster_split[c] == "train" for i in members]
    te = [i for c, members in enumerate(clusters) if cluster_split[c] == "test" for i in members]
    if not tr or not te:
        return {}
    te = rng.choice(te, min(sample, len(te)), replace=False)
    tr = rng.choice(tr, min(4000, len(tr)), replace=False)
    A, B = X[te], X[tr]
    d = np.sqrt(np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :]
                           - 2 * A @ B.T, 0) / X.shape[1])
    mn = d.min(1)
    return {"median_nearest_rmse": round(float(np.median(mn)), 1),
            "pct_within_1": round(float((mn <= 1).mean() * 100), 1),
            "pct_within_3": round(float((mn <= 3).mean() * 100), 1),
            "pct_within_6": round(float((mn <= 6).mean() * 100), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True,
                    help="a folder from fetch_public_data.py (repeatable)")
    ap.add_argument("--out", default="dataset_public")
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--dup-rmse", type=float, default=3.0,
                    help="thumbnails closer than this are the same moment")
    ap.add_argument("--link", action="store_true",
                    help="hardlink instead of copy (same filesystem only)")
    args = ap.parse_args()

    pairs = collect(args.src)
    if not pairs:
        raise SystemExit("no annotated images found - run fetch_public_data.py first")

    # 1. augmented copies of one source frame are one unit
    units = defaultdict(list)
    for i, (img, _) in enumerate(pairs):
        units[(img.parent, source_stem(img))].append(i)
    unit_keys = sorted(units, key=lambda k: (str(k[0]), k[1]))
    reps = [pairs[units[k][0]][0] for k in unit_keys]
    copies = Counter(len(units[k]) for k in unit_keys)
    print(f"{len(pairs)} annotated images from {len(args.src)} source(s)")
    print(f"  {len(unit_keys)} distinct source frames "
          f"(copies per frame: {dict(sorted(copies.items()))})")

    # 2. duplicate source frames are one cluster
    print(f"  thumbnailing and de-duplicating at RMSE <= {args.dup_rmse} ...")
    unit_clusters, X = duplicate_clusters(reps, args.dup_rmse)
    clusters = [[i for u in c for i in units[unit_keys[u]]] for c in unit_clusters]
    sizes = [len(c) for c in clusters]
    multi = sum(1 for c in unit_clusters if len(c) > 1)
    print(f"  {len(clusters)} clusters  (largest {sorted(sizes, reverse=True)[:4]}, "
          f"{multi} hold duplicate source frames)")

    where = assign(sizes, args.val, args.test)

    out = Path(args.out)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {s: [0, 0, 0] for s in ("train", "val", "test")}   # images, boxes, negatives
    for ci, members in enumerate(clusters):
        split = where[ci]
        for i in members:
            img, lf = pairs[i]
            dst = out / "images" / split / img.name
            if args.link:
                if not dst.exists():
                    dst.hardlink_to(img.resolve())
            else:
                shutil.copy2(img, dst)
            text = lf.read_text(encoding="utf-8", errors="ignore")
            (out / "labels" / split / (img.stem + ".txt")).write_text(text, encoding="utf-8")
            n = len([l for l in text.splitlines() if l.strip()])
            counts[split][0] += 1
            counts[split][1] += n
            counts[split][2] += (n == 0)

    leak = verify(X, where, unit_clusters)

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\n\nnc: 1\nnames: [\'drone\']\n", encoding="utf-8")
    (out / "split_report.json").write_text(json.dumps({
        "sources": args.src, "images": len(pairs), "source_frames": len(unit_keys),
        "clusters": len(clusters), "dup_rmse": args.dup_rmse,
        "counts": {s: dict(zip(("images", "boxes", "negatives"), v))
                   for s, v in counts.items()},
        "test_vs_train_similarity": leak,
    }, indent=1), encoding="utf-8")

    print("\nsplit          images   boxes   negatives")
    for s in ("train", "val", "test"):
        i, b, n = counts[s]
        print(f"  {s:<10} {i:>7} {b:>7} {n:>11}")
    if leak:
        print(f"\nleak check - nearest training image to each test image "
              f"(thumbnail RMSE, 0-255):")
        print(f"  median {leak['median_nearest_rmse']}   "
              f"within 1: {leak['pct_within_1']}%   "
              f"within 3: {leak['pct_within_3']}%   "
              f"within 6: {leak['pct_within_6']}%")
        print("  (the official random split scores ~4.5% within 1 and 25% within 3)")
    print(f"\nwrote {out}/data.yaml and {out}/split_report.json")


if __name__ == "__main__":
    main()
