"""
Merge everything in external/ with dataset/ into dataset_combined/.

The point is more backgrounds and more distances in TRAIN, without breaking
the one thing that makes this project's numbers meaningful:

    val   = dataset/images/val    (local footage, unchanged)
    test  = dataset/images/test   (local footage, unchanged, still touched once)
    train = dataset/images/train  +  every external source

So a V3 number is directly comparable to the V2 numbers in the README. Public
data is a training aid, not a new yardstick.

Optionally a slice of the external data is held out as a SECOND val set
(`--ext-val 0.05`), written to dataset_combined/data_extval.yaml. That one
answers the question the local val cannot: does it work on backgrounds it has
never seen? Selection still happens on the local val.

    python build_combined_dataset.py                 # dry run, prints the plan
    python build_combined_dataset.py --apply
    python build_combined_dataset.py --apply --ext-val 0.05 --max-neg-ratio 0.4
    python build_combined_dataset.py --apply --copy  # if hardlinks fail

Leak safety: every external image is content-hashed against local val and test
and dropped on a match, and against the rest of external to kill duplicates.
Files are hardlinked by default (instant, no extra disk); dataset/ is never
modified.
"""
import argparse, hashlib, os, random, shutil
from pathlib import Path

LOCAL = Path("dataset")
EXT_ROOT = Path("external")
OUT = Path("dataset_combined")
SPLITS = ("train", "val", "test")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def images_in(d):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS) \
        if d.is_dir() else []


def label_for(img, images_root, labels_root):
    return labels_root / img.relative_to(images_root).with_suffix(".txt")


def digest(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def place(src_img, src_lbl, split, copy):
    """Hardlink (or copy) one image+label pair into the combined dataset."""
    di = OUT / "images" / split / src_img.name
    dl = OUT / "labels" / split / (src_img.stem + ".txt")
    if di.exists():
        di.unlink()
    if copy:
        shutil.copy2(src_img, di)
    else:
        try:
            os.link(src_img, di)          # instant, and no second copy on disk
        except OSError:                   # different volume, or a filesystem
            shutil.copy2(src_img, di)     # without hardlinks
    dl.write_text(src_lbl.read_text(encoding="utf-8") if src_lbl.exists() else "",
                  encoding="utf-8")


def count_boxes(lbl):
    if not lbl.exists():
        return 0
    return len([l for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="without this it is a dry run")
    ap.add_argument("--copy", action="store_true", help="copy instead of hardlink")
    ap.add_argument("--ext-val", type=float, default=0.0,
                    help="fraction of external images held out as a second, "
                         "unseen-background val set (e.g. 0.05)")
    ap.add_argument("--max-neg-ratio", type=float, default=0.5,
                    help="cap external negatives at this fraction of external "
                         "positives, so background images cannot drown the "
                         "drones (default 0.5)")
    ap.add_argument("--max-per-source", type=int, default=0,
                    help="cap images taken from any one source (0 = no cap); "
                         "use it so one huge dataset does not dominate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (LOCAL / "images" / "train").is_dir():
        raise SystemExit("dataset/images/train not found - run this from the "
                         "project folder.")
    rng = random.Random(args.seed)

    # ---- local: the backbone of the combined set, splits preserved exactly ----
    plan = {s: [] for s in SPLITS}
    plan["val_ext"] = []
    holdout_hashes = set()
    local_counts = {}
    for s in SPLITS:
        imgs = images_in(LOCAL / "images" / s)
        for p in imgs:
            plan[s].append((p, LOCAL / "labels" / s / (p.stem + ".txt")))
        local_counts[s] = len(imgs)
        if s in ("val", "test"):
            holdout_hashes |= {digest(p) for p in imgs}
    print(f"local:  train {local_counts['train']}  val {local_counts['val']}  "
          f"test {local_counts['test']}")
    print(f"hashed {len(holdout_hashes)} local val/test images for leak checking\n")

    # ---- external: everything goes to train, minus an optional ext-val slice ----
    sources = sorted(d for d in EXT_ROOT.iterdir()
                     if d.is_dir() and d.name != "_raw" and (d / "images").is_dir()) \
        if EXT_ROOT.is_dir() else []
    if not sources:
        print("No sources under external/. Run fetch_public_data.py --list first.")
        return

    seen = set(holdout_hashes)
    ext_stats = {}
    for src in sources:
        pos, neg, leaked, dup = [], [], 0, 0
        for p in images_in(src / "images"):
            h = digest(p)
            if h in holdout_hashes:
                leaked += 1
                continue
            if h in seen:
                dup += 1
                continue
            seen.add(h)
            lbl = src / "labels" / (p.stem + ".txt")
            (pos if count_boxes(lbl) else neg).append((p, lbl))

        rng.shuffle(pos)
        rng.shuffle(neg)
        if args.max_per_source:
            pos = pos[:args.max_per_source]
        cap = int(len(pos) * args.max_neg_ratio)
        dropped_neg = max(0, len(neg) - cap)
        neg = neg[:cap]

        items = pos + neg
        rng.shuffle(items)
        n_hold = int(len(items) * args.ext_val) if args.ext_val else 0
        plan["val_ext"] += items[:n_hold]
        plan["train"] += items[n_hold:]

        ext_stats[src.name] = dict(pos=len(pos), neg=len(neg), leaked=leaked,
                                   dup=dup, dropped_neg=dropped_neg, held=n_hold)
        print(f"{src.name}: +{len(pos)} drone, +{len(neg)} negative"
              f"   (leak {leaked}, dup {dup}, negatives over cap {dropped_neg}"
              + (f", held out {n_hold}" if n_hold else "") + ")")

    boxes = sum(count_boxes(l) for _, l in plan["train"])
    print(f"\ncombined train: {len(plan['train'])} images, {boxes} boxes "
          f"({len(plan['train'])/max(1,local_counts['train']):.1f}x the local train set)")
    print(f"combined val:   {len(plan['val'])} (local, unchanged)")
    print(f"combined test:  {len(plan['test'])} (local, unchanged)")
    if plan["val_ext"]:
        print(f"external val:   {len(plan['val_ext'])} (unseen backgrounds)")

    if not args.apply:
        print("\nDry run. Re-run with --apply to write dataset_combined/.")
        return

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in list(SPLITS) + (["val_ext"] if plan["val_ext"] else []):
        (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)
    for s, items in plan.items():
        for img, lbl in items:
            place(img, lbl, s, args.copy)

    root = OUT.resolve().as_posix()
    (OUT / "data.yaml").write_text(
        f"# Local footage + public drone data. Generated by build_combined_dataset.py\n"
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 1\nnames:\n  0: drone\n", encoding="utf-8")
    if plan["val_ext"]:
        (OUT / "data_extval.yaml").write_text(
            f"# Same data, but val = held-out PUBLIC images (unseen backgrounds).\n"
            f"# Report this alongside the local val; never select on it and never\n"
            f"# tune on test.\n"
            f"path: {root}\ntrain: images/train\nval: images/val_ext\n"
            f"test: images/test\nnc: 1\nnames:\n  0: drone\n", encoding="utf-8")

    print(f"\nWrote {OUT.resolve()}")
    print("Verify, then train:")
    print("  python validate_labels.py --dataset dataset_combined")
    print("  python train_v3.py")


if __name__ == "__main__":
    main()
