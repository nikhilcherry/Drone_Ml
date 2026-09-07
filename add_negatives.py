"""
Add background (negative) images to the dataset so the model learns what is
NOT a drone. Fixes false positives on dark compact blobs (eyes, holes, etc).

Source: dataset/removed/<split>/ - frames you reviewed and confirmed have no
drone. They are COPIED, not moved; dataset/removed stays intact. Each copy
gets an empty .txt, which is how YOLO represents a background image.

Split integrity is preserved: a negative from removed/train goes to
images/train, so no video ever crosses splits.

    python add_negatives.py                    # dry run
    python add_negatives.py --apply
    python add_negatives.py --apply --fraction 1.0    # use all of them
    python add_negatives.py --undo             # list what would be removed
"""
import argparse, random, shutil
from pathlib import Path

DATASET = Path("dataset")
IMAGES, LABELS, REMOVED = DATASET/"images", DATASET/"labels", DATASET/"removed"
SPLITS = ("train", "val", "test")
MARKER = DATASET / "negatives_added.txt"     # record of what we copied in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--fraction", type=float, default=0.5,
                    help="fraction of available negatives to use (default 0.5; "
                         "Ultralytics suggests roughly 10-50%% background images)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--undo", action="store_true",
                    help="list the negatives previously added")
    args = ap.parse_args()

    if args.undo:
        if not MARKER.exists():
            print("No record of added negatives.")
            return
        rows = [l for l in MARKER.read_text().splitlines() if l.strip()]
        print(f"{len(rows)} negative(s) were added:")
        for r in rows[:10]:
            print("  " + r)
        if len(rows) > 10:
            print(f"  ... and {len(rows)-10} more")
        print("\nDelete those image+label pairs to revert. "
              "dataset/removed/ still holds the originals.")
        return

    rng = random.Random(args.seed)
    plan = []
    for split in SPLITS:
        src = REMOVED / split
        if not src.is_dir():
            continue
        pool = sorted(p for p in src.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(pool)
        take = pool[:max(0, int(len(pool) * args.fraction))]
        for p in take:
            plan.append((split, p))

    by = {s: sum(1 for x, _ in plan if x == s) for s in SPLITS}
    print("negatives to add:")
    for s in SPLITS:
        pos = len(list((IMAGES/s).glob("*.jpg")))
        n = by.get(s, 0)
        share = n/(pos+n) if pos+n else 0
        print(f"  {s:<6} +{n:<4} -> {pos+n} total ({share:.0%} background)")

    if not args.apply:
        print("\nDRY RUN - nothing copied. Re-run with --apply.")
        return

    added = []
    for split, p in plan:
        name = "neg_" + p.name          # prefix so negatives are obvious
        dst_img = IMAGES / split / name
        dst_lbl = LABELS / split / (Path(name).stem + ".txt")
        if dst_img.exists():
            continue
        shutil.copy2(p, dst_img)
        dst_lbl.write_text("", encoding="utf-8")   # empty = background image
        added.append(f"{split}/{name}")
    MARKER.write_text("\n".join(added) + "\n", encoding="utf-8")
    print(f"\nAdded {len(added)} negative image(s). Record: {MARKER}")
    print("dataset/removed/ is untouched. Now run: python validate_labels.py")


if __name__ == "__main__":
    main()
