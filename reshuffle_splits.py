"""
Move whole VIDEOS between train/val/test after filtering.

Frames are matched by the video name embedded in the filename, and both
dataset/images/<split>/ and dataset/removed/<split>/ are moved together so the
two trees stay parallel (filter_drone_frames.py --restore keeps working).

Edit MOVES below, then:
    python reshuffle_splits.py            # dry run
    python reshuffle_splits.py --apply
"""
import re, shutil, sys
from pathlib import Path
from collections import defaultdict

DATASET = Path("dataset")
TREES   = ("images", "removed")
SPLITS  = ("train", "val", "test")
FRAME_RE = re.compile(r"^(?P<video>.+)_frame(?P<idx>\d+)$")

# video name (without _frameNNNNNN) -> destination split
MOVES = {
    "WhatsApp_Video_2026-09-07_at_18.23.23": "train",
    "WhatsApp_Video_2026-09-07_at_18.27.25": "test",
}

def video_of(path):
    m = FRAME_RE.match(path.stem)
    return m.group("video") if m else None

def main():
    apply = "--apply" in sys.argv
    planned = []
    for tree in TREES:
        for split in SPLITS:
            d = DATASET / tree / split
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                dest_split = MOVES.get(video_of(p))
                if dest_split and dest_split != split:
                    planned.append((p, DATASET / tree / dest_split / p.name))

    per = defaultdict(int)
    for src, dst in planned:
        per[f"{src.parent.parent.name}: {src.parent.name} -> {dst.parent.name}"] += 1
    for k, v in sorted(per.items()):
        print(f"  {k}: {v} frame(s)")
    print(f"total: {len(planned)} frame(s)")

    if not apply:
        print("\nDRY RUN - nothing moved. Re-run with --apply.")
        return
    for src, dst in planned:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))       # files themselves are unmodified
    print("\nMoved. New counts:")
    total = sum(len(list((DATASET/'images'/s).glob('*.jpg'))) for s in SPLITS)
    for s in SPLITS:
        n = len(list((DATASET / "images" / s).glob("*.jpg")))
        r = len(list((DATASET / "removed" / s).glob("*.jpg")))
        print(f"  {s:<6} kept {n:>4}  ({n/total*100:4.1f}%)   removed {r:>4}")

if __name__ == "__main__":
    main()
