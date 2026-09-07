"""
Drone dataset builder: video -> frames, split by VIDEO (not by frame).

- Reads every video in VIDEO_DIR
- Samples frames at ~TARGET_FPS (default 5 FPS)
- Prefers frames that likely contain a moving drone (motion + sharpness heuristic)
- Splits the VIDEOS 70/15/15 into train/val/test
- Writes frames to dataset/images/{train,val,test}
- Frames are saved exactly as decoded (no resize, no crop, no edits)

Usage:
    python extract_frames.py                # looks in ./videos
    python extract_frames.py path/to/videos # or point it anywhere
"""

import os
import sys
import random
import cv2

# ------------------- settings you may want to change -------------------
VIDEO_DIR   = sys.argv[1] if len(sys.argv) > 1 else "videos"
OUTPUT_DIR  = "dataset/images"
TARGET_FPS  = 5.0          # frames sampled per second of video
SPLITS      = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED        = 42           # change for a different (still reproducible) shuffle

# "drone likely present" heuristic
PREFER_MOTION_FRAMES = True
KEEP_FRACTION        = 0.75   # keep the best 75% of sampled frames per video
MIN_FRAMES_PER_VIDEO = 10     # never drop below this many frames for a video
BLUR_WEIGHT          = 0.30   # how much sharpness counts vs motion
# ----------------------------------------------------------------------

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv", ".webm")


def list_videos(folder):
    if not os.path.isdir(folder):
        sys.exit(f"Video folder not found: {os.path.abspath(folder)}")
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(VIDEO_EXTS)]
    if not files:
        sys.exit(f"No video files found in {os.path.abspath(folder)}")
    return files


def safe_name(filename):
    """Keep the original video name in the frame filename, minus the extension."""
    stem = os.path.splitext(filename)[0]
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)


def split_videos(videos):
    """Split the list of VIDEOS (never frames) into train/val/test."""
    shuffled = videos[:]
    random.Random(SEED).shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, round(n * SPLITS["train"]))
    n_val = max(1, round(n * SPLITS["val"])) if n >= 3 else 0
    # make sure test gets at least one video when we have enough
    if n - n_train - n_val < 1 and n >= 3:
        n_train = n - n_val - 1
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def score_frame(gray_small, prev_small):
    """Higher score = more motion and sharper. Cheap stand-in for 'drone visible'."""
    sharpness = cv2.Laplacian(gray_small, cv2.CV_64F).var()
    motion = 0.0
    if prev_small is not None:
        motion = float(cv2.absdiff(gray_small, prev_small).mean())
    return motion + BLUR_WEIGHT * (sharpness ** 0.5)


def extract(video_path, out_dir, stem):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  !! could not open {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0 or src_fps != src_fps:  # 0 / None / NaN
        src_fps = 30.0
    step = max(1, int(round(src_fps / TARGET_FPS)))

    # ---- pass 1: score the sampled frames (frames are not kept in memory) ----
    scored = []              # (score, frame_index)
    prev_small = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            scored.append((score_frame(small, prev_small), idx))
            prev_small = small
        idx += 1
    cap.release()

    if PREFER_MOTION_FRAMES and len(scored) > MIN_FRAMES_PER_VIDEO:
        keep = max(MIN_FRAMES_PER_VIDEO, int(len(scored) * KEEP_FRACTION))
        scored.sort(key=lambda c: c[0], reverse=True)
        scored = scored[:keep]

    wanted = sorted(i for _, i in scored)

    # ---- pass 2: re-read and save only the chosen frames, unmodified ----
    cap = cv2.VideoCapture(video_path)
    wanted_set = set(wanted)
    written = 0
    idx = 0
    while wanted_set:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted_set:
            wanted_set.discard(idx)
            out_path = os.path.join(out_dir, f"{stem}_frame{idx:06d}.jpg")
            # frame is written exactly as decoded: no resize, no crop, no edits
            if cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                written += 1
        idx += 1
    cap.release()
    return written


def main():
    videos = list_videos(VIDEO_DIR)
    assignment = split_videos(videos)

    for split in SPLITS:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    print(f"Found {len(videos)} videos in {os.path.abspath(VIDEO_DIR)}\n")

    totals = {}
    for split, vids in assignment.items():
        out_dir = os.path.join(OUTPUT_DIR, split)
        count = 0
        print(f"[{split}] {len(vids)} video(s)")
        for v in vids:
            n = extract(os.path.join(VIDEO_DIR, v), out_dir, safe_name(v))
            print(f"   {v}  ->  {n} frames")
            count += n
        totals[split] = (len(vids), count)
        print()

    print("=" * 46)
    print(f"{'split':<8}{'videos':>8}{'frames':>10}")
    print("-" * 46)
    for split in ("train", "val", "test"):
        nv, nf = totals.get(split, (0, 0))
        print(f"{split:<8}{nv:>8}{nf:>10}")
    print("-" * 46)
    print(f"{'total':<8}{sum(v for v, _ in totals.values()):>8}"
          f"{sum(f for _, f in totals.values()):>10}")
    print("=" * 46)
    print(f"\nFrames written under {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
