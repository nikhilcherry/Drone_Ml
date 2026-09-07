"""
Capture background (no-drone) images from your webcam to kill false positives.

Point the camera at whatever V1 is wrongly detecting - your face, the room,
a desk, a window - and hold SPACE to grab frames. Every frame is saved with an
empty .txt, telling YOLO "there is no drone anywhere in this image".

MAKE SURE NO DRONE IS IN VIEW. These are labeled as containing nothing.

    python capture_negatives.py                     # 60 frames into train
    python capture_negatives.py --n 40 --split val
    python capture_negatives.py --source 1

    SPACE (hold)  capture frames
    Q             stop
"""
import argparse, time
from pathlib import Path
import cv2

IMAGES, LABELS = Path("dataset/images"), Path("dataset/labels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="frames to capture")
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--source", default="0")
    ap.add_argument("--every", type=float, default=0.15,
                    help="min seconds between captures, avoids near-duplicates")
    ap.add_argument("--prefix", default="webcam_neg")
    args = ap.parse_args()

    idir, ldir = IMAGES/args.split, LABELS/args.split
    idir.mkdir(parents=True, exist_ok=True); ldir.mkdir(parents=True, exist_ok=True)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.source}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    saved, last = 0, 0.0
    print(f"Hold SPACE to capture up to {args.n} frames into {args.split}. Q to stop.")
    print("Make sure NO drone is visible - these are labeled as empty.\n")
    while saved < args.n:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()
        cv2.putText(view, f"NEGATIVES {saved}/{args.n}  - hold SPACE, Q to quit",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(view, "no drone should be in frame",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("capture negatives (Q to quit)", view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q'), 27):
            break
        if k == 32 and time.time() - last >= args.every:
            name = f"{args.prefix}_{stamp}_{saved:04d}"
            cv2.imwrite(str(idir / f"{name}.jpg"), frame)   # written unmodified
            (ldir / f"{name}.txt").write_text("", encoding="utf-8")
            saved += 1
            last = time.time()

    cap.release(); cv2.destroyAllWindows()
    print(f"\nSaved {saved} negative(s) to {idir}")
    print("Vary the scene: your face, the room, a window, different lighting.")
    print("Then: python validate_labels.py  &&  python train.py")


if __name__ == "__main__":
    main()
