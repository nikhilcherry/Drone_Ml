"""
Real-time drone detection + tracking with the V1 model.

    python realtime_track.py                          # default webcam
    python realtime_track.py --source 1               # another camera
    python realtime_track.py --source rtsp://user:pass@ip:554/stream
    python realtime_track.py --source myclip.mp4 --save

ByteTrack keeps a drone's ID alive across frames where detection drops out,
which matters here: V1 detects intermittently, so raw per-frame boxes flicker.

Press Q to quit.

HONEST LIMITATION: V1 was trained on drones against sky and buildings only.
Against grass, trees, or any unseen background it will detect nothing at all -
tracking cannot recover a target the detector never finds even once.
"""
import argparse, time
from collections import deque
from pathlib import Path

import cv2


def find_weights():
    h = (sorted(Path(".").rglob("drone_v1/weights/best.pt"))
         or sorted(Path(".").rglob("best.pt")))
    if not h:
        raise SystemExit("best.pt not found - pass --weights")
    return h[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index, file, or RTSP URL")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="0.25 was the F1 optimum on val; lower = more "
                         "detections and more false alarms")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--save", action="store_true", help="write annotated mp4")
    args = ap.parse_args()

    from ultralytics import YOLO
    w = Path(args.weights) if args.weights else find_weights()
    print(f"weights: {w.resolve()}")
    model = YOLO(str(w))

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    writer = None
    if args.save:
        writer = cv2.VideoWriter("realtime_output.mp4",
                                 cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))

    fps_hist = deque(maxlen=30)
    seen, hits = 0, 0
    print("Running. Q to quit.\n")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        # persist=True keeps track IDs alive between calls
        r = model.track(frame, imgsz=args.imgsz, device=args.device,
                        conf=args.conf, persist=True, tracker="bytetrack.yaml",
                        verbose=False)[0]
        fps_hist.append(1.0 / max(1e-6, time.time() - t0))
        seen += 1

        n = 0
        if r.boxes is not None and len(r.boxes):
            n = len(r.boxes)
            hits += 1
            ids = (r.boxes.id.int().tolist()
                   if r.boxes.id is not None else [None] * n)
            for b, c, tid in zip(r.boxes.xyxy.cpu().numpy(),
                                 r.boxes.conf.cpu().numpy(), ids):
                x1, y1, x2, y2 = (int(v) for v in b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                tag = f"drone {c:.2f}" + (f"  id{tid}" if tid is not None else "")
                cv2.putText(frame, tag, (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        fps = sum(fps_hist) / len(fps_hist)
        cv2.putText(frame, f"{fps:5.1f} FPS   detections: {n}   "
                           f"hit rate: {hits/seen:.0%}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("drone v1 - live (Q to quit)", frame)
        if writer:
            writer.write(frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    if writer:
        writer.release()
        print("saved realtime_output.mp4")
    cv2.destroyAllWindows()
    print(f"\nframes: {seen}   frames with a detection: {hits} "
          f"({hits/max(1,seen):.0%})")


if __name__ == "__main__":
    main()
