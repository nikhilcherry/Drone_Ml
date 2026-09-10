"""
Real-time drone detection + tracking with the V1 model.

    python realtime_track.py                          # default webcam
    python realtime_track.py --source 1               # another camera
    python realtime_track.py --source rtsp://user:pass@ip:554/stream
    python realtime_track.py --source myclip.mp4 --save
    python realtime_track.py --weights v3.pt --imgsz 960 --tta
    python realtime_track.py --weights v3.pt --source clip.mp4 --tile

ByteTrack keeps a drone's ID alive across frames where detection drops out,
which matters here: V1 detects intermittently, so raw per-frame boxes flicker.

The detector is far less blind than its confident output suggests. Measured on
val: at conf 0.25 it has no box at all for 18.8% of ground-truth drones; at
conf 0.001 that falls to 1.8%. It sees ~98% of them and scores them low - the
ones it loses at 0.25 have a median confidence of 0.074. Feeding the tracker
only boxes above 0.25 therefore throws away precisely the weak detections
ByteTrack's second association stage exists to use. `--det-floor` (default 0.03)
is what reaches the tracker; `--conf` still governs what counts as a confident
detection. NOT YET VALIDATED ON VIDEO - there is no footage in this repo to
test it against. The still-image measurement behind it is in
docs/ACCURACY_INVESTIGATION.md; validating it needs a clip and a count of
how long a track survives.

Press Q to quit.

--tile runs sliced inference (tiled_infer.py): far better on distant drones,
several times slower, and it turns tracking off because it bypasses the
Ultralytics tracker. Use it on recorded footage, not on a live feed.

HONEST LIMITATION: V1 was trained on drones against sky and buildings only.
Against grass, trees, or any unseen background it will detect nothing at all -
tracking cannot recover a target the detector never finds even once. A model
trained by train_v3.py on public data is the fix for that; TTA and tiling only
sharpen what the detector can already see.
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
    ap.add_argument("--det-floor", type=float, default=0.03,
                    help="detector floor fed to the TRACKER (not the display). "
                         "ByteTrack's second association stage needs low-score "
                         "boxes to keep a track alive, and on this data the "
                         "drones that go missing at conf 0.25 have a median "
                         "confidence of 0.074 - filtering at 0.25 in the "
                         "detector leaves that stage nothing to work with. "
                         "Starting a NEW track still needs conf >= "
                         "new_track_thresh, so this cannot spawn noise tracks. "
                         "Set equal to --conf to restore the old behaviour")
    ap.add_argument("--tracker", default="cfg/bytetrack_drone.yaml",
                    help="tracker config; the shipped bytetrack.yaml has "
                         "track_low_thresh 0.1, too high for this detector")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--save", action="store_true", help="write annotated mp4")
    ap.add_argument("--tta", action="store_true",
                    help="test-time augmentation: more recall, fewer FPS")
    ap.add_argument("--tile", action="store_true",
                    help="sliced inference for distant drones; disables tracking")
    ap.add_argument("--tile-size", type=int, default=320,
                    help="crop size in source pixels; must be SMALLER than the "
                         "frame or slicing does nothing")
    ap.add_argument("--overlap", type=float, default=0.25)
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

    if args.tile:
        from tiled_infer import predict_tiled
        print("tiled inference: no track IDs (the tracker needs whole frames)")

    fps_hist = deque(maxlen=30)
    seen, hits = 0, 0
    print("Running. Q to quit.\n")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        if args.tile:
            det = predict_tiled(model, frame, tile=args.tile_size,
                                overlap=args.overlap, conf=args.conf,
                                device=args.device, full_imgsz=args.imgsz,
                                tta=args.tta)
            boxes = [(b[:4], b[4], None) for b in det]
        else:
            # persist=True keeps track IDs alive between calls
            r = model.track(frame, imgsz=args.imgsz, device=args.device,
                            conf=min(args.det_floor, args.conf), persist=True,
                            augment=args.tta, tracker=args.tracker,
                            verbose=False)[0]
            boxes = []
            if r.boxes is not None and len(r.boxes):
                ids = (r.boxes.id.int().tolist()
                       if r.boxes.id is not None else [None] * len(r.boxes))
                boxes = list(zip(r.boxes.xyxy.cpu().numpy(),
                                 r.boxes.conf.cpu().numpy(), ids))
        fps_hist.append(1.0 / max(1e-6, time.time() - t0))
        seen += 1

        n = len(boxes)
        if n:
            hits += 1
        for b, c, tid in boxes:
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
