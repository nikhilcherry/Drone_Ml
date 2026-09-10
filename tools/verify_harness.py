import sys, time; sys.path.insert(0, "tools")
from pathlib import Path
from ultralytics import YOLO
from evalkit import load_gt, report
from infer import run_plain

W = "runs/detect/runs/detect/public_s/weights/best.pt"
split = sys.argv[1] if len(sys.argv) > 1 else "val"
model = YOLO(W)

print(f"--- Ultralytics val() on {split} (reference) ---", flush=True)
m = model.val(data="dataset_public/data.yaml", split=split, imgsz=640,
              device="0", plots=False, verbose=False)
print(f"  ultralytics: P {m.box.mp:.4f}  R {m.box.mr:.4f}  "
      f"mAP50 {m.box.map50:.4f}  mAP50-95 {m.box.map:.4f}", flush=True)

print(f"\n--- evalkit on {split} ---", flush=True)
t = time.time(); gt = load_gt(split); print(f"  gt loaded {len(gt)} imgs in {time.time()-t:.0f}s", flush=True)
t = time.time(); preds = run_plain(model, gt, imgsz=640, conf=0.001, iou=0.7)
secs = time.time() - t
report(f"evalkit baseline {split}", gt, preds, conf_thr=0.25, secs=secs)
