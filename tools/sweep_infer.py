"""Sweep inference-time configs on VAL. Never touches test."""
import sys, time, json, os
sys.path.insert(0, "tools")
os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO
from evalkit import load_gt, report
from infer import run_plain, run_tiled

W = os.environ.get("WEIGHTS", "runs/detect/runs/detect/public_s/weights/best.pt")
SPLIT = os.environ.get("SPLIT", "val")
model = YOLO(W)
gt = load_gt(SPLIT)
print(f"weights={W} split={SPLIT} images={len(gt)}", flush=True)

results = []
which = sys.argv[1] if len(sys.argv) > 1 else "res"

if which == "res":
    for imgsz in (640, 960, 1280):
        t = time.time()
        p = run_plain(model, gt, imgsz=imgsz, conf=0.001, iou=0.7,
                      batch=16 if imgsz > 900 else 32)
        results.append(report(f"plain imgsz={imgsz}", gt, p, secs=time.time()-t))
elif which == "tile":
    cfgs = [
        dict(tile=320, net=640,  overlap=0.25),   # 2.0x magnification
        dict(tile=320, net=960,  overlap=0.25),   # 3.0x
        dict(tile=213, net=640,  overlap=0.25),   # 3.0x, finer grid
    ]
    for c in cfgs:
        t = time.time()
        p = run_tiled(model, gt, conf=0.001, iou=0.7, full_pass=True,
                      full_imgsz=640, merge_iou=0.6, **c)
        mag = c["net"] / c["tile"]
        results.append(report(f"tiled tile={c['tile']} net={c['net']} "
                              f"({mag:.1f}x)", gt, p, secs=time.time()-t))

out = f"sweep_{which}_{SPLIT}.json"
json.dump(results, open(out, "w"), indent=1, default=float)
print(f"\nwrote {out}")
