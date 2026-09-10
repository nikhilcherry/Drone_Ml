"""Is a missed drone invisible to the model, or merely under threshold?

'10.5% of GT boxes have no overlapping prediction' was measured at conf 0.25.
That number conflates two very different failures. Re-running the same images
at conf 0.001 separates them:
  * still no overlapping box   -> the detector has no response at all
  * a box appears              -> it responded, just weakly. That is a
                                  confidence/calibration problem, and it is
                                  addressable by a threshold rather than by a
                                  bigger model.
"""
import sys, os
sys.path.insert(0,"tools"); sys.path.insert(0,"."); os.environ["YOLO_VERBOSE"]="False"
import numpy as np, cv2
from ultralytics import YOLO
from evalkit import load_gt, iou_matrix, band_of

m = YOLO("runs/detect/runs/detect/public_s/weights/best.pt")
gt = load_gt("val")
import random; random.seed(0)
stems = random.sample([s for s, g in gt.items() if len(g["boxes"])], 400)  # RANDOM: sorted order puts neg_rf_anti_drone_* first and biases the sample to the hardest source
ngt = sum(len(gt[s]["boxes"]) for s in stems)
print(f"{len(stems)} images, {ngt} GT boxes", flush=True)

res = {}
for c in (0.25, 0.001):
    miss = 0; weak_conf = []; band_miss = {"small":0,"medium":0,"large":0}
    band_tot = {"small":0,"medium":0,"large":0}
    for s in stems:
        g = gt[s]["boxes"]
        r = m.predict(str(gt[s]["path"]), imgsz=640, conf=c, device="cpu",
                      verbose=False, max_det=300)[0]
        p = (r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes)
             else np.zeros((0,4), np.float32))
        cf = (r.boxes.conf.cpu().numpy() if r.boxes is not None and len(r.boxes)
              else np.zeros((0,), np.float32))
        M = iou_matrix(p, g)
        for gi in range(len(g)):
            b = g[gi]; bd = band_of((b[2]-b[0])*(b[3]-b[1]))
            band_tot[bd] += 1
            if M.shape[0] == 0 or M[:, gi].max() < 0.01:
                miss += 1; band_miss[bd] += 1
            elif c < 0.01:
                weak_conf.append(float(cf[M[:, gi].argmax()]))
    res[c] = miss
    print(f"  conf {c}: no overlapping prediction for {miss}/{ngt} = {miss/ngt:.1%}"
          + (f"   [small {band_miss['small']}/{band_tot['small']}, "
             f"med {band_miss['medium']}/{band_tot['medium']}, "
             f"large {band_miss['large']}/{band_tot['large']}]"), flush=True)
    if weak_conf:
        w = np.array(weak_conf)
        print(f"    of the boxes found only below 0.25: median conf {np.median(w[w<0.25]):.3f}, "
              f"count {(w<0.25).sum()}")
print(f"\n  recoverable by lowering the threshold alone: "
      f"{(res[0.25]-res[0.001])/max(res[0.25],1):.1%} of the conf-0.25 misses")
