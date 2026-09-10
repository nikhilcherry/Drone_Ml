"""Batched inference runners for evalkit. Plain and tiled.

The tiler here differs from tiled_infer.py in one decisive way: the crop is
run at a NETWORK size larger than the crop (--net > --tile), so a 320 px crop
fed to a 640 px network arrives 2x magnified. tiled_infer.py passes
imgsz=tile, i.e. no magnification at all, which on a 640x640 dataset makes
slicing a no-op twice over (one tile, and no upscale).
"""
import numpy as np
import cv2
from evalkit import iou_matrix


def nms(b, thr=0.6):
    if len(b) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    order = b[:, 4].argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ov = iou_matrix(b[i:i+1, :4], b[rest, :4])[0]
        order = rest[ov < thr]
    return b[keep]


def run_plain(model, gt, imgsz=640, conf=0.001, iou=0.7, batch=32, device="0",
              augment=False, max_det=300):
    stems = list(gt.keys())
    out = {}
    for i in range(0, len(stems), batch):
        chunk = stems[i:i+batch]
        paths = [str(gt[s]["path"]) for s in chunk]
        res = model.predict(paths, imgsz=imgsz, conf=conf, iou=iou,
                            device=device, verbose=False, augment=augment,
                            max_det=max_det)
        for s, r in zip(chunk, res):
            if r.boxes is None or not len(r.boxes):
                out[s] = np.zeros((0, 5), dtype=np.float32)
            else:
                out[s] = np.column_stack([r.boxes.xyxy.cpu().numpy(),
                                          r.boxes.conf.cpu().numpy()]
                                         ).astype(np.float32)
    return out


def tile_origins(W, H, tile, overlap):
    step = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, max(1, W - tile + 1), step))
    ys = list(range(0, max(1, H - tile + 1), step))
    if xs[-1] + tile < W: xs.append(max(0, W - tile))
    if ys[-1] + tile < H: ys.append(max(0, H - tile))
    return [(x, y) for y in sorted(set(ys)) for x in sorted(set(xs))]


def run_tiled(model, gt, tile=320, net=640, overlap=0.25, conf=0.001, iou=0.7,
              device="0", full_pass=True, full_imgsz=640, merge_iou=0.6,
              batch=32, max_det=300):
    """Crop at `tile`, run the network at `net` (net>tile => magnification)."""
    out = {}
    stems = list(gt.keys())
    for s in stems:
        g = gt[s]
        im = cv2.imread(str(g["path"]))
        if im is None:
            out[s] = np.zeros((0, 5), dtype=np.float32); continue
        H, W = im.shape[:2]
        t = min(tile, min(H, W))
        crops, offs = [], []
        for (x, y) in tile_origins(W, H, t, overlap):
            crops.append(im[y:y+t, x:x+t]); offs.append((x, y))
        acc = []
        if crops:
            res = model.predict(crops, imgsz=net, conf=conf, iou=iou,
                                device=device, verbose=False, max_det=max_det)
            for r, (ox, oy) in zip(res, offs):
                if r.boxes is None or not len(r.boxes):
                    continue
                xy = r.boxes.xyxy.cpu().numpy().copy()
                cf = r.boxes.conf.cpu().numpy()
                xy[:, [0, 2]] += ox; xy[:, [1, 3]] += oy
                acc.append(np.column_stack([xy, cf]))
        if full_pass:
            r = model.predict(im, imgsz=full_imgsz, conf=conf, iou=iou,
                              device=device, verbose=False, max_det=max_det)[0]
            if r.boxes is not None and len(r.boxes):
                acc.append(np.column_stack([r.boxes.xyxy.cpu().numpy(),
                                            r.boxes.conf.cpu().numpy()]))
        if not acc:
            out[s] = np.zeros((0, 5), dtype=np.float32); continue
        m = nms(np.vstack(acc), merge_iou).astype(np.float32)
        m[:, [0, 2]] = m[:, [0, 2]].clip(0, W)
        m[:, [1, 3]] = m[:, [1, 3]].clip(0, H)
        out[s] = m
    return out
