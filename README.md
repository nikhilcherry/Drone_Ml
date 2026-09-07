# Drone Detection — YOLO11n

Single-class (`0 = drone`) detector trained on handheld video of a quadcopter.
Covers the full pipeline: raw video → frames → manual labels → training →
evaluation → real-time tracking.

**Current best model**

```
runs\detect\runs\detect\v2_hires\weights\best.pt
```

Run it live (the `--imgsz 960` is **required** — at 640 you lose most of the accuracy):

```bat
python realtime_track.py --weights "runs\detect\runs\detect\v2_hires\weights\best.pt" --imgsz 960
```

---

## Results

Test set: 67 images, 42 ground-truth boxes, 25 background images.

| | V1 (640px) | **V2_hires (960px)** |
|---|---|---|
| Precision | 0.616 | **0.776** |
| Recall | 0.534 | **0.667** |
| mAP50 | 0.559 | **0.771** |
| mAP50-95 | 0.225 | **0.295** |
| Best epoch | 34 | 21 |
| Train time | 6 min | 18 min |

Inference ≈12 ms/frame on an RTX 4050 (~80 FPS), comfortably real-time.

**Read these numbers with care.** Val and test disagree by ~0.09 mAP50 on the
same model. With only 40–42 boxes per split, differences smaller than ~0.1 are
noise. Treat V2 as "roughly 0.7 mAP50", not 0.771.

### What worked

- **960px input** (from 640) — worth ~0.13 mAP50. The drone is only tens of
  pixels wide and frames are mixed portrait/landscape, so letterboxing to 640
  was throwing away the target.
- **Background/negative images** — precision 0.616 → 0.776. Training on
  positives only taught the model "darkest compact blob = drone", so it fired
  on eyes and other dark spots. 130 confirmed drone-free frames fixed that.

### What did not work

- **Heavier augmentation** (`scale` 0.5, `hsv_v` 0.4, `degrees` 6) — cost about
  0.20 mAP50 vs the original settings. Reverted.
- **Automatic frame filtering / box proposals** — COCO-pretrained YOLO scores
  0.000 on drones plainly visible in this footage, and motion-based proposals
  landed on people and trees instead. All labels are hand-drawn.

---

## Known limitations

1. **Fails completely on unseen backgrounds.** 0 of 198 drone training frames
   have a grass background; the model detects nothing over a green field
   (IoU 0.00, not merely degraded). It learned *this courtyard*, not *drones*.
2. **Distance is untested.** Exactly 1 of 280 boxes is under 32×32 px. The
   stated goal is detection at distance; this dataset cannot measure it.
3. **8 videos, one location.** The effective sample size is 8, not 282 frames.
4. **Single video per val/test split**, which is why the two disagree.

**The fix for all four is footage, not hyperparameters.** ~12 clips over grass,
trees, roads, at 50–100 m, in varied light. Every script here handles new video
unchanged.

---

## Dataset

```
dataset/
├── images/{train,val,test}/     294 / 51 / 67   (incl. 96/11/23 negatives)
├── labels/{train,val,test}/     YOLO txt, one box per drone, class 0
├── removed/{train,val,test}/    frames reviewed as having no drone
├── labels_backup_<timestamp>/   pre-clamp label backup
└── data.yaml
```

- Filenames keep the source video: `<video>_frame<NNNNNN>.jpg`. Splits are by
  **video**, never by frame — no video appears in two splits (verified).
- Negatives are prefixed `neg_` and have empty `.txt` files.
- Images are original resolution, never resized or cropped.

---

## Pipeline

Run from this folder. Requires `pip install ultralytics opencv-python` and a
CUDA build of torch (`train.py` refuses to run on CPU).

| Step | Command |
|---|---|
| 1. Video → frames at 5 FPS, split by video | `python extract_frames.py videos` |
| 2. Review frames, drop empty ones | `python review_frames.py` |
| 3. Draw boxes (magnifier + 4× zoom) | `python label_tool.py --split train` |
| 4. Validate labels — must pass before training | `python validate_labels.py` |
| 5. Eyeball 20 random labels | `python visualize_labels.py --n 20 --zoom` |
| 6. Add background images | `python add_negatives.py --apply` |
| 7. Train | `python train.py` |
| 8. Evaluate on test + size breakdown | `python evaluate.py` |
| 9. Live detection + ByteTrack | `python realtime_track.py --imgsz 960` |

### Other scripts

| Script | Purpose |
|---|---|
| `experiment.py` | Trains several configs, picks the winner on **val**, tests it once |
| `analyze_failures.py` | Confidence sweep on val; writes `failure_montage.jpg` of misses |
| `recall_by_background.py` | Splits recall by background type (green field vs sky/building) |
| `fix_label_bounds.py` | Clamps boxes to [0,1]; backs up all labels first |
| `reshuffle_splits.py` | Moves whole videos between splits, `images/` and `removed/` in lockstep |
| `capture_negatives.py` | Grabs webcam background frames as negatives |
| `filter_drone_frames.py` | Frame filter. **Useless with COCO weights** — only worth running with a trained drone model via `--model best.pt` |
| `propose_boxes.py` | Auto box proposals. **Do not use** — proposals landed on people and trees, not drones |

---

## Reproducing V2

```bat
python validate_labels.py
python experiment.py --only v2_hires --epochs 150 --patience 25
```

Config: `yolo11n.pt` pretrained · imgsz 960 · batch 8 · AdamW via
`optimizer="auto"` (~0.00125 lr) · patience 25 · augmentation
`fliplr 0.5, degrees 5, translate 0.1, scale 0.4, perspective 0.0005,
hsv_s 0.5, hsv_v 0.3, mosaic 1.0, close_mosaic 15` · no flipud, shear, or mixup.

**Method notes:** the test set is evaluated once, by the val-selected winner
only. Never tune against test. Ultralytics may nest run folders under its own
`runs_dir` — always take the path from `model.trainer.save_dir`, don't rebuild
it by hand.

---

## Next: V3

In priority order.

1. **Film varied backgrounds** — grass, trees, roads, overcast, dusk. Biggest
   single gap; unseen backgrounds are a total failure, not a degradation.
2. **Film at distance** — 50 m and 100 m, for the small-object case that is the
   point of the project.
3. **More videos, not more frames per video.**
4. **3–4 videos per split** so val and test stop disagreeing by 0.09.
5. **Tighter boxes** — mAP50-95 0.295 vs mAP50 0.771 says localization is loose;
   some labels carry extra sky margin.
6. **Then try 1280px / tiled inference** — only once genuinely small drones exist.
