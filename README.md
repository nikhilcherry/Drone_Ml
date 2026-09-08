# Drone Detection — YOLO11n

Single-class (`0 = drone`) detector trained on handheld video of a quadcopter.
Covers the full pipeline: raw video → frames → manual labels → training →
evaluation → real-time tracking.

**Current best model** — V2. V3 (local footage + public drone data, see below)
lands in `runs\detect\v3_finetune\weights\best.pt`.

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
CUDA build of torch (`train.py` refuses to run on CPU). The V3 steps below add
`pip install roboflow`, and only for sources fetched through Roboflow.

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

### V3 pipeline — public data (see "V3: more data, from the internet" below)

| Step | Command |
|---|---|
| A. See what public data is available | `python fetch_public_data.py --list` |
| B. Pull and normalise a source | `python fetch_public_data.py --source <name> --rf-key KEY` |
| C. Merge with the local footage | `python build_combined_dataset.py --apply --ext-val 0.05` |
| D. Check the merge | `python validate_labels.py --dataset dataset_combined --allow-multi` |
| E. Pretrain on everything, fine-tune on local | `python train_v3.py` |
| F. Free accuracy, no retraining | `python evaluate.py --weights <best.pt> --imgsz 960 --tta --tile` |

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
| `fetch_public_data.py` | Downloads public drone datasets and converts any of them (YOLO / COCO / VOC, single- or multi-class) to this project's single-class format under `external/` |
| `build_combined_dataset.py` | Merges `external/` into `dataset/` as `dataset_combined/`. Local val/test stay untouched; every external image is hash-checked against them for leakage |
| `train_v3.py` | Two-stage training: pretrain on the combined set, fine-tune on the local footage |
| `tiled_infer.py` | Sliced inference — the small-object fix that needs no retraining. Also importable: `predict_tiled(model, frame)` |

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

## V3: more data, from the internet

The README above says it plainly: the four known limitations are all one
problem, and **the fix is footage, not hyperparameters**. Filming ~12 new clips
is still the best thing you can do. Until that happens, public drone datasets
are the same medicine from a different bottle — thousands of drones over grass,
trees, roads and open sky, at distances this dataset does not contain.

```bat
python fetch_public_data.py --list
python fetch_public_data.py --source roboflow_drone_detection --rf-key YOUR_KEY
python build_combined_dataset.py --apply --ext-val 0.05
python validate_labels.py --dataset dataset_combined --allow-multi
python train_v3.py
```

### How the merge protects the numbers

```
dataset_combined/
├── images/train     local train + every external source
├── images/val       local val,  UNCHANGED
├── images/test      local test, UNCHANGED
└── images/val_ext   a slice of public data held out (optional, --ext-val)
```

Val and test stay exactly the local footage, so a V3 score is directly
comparable to the V2 row in the table above. Public data is a training aid, not
a new yardstick. Every external image is content-hashed against local val and
test and dropped on a match, so a public dataset that happens to contain a frame
you already have cannot leak.

`val_ext` is the number that answers limitation #1. The local val cannot tell
you whether the model learned *drones* or *this courtyard*; a held-out public
split can. It is reported, never selected on.

### What the normaliser handles

Public datasets arrive in every shape. `fetch_public_data.py` takes YOLO,
COCO `instances_*.json` and Pascal VOC, in `images/train`, `train/images` or
flat layouts, and emits one format: class `0 = drone`, boxes clamped to [0,1].

Multi-class sources are mapped by class **name**, and narrowly — `bird`,
`person`, `plane` and `helicopter` are *not* drones, and letting them through
would poison a single-class detector. An image whose only boxes were dropped
becomes a **negative**, which is exactly what this project wants more of:
negatives took precision from 0.616 to 0.776. A drone-vs-bird source is
therefore worth more than its drone count suggests — the birds become hard
negatives at drone-like scale.

### Why two stages

Stage A trains on the combined set: enough varied data to justify `yolo11s` and
to teach what a drone *is*. Stage B fine-tunes that on the local footage alone
at `lr0=0.0005`, so the model specialises back onto the actual camera and scene
without forgetting stage A. Stage B keeps the V2 augmentation recipe on purpose
— heavier augmentation cost ~0.20 mAP50 on this footage, and that finding still
stands for the local data even though stage A augments harder on a much larger
set.

Selection is on the local val, the test set is still evaluated once, by the
winner only.

If stage B overfits the 294 local images (val mAP falls below stage A), try
`--freeze-b 10`, or skip it: `python train_v3.py --skip-b`.

### Accuracy without retraining

Two switches on an existing model:

```bat
python evaluate.py --weights best.pt --imgsz 960 --tta
python evaluate.py --weights best.pt --imgsz 960 --tile --tile-size 640
```

- `--tta` — flips and scales at inference, merged. A few points of recall for
  roughly 3x the time.
- `--tile` — sliced inference. The drone is ~30x15 px in an 848x480 frame, so
  the network sees it tiny no matter what `imgsz` is; a 640 px crop upscaled to
  the network input shows it several times larger. This is the switch that
  moves the `small (<32x32)` band in the size breakdown. `realtime_track.py
  --tile` does the same on recorded footage (it disables track IDs — the
  tracker needs whole frames).

Neither changes the weights, so both are honest to report as long as you say
which one produced a number.

### Expect the metrics to move, and read them carefully

More data should raise recall on unseen backgrounds a lot — from "0.00 IoU over
a green field", almost anything is an improvement. The **local** test number may
move much less, or dip: it measures one video of one courtyard, which V2 already
fit well. That is not a regression in the model, it is the local test set being
too small and too narrow to see the improvement. With ~42 test boxes,
differences under ~0.1 mAP50 are noise — the README said so for V2 and it is
just as true for V3.

Report three numbers, always together: local val, `val_ext`, and local test.

---

## V4: training on public data, on a split that has not been leaked into

The V3 section above was never able to run: `fetch_public_data.py` could not
download anything. Two separate faults, both now fixed.

1. The curated Roboflow slug 404s - that workspace has moved or gone private.
   Four working sources are registered in its place.
2. `main()` pre-created the download directory, and the Roboflow SDK treats an
   existing `location` as "already downloaded" and returns immediately without
   fetching. The symptom was the confusing one: a clean exit reporting
   `0 drone images` with no download progress bar and no error.

### The public test splits are contaminated, and it matters

Public drone datasets are video frames dealt out at random, with several
augmented copies of each frame. The same moment of the same flight therefore
lands in train *and* test. On the anti-UAV set:

| | official random split | grouped split |
|---|---|---|
| test images bit-identical to a training image | 23% | 0% |
| held-out frames within RMSE 3 of a training frame | 25% | **0.0%** |

A model that memorises training frames scores near-perfectly on the official
split without having learned anything transferable. Quoting that number would
be the exact mistake this README spends four paragraphs refusing to make about
the local footage.

`split_public.py` fixes it. It groups the augmented copies of a source frame,
then joins genuinely duplicated frames by 32x32 thumbnail RMSE, and deals
*clusters* into train/val/test - the role `reshuffle_splits.py` gives a whole
video. It then measures its own leakage and writes the result to
`split_report.json`, so the claim is auditable rather than asserted.

A note on method: an earlier version used a 256-bit average hash and reported
27% duplicates where pixel comparison finds 4.5%. Drone imagery is mostly
smooth sky, and aHash thresholds a smooth gradient into near-identical bit
patterns for unrelated images. Thumbnail RMSE does not have that failure mode.

### One source had to be thrown away

`rf_zhejiang` declares its classes as `0` and `drone`. The normaliser correctly
refused to treat a class named `0` as a drone, which dropped 12,183 boxes and
silently converted 4,996 images into negatives - images that would then have
taught the detector that drones are background. Rendering them settled it: the
dataset is mosaic-augmented crops of blank surfaces, with `drone` boxes drawn
on featureless white noise. It is excluded.

**The general lesson, which applies to any source added later:** look at the
images before training on them. `SOURCE.json` records
`dropped_nondrone_boxes` and the negative count for exactly this reason - a
large drop count on a single-class source means the class mapping is wrong or
the data is not what it claims.

### The pipeline

| Step | Command |
|---|---|
| A. List sources | `python fetch_public_data.py --list` |
| B. Fetch one | `python fetch_public_data.py --source rf_anti_uav --rf-key KEY` |
| C. **Look at what arrived** | `python visualize_labels.py --n 20 --zoom` |
| D. Build a grouped split | `python split_public.py --src external/rf_anti_uav --link` |
| E. Check it | `python validate_labels.py --dataset dataset_public --allow-multi` |
| F. Train | `python train_public.py --workers 3` |
| G. Fine-tune onto local footage | `python train_v3.py --skip-a --stage-a-weights <best.pt>` |

`train_public.py` is stage A of `train_v3.py` for the case where `dataset/`
does not exist - `train_v3.py` requires the local footage and refuses to run
without it. It selects on val, touches test once, and reports per-source and
per-size breakdowns, because a studio catalogue shot and a 20-pixel drone over
a field are not the same problem and averaging them hides that.

### Dataset

30,875 images from three verified sources, split 24,699 / 3,088 / 3,088:

| Source | Images | What it contributes |
|---|---|---|
| `rf_anti_uav` | 19,834 | drones over buildings, trees, open sky; mostly tiny |
| `rf_anti_drone` | 7,862 | **grass fields and overcast sky** - limitation #1 |
| `rf_drone_yolov7` | 3,165 | close-range studio shots; what a drone looks like |

**66% of boxes are smaller than 32x32 px.** The local dataset has exactly one
such box in 280. This set can finally measure the distance case the project
was built for, and it is a much harder problem than the courtyard footage -
scores here are not comparable to the V2 row above, and are not meant to be.


### Still the priority list

Public data narrows the gap; it does not close it. Nothing online was shot on
your camera, at your angles, in your light.

1. **Film varied backgrounds** — grass, trees, roads, overcast, dusk.
2. **Film at distance** — 50 m and 100 m.
3. **More videos, not more frames per video.**
4. **3–4 videos per split** so val and test stop disagreeing by 0.09.
5. **Tighter boxes** — mAP50-95 0.295 vs mAP50 0.771 says localization is loose.
6. **1280px / tiled inference** — `tiled_infer.py` exists now; it earns its
   keep once genuinely small drones are in the data.
