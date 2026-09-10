# Experiments log

Ground rules, inherited from the README and not negotiable:

- **Selection is on val. The test split is spent once, by the winner.** Nothing
  in this log is a test number.
- A change that alters the weights is a separate run. A change that only alters
  inference is reported as a separate row against the same weights.
- Baseline for every row: V4 `yolo11s`, 640 px, batch 12, 70 epochs
  (`runs/detect/public_s`), val mAP50 **0.9035** by Ultralytics `val()`,
  **0.9004** by `tools/evalkit.py`. Compare like with like - evalkit rows
  against the evalkit baseline.

---

## Inference-only (V4 weights, unchanged)

| # | change | val mAP50 | vs base | verdict |
|---|---|---|---|---|
| I1 | plain 640 (evalkit baseline) | 0.9004 | - | reference |
| I2 | plain 960 | 0.8909 | -0.0095 | **rejected** - small +2.1 pts, large -12.4 pts |
| I3 | plain 1280 | 0.7783 | -0.1221 | **rejected** |
| I4 | TTA (README's V4 result, test split) | 0.891 | -0.008 | **rejected**, confirms README |
| I5 | targeted despeckle, `rf_anti_drone` only | 0.6958 vs 0.6980 | -0.0022 | **inconclusive** (see note) |
| I6 | **tiled 320->640 (2x), sub-16px subset** | recall .439 -> **.528** | **+8.9 pts** | superseded by I8 |
| I7 | tiled 320->960 (3x), sub-16px subset | recall .483, FP 275 | worse than I6 | rejected |
| I8 | **tiled 2x + full frame + drop-edge** | see below | **+10.0 small / +1.7 large** | **accepted** |
| I9 | **operating point conf 0.25 -> 0.13** | R .869 -> **.902** at P>=0.80 | **+3.3 pts recall** | **accepted** |
| I10 | tracker `--det-floor` 0.03 + tuned bytetrack | - | - | **unvalidated** - no footage |

**I2/I3.** Resolution above the native 640 helps the small band and destroys the
large one, because `rf_drone_yolov7` is 96.8% large and leaves the trained scale
range. The dataset is bimodal; a single inference scale cannot serve both.

**I6/I7.** Measured on a deliberately hard subset - 180 val images whose GT
boxes are *all* <=16 px - because an easier sample saturates at 1.000 recall for
every config and measures nothing. Once `tiled_infer.py` actually magnifies
(see below), slicing does what it was always supposed to do:

| config | recall | FP |
|---|---|---|
| plain 640 | 0.439 | 69 |
| **tiled 320->640 (2x)** | **0.528** | 88 |
| tiled 320->960 (3x) | 0.483 | 275 |

2x magnification buys **+8.9 points of recall on the band that holds the entire
deficit**, for +19 false positives. 3x overshoots: pushing a 12 px drone to 36 px
takes it past the scale the model was trained on, exactly the failure seen in
I2/I3, and the false positives quadruple. There is an optimum and it is near 2x.

**I8 - the tiler that wins on both bands.** Tiling large drones was catastrophic
before this: a 380 px drone does not fit a 320 px crop, so each tile reports a box
around the fragment it can see, and none of those match the whole object.

Dropping detections that touch a tile's *interior* edge removes exactly those
fragments. The 25% overlap is what makes it safe - an object near one tile's seam
sits well inside its neighbour, so it is still seen once, whole. Edges that lie on
the image border are real edges and are not treated as cuts.

| subset | config | recall | FP |
|---|---|---|---|
| small, all boxes <=16 px (150 imgs) | plain 640 | 0.447 | 57 |
| | tiled 2x, keep edge boxes | 0.547 | 78 |
| | **tiled 2x, drop edge boxes** | **0.547** | **71** |
| large, all boxes >96 px (120 imgs) | plain 640 | 0.908 | **5** |
| | tiled 2x, keep edge boxes | 0.933 | **242** |
| | **tiled 2x, drop edge boxes** | **0.925** | **8** |

Dropping edge boxes is **free on the small band** - identical recall, 7 fewer
false positives - and it takes the large band from 242 false positives back to 8
while keeping recall above the plain baseline. The full-frame pass is what keeps
large objects at all (without it the large band falls to 0.717).

Net: **+10.0 pts recall on sub-16px drones and +1.7 pts on large ones, with no
retraining.**

Caveat, stated because it matters: these are two deliberately-chosen subsets, not
a headline mAP. They were chosen hard on purpose - an average subset saturates at
1.000 and measures nothing - and a subset recall is not comparable to the 0.9004
figure at the top of this file. This needs one full-val sweep on the GPU to become
a headline number, which is queued for when the T1 run finishes.

**I5.** Measured on the `rf_anti_drone` subset of val only (778 images), so the
mAP50 is a subset number and comparable only to the raw row beside it. The model
was *trained* on speckled images, so feeding it clean ones is a train/test
mismatch that should hurt - and it mildly does (recall 0.6615 -> 0.6448). This
does **not** settle whether training on despeckled data helps; it only says
there is no free win at inference. Deprioritised rather than disproved.

---

## Training runs

| # | change | val mAP50 | vs base | status |
|---|---|---|---|---|
| T0 | baseline `yolo11s` 640 | 0.9035 | - | done (V4) |
| T1 | `yolo11s-p2` (stride-4 head) | - | - | **running** |

**T1 rationale.** 44% of GT boxes are under 16x16 px and the median equivalent
side is 18 px. Stock YOLO11 detects at strides 8/16/32, so a 9 px drone lands on
roughly one cell of the finest feature map, and the DFL head that places the box
edges is regressing distances that are under one stride unit wide. A P2/4 head
doubles the spatial support for the band that holds the whole deficit, and -
unlike raising the inference resolution - leaves the P3/P4/P5 scales that
already work at 0.944 large-box recall untouched.

Ultralytics ships `yolov8-p2.yaml` but no YOLO11 equivalent, so
`cfg/yolo11s-p2.yaml` applies that pattern to the YOLO11 backbone. Layers 0-16
keep stock indices, so 297/593 tensors transfer from `yolo11s.pt` (the whole
backbone). 9.57 M params vs 9.46 M; 29.1 GFLOPs vs 21.7.

Tracked live against the baseline's own per-epoch curve rather than waiting for
the end - `runs/detect/public_s/results.csv` has it:

| epoch | 1 | 2 | 3 | 5 | 10 | 15 | 20 | 30 | 50 | 70 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline mAP50 | .583 | .601 | .566 | .681 | .782 | .834 | .865 | .884 | .898 | .902 |

---

**I9.** The detector has an overlapping box for **98.7%** of GT drones at conf
0.001 and only 8.9% are missing at conf 0.25 - it sees them and scores them low
(median 0.074 for the ones lost at 0.25). Priced honestly on 600 randomly
sampled val images, moving the threshold to ~0.13 buys +3.3 points of recall
while holding precision at 0.80. Below that precision falls off a cliff
(conf 0.03 -> P 0.597). mAP50 already integrates over this, so it is an
operating-point improvement, not a model improvement.

Methodological note worth keeping: the first version of this measurement sliced
`[:400]` off a **sorted** image list, which puts every `neg_rf_anti_drone_*`
first and sampled almost entirely the hardest source - it reported precision
0.712 where the representative figure is 0.902. Random-sample, always.

**I10.** Follows from I9 but is *not* measured: a tracker is the one consumer
that can use sub-threshold boxes safely, because association gates them
spatially and `new_track_thresh` still gates track creation. There is no video
in this repo, so this is reasoned, not validated, and is logged as such.

## Training runs, continued

| # | change | status |
|---|---|---|
| T2 | **fine-tune P2 on 320px crops** (`dataset_tiled/`) | **queued, dataset built** |

**T2 rationale - close the train/inference gap that I8 opened.** Magnified
tiling is worth +10 points of recall on sub-16px drones, but the detector was
trained on whole 640 px frames: at inference a 320 px crop is fed to a 640 px
network, so every object arrives 2x larger than anything training ever showed
it. Training on the same crops removes that mismatch. This is the standard SAHI
fine-tuning recipe and it targets the one band that holds the deficit.

`tools/build_tiled_dataset.py` builds it. Only **train** is tiled; val and test
stay whole, with their original labels, scored by tiled inference - which is the
deployment path anyway, and keeps the number directly comparable to "V4 weights
+ tiled inference" instead of inventing a new yardstick.

| | |
|---|---|
| positive crops | 55,032 (55,208 boxes) |
| negative crops | 18,368 |
| tiles dropped, a box only partly inside | 9,230 |
| source images skipped, smallest box > 64px | 5,571 |
| box side inside the crop | median 14.4 px |
| **as the network sees it at imgsz 640** | **median 28.8 px** (was 18.7) |

A tile holding a partly-visible drone is dropped rather than clipped: labelling
half a drone as a whole one, and leaving the visible half unlabelled, are both
lies to the detector. Overlapping tiles do re-present the same drone more than
once, which is deliberate - it oversamples exactly the objects being missed.

Labels verified on a **random** 6,000-file sample (6,018 boxes, 0 out of range)
and eyeballed as crops with boxes drawn. Random matters: a sorted sample is all
`neg_*` and silently checks nothing, which is the same trap that spoiled the
first confidence measurement.

## Queued, in priority order

1. **T2 above**, then **`multi_scale=True`.** The bimodality is the clearest structural
   problem in the data and multi-scale training is the direct answer to it: it
   would make one model tolerate both the 14 px and the 382 px source, and would
   also make the higher-resolution inference in I2 usable instead of harmful.
2. **T3 - despeckled retrain.** Only worth the GPU time if T1/T2 leave
   `rf_anti_drone` still the worst source. I5 says there is no free win, not
   that the retrain would fail.
3. **Label repair.** ~13 negatives across all splits contain a clearly airborne
   drone (about 2% of negatives). Small, real, and cheap to fix; too small to
   explain any headline number.

## Rejected, with the measurement that killed them

- 960 px / 1280 px inference (I2, I3)
- TTA (I4, and the README's own V4 result)
- Tiled inference *as written* - `--tile 640` on 640 px images is one tile
  covering the frame, and the crops were run at `imgsz=tile` so nothing was ever
  magnified. Both fixed in `tiled_infer.py`; the idea is not disproved, the
  implementation simply never did what it claimed.
- Blanket 3x3 median denoising - erases drones at <=14 px along with the noise.
