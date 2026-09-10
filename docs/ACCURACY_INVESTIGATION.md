# Where the missing 10% actually lives

V4 scores mAP50 0.899 on the leak-free public test split. The README attributes
the whole deficit to the sub-32px band and nominates two fixes: tiled inference
and 960px training. This document measures the deficit instead of assuming it,
and finds that **one of those two fixes cannot work as written, the other is
counterproductive, and the largest single confound is in the pixels rather than
in the model.**

Every number below is on **val**. The test split is not touched by any of this;
it is spent once, at the end, by the winner.

---

## Findings at a glance

| # | Finding | Effect | Status |
|---|---|---|---|
| 1 | `--tile 640` on 640 px images is one tile - slicing never happened | - | fixed |
| 2 | The tiler ran crops at `imgsz=tile`, so it never magnified anything | - | fixed |
| 3 | Magnified tiling (2x) + dropping tile-edge fragments | **+10.0 pts** recall on sub-16px, **+1.7** on large | **accepted** |
| 4 | Inference above 640 px helps small, destroys large - the data is bimodal | -0.0095 mAP50 at 960 | rejected 960/1280 and TTA |
| 5 | The detector boxes 98.7% of drones but scores them low (median 0.074) | **+3.3 pts** recall at precision 0.80 | **accepted** |
| 6 | `realtime_track.py` filtered at conf 0.25 *before* ByteTrack, starving its low-score stage | unknown | fixed, **unvalidated** |
| 7 | `rf_anti_drone` carries 12x the salt-and-pepper noise of the other sources | explains the worst source | diagnosed |
| 8 | A 3x3 median filter erases drones <=14 px | - | rejected |
| 9 | `split_public.verify()` was tautological and sampled 16% of train | reported 0.0% leakage on a split with 23.8% | fixed + tests |
| 10 | Residual train/test near-duplicates | headline inflated **~0.004-0.01 mAP50** | measured |
| 11 | Localisation is near the pixel-quantisation limit; mAP50-95 is data-capped | - | documented |
| 12 | ~2% of negatives contain a clearly airborne drone | too small to explain the FPs | logged |

Two things stated wrongly at first and corrected in place, because the
correction is the useful part: mislabelled negatives are **not** the source of
the false positives (3 of 204), and the deficit is **not** primarily
localisation (10.5 pts undetected against 4.0 pts loosely boxed).

Training experiments are logged separately in `EXPERIMENTS.md`.

---

## 0. A harness you can sweep with

`evaluate.py` runs one `model.predict()` per image plus a full annotated-save
pass. That is fine for a final report and far too slow to compare a dozen
inference configs across 3,088 images.

`tools/evalkit.py` batches inference, caches image dimensions, and computes AP
itself so that a tiled or merged prediction set can be scored with the same
metric as a plain one. A full val sweep takes **21 s**.

It is checked against Ultralytics' own `val()` before being trusted:

| | mAP50 | mAP50-95 |
|---|---|---|
| Ultralytics `val()` | 0.9035 | 0.5540 |
| `evalkit` | 0.9004 | 0.5423 |

0.003 apart on mAP50. Close enough for the **relative** comparisons this
document makes, and not quoted as an absolute anywhere.

---

## 1. The drones are smaller than the README says

| split | n | p10 side | median side | <16px | small (<32x32) |
|---|---|---|---|---|---|
| test | 3,069 | 9.0 px | **18.3 px** | 44.4% | 65.3% |
| val | 3,078 | 8.9 px | 18.7 px | 42.8% | 66.0% |

Per source, the dataset is not one problem but two:

| source | median side | small | large |
|---|---|---|---|
| `rf_anti_uav` | 14.3 px | 77.7% | 10.0% |
| `rf_anti_drone` | 24.1 px | 62.3% | 12.6% |
| `rf_drone_yolov7` | **382.4 px** | 0.9% | **96.8%** |

**The dataset is bimodal.** One source is studio close-ups at 382 px median;
another is 14 px specks. Averaging them produces a headline number that
describes neither.

---

## 2. Two of the README's "still to try" items are broken

### `--tile 640` is a no-op

The public images are natively **640x640 and 640x480**. `tiled_infer.py` does
`tile = min(tile, max(H, W))`, so `--tile 640` yields exactly one tile covering
the whole frame. Slicing never happens.

### The tiler never magnifies anything

More fundamental, and it applies at any tile size:

```python
res = model.predict(crops, imgsz=tile, ...)   # tiled_infer.py
```

The crop is run at **its own size**. The docstring's premise - "a 640 px crop
is upscaled to the network's input, so the drone arrives several times larger"
- requires the network input to be *larger* than the crop. It never is. To
magnify, the crop must be run at `net > tile`; `tools/infer.py:run_tiled` takes
both as separate arguments for this reason.

### Upscaling at inference helps small and destroys large

The README's other proposal is 960px. Inference-time resolution, on the V4
weights:

| imgsz | mAP50 | small R | medium R | large R | `rf_drone_yolov7` R |
|---|---|---|---|---|---|
| **640** | **0.9004** | 0.819 | 0.892 | **0.944** | **0.911** |
| 960 | 0.8909 | **0.840** | **0.917** | 0.820 | 0.704 |
| 1280 | 0.7783 | 0.787 | 0.872 | 0.542 | 0.317 |

Small-object recall does improve (+2.1 pts at 960). It is paid for several
times over by large-object recall collapsing (-12.4 pts), because the studio
source leaves the scale range the model was trained on.

This also explains the README's unexplained TTA result. TTA augments across
scales; on a bimodal dataset that is the same trade, and it lost.

---

## 3. Decomposing the gap: detection vs localisation

For every GT box, the best IoU against any prediction. Recall as a function of
the IoU threshold separates "never found it" from "found it, boxed it loosely".

```
    IoU     0.1     0.2     0.3     0.4     0.5     0.6     0.7
    ALL   0.894   0.890   0.884   0.870   0.854   0.814   0.710
  small   0.856   0.854   0.848   0.837   0.819   0.771   0.651
  medium  0.959   0.947   0.933   0.906   0.892   0.853   0.794
  large   0.972   0.971   0.971   0.954   0.944   0.930   0.845
```

**GT boxes with no overlapping prediction at all: 10.5%** - at conf 0.25.
That qualifier turns out to carry the whole section; see 3a.

So the 14.6-point gap to perfect recall at IoU 0.5 splits roughly:

- **10.5 pts** - no box at conf 0.25.
- **4.0 pts** - boxed, but not to IoU 0.5.

Localisation is a real secondary term, not the main story. But per source it is
distributed very unevenly:

| source | R@IoU 0.1 | R@IoU 0.5 | localisation loss |
|---|---|---|---|
| `rf_anti_uav` | 0.922 | 0.913 | **0.9 pts** |
| `rf_drone_yolov7` | 0.962 | 0.911 | 5.1 pts |
| `rf_anti_drone` | 0.780 | 0.662 | **11.8 pts** |

`rf_anti_uav` has the *smallest* drones (14 px median) and essentially no
localisation problem. `rf_anti_drone` has *larger* drones (24 px median) and
loses 11.8 points to loose boxes. **Size is not the explanation. The source is.**

### Localisation quality barely depends on size

Mean IoU of a box that was *found* (best IoU >= 0.1), by GT size:

| GT side | n | mean IoU | fraction >= 0.75 |
|---|---|---|---|
| 0-12 px | 635 | 0.734 | 53.5% |
| 12-16 px | 469 | 0.791 | 74.8% |
| 16-24 px | 436 | 0.793 | 72.5% |
| 24-32 px | 199 | 0.801 | 72.9% |
| 32-64 px | 308 | 0.804 | 78.9% |
| 64+ px | 704 | 0.808 | 77.8% |

Seven IoU points separate an 11 px drone from a 380 px one. Once the detector
finds a tiny drone it boxes it about as well as a large one, which is the
opposite of the usual intuition and worth stating: the small-object problem here
is **finding** them, not **boxing** them.

It also explains mAP50-95. A mean matched IoU of 0.785 caps an average over
IoU .50:.05:.95 at roughly 0.55 while mAP50 sits at 0.90 - and on an 11 px box,
IoU 0.734 is about 0.6 px of error per edge, which is close to the quantisation
limit of the pixels themselves. **mAP50-95 on this dataset is near a ceiling
imposed by the data, not by the training.** Quote mAP50 here, and say the box
size with it.

### The false positives are the same defect counted twice

On `rf_anti_drone` val, at conf 0.25:

```
GT 715   TP 473   recall 0.662   FP 204
  FP on images WITH gt boxes : 201
  FP on negative images      :   3
  FP overlapping a GT box but missing IoU 0.5 : 116  (57%)
  FP box side: median 18.6 px
```

57% of that source's false positives are the detector finding the drone and
drawing a box that misses IoU 0.5. Each one is charged twice - once as a false
positive, once as a missed detection on the same drone.

---

## 3a. The detector is not blind. It is unconfident.

"No overlapping prediction" was measured at conf 0.25. Re-running the identical
images at conf 0.001 separates two failures that number had merged - a detector
with no response at all, and a detector that responded and was ignored.

600 randomly sampled val images (`tools/pr_curve.py`). *Randomly* matters: the
image list is sorted, and sorted order puts every `neg_rf_anti_drone_*` first,
so a `[:N]` slice silently samples the hardest source and reports precision 0.71
where the split as a whole gives 0.90. The first version of this section did
exactly that; these are the corrected numbers.

| conf | GT boxes with no overlapping prediction |
|---|---|
| 0.25 | 54/605 = **8.9%** (full-val figure: 10.5%) |
| 0.001 | 8/605 = **1.3%** |

**The detector puts a box on 98.7% of the drones.** It is not blind to small
drones - it sees nearly all of them and scores them low.

### What that recall actually costs

"Recoverable by lowering the threshold" is true and, alone, misleading: recall
bought at conf 0.001 is worthless if precision has collapsed by then. Priced:

| conf | precision | recall | F1 |
|---|---|---|---|
| 0.50 | 0.964 | 0.793 | 0.870 |
| 0.35 | 0.937 | 0.856 | **0.895** |
| **0.25** | **0.902** | **0.869** | 0.886 |
| 0.15 | 0.824 | 0.896 | 0.858 |
| 0.10 | 0.763 | 0.906 | 0.828 |
| 0.05 | 0.667 | 0.921 | 0.774 |
| 0.001 | 0.088 | 0.964 | 0.162 |

| constraint | best recall | at conf |
|---|---|---|
| precision >= 0.90 | 0.869 | 0.248 |
| precision >= 0.85 | 0.881 | 0.186 |
| **precision >= 0.80** | **0.902** | **0.132** |
| precision >= 0.70 | 0.919 | 0.061 |

So the honest size of the prize: **+3.3 points of recall (0.869 -> 0.902) while
holding precision at 0.80**, by moving the operating point from 0.25 to ~0.13.
Real, free, and much smaller than "98.7% of drones are seen" suggests on its own.
The default 0.25 is close to the F1 optimum and is not a bad choice; it is simply
tuned for a balanced metric rather than for recall.

### Where the weak detections are genuinely worth having: the tracker

`realtime_track.py` passed `--conf 0.25` straight into `model.track()`, which
filters at the detector, before the tracker sees anything. ByteTrack's whole
design is a **second association stage that uses low-scoring boxes** to sustain
an existing track. Ultralytics' `bytetrack.yaml` sets `track_low_thresh: 0.1`,
and that band was always empty because the detector had already cut at 0.25.

A tracker is the one consumer that can afford low-confidence input, because it
gates twice more: a weak box must also fall where an existing track predicts it,
and starting a *new* track still requires a confident detection. The precision
collapse in the table above is a per-frame number; association and temporal
persistence do not pay it in full.

`cfg/bytetrack_drone.yaml` lowers `track_low_thresh` to 0.03 and doubles
`track_buffer`, leaving `new_track_thresh` at 0.25 so faint noise cannot spawn
tracks. `realtime_track.py --det-floor` (default 0.03) controls what reaches the
tracker, separately from `--conf`.

**Reasoned from the still-image measurement, NOT validated on video** - there is
no footage in this repo to test against. Validating it means measuring track
lifetime and ID switches on a clip. It should not be quoted as a win until
someone does that.

---

## 4. What is different about `rf_anti_drone`: the pixels

Rendering its images at 4x, the answer is visible before it is measurable.
The source has been exported through a salt-and-pepper noise augmentation.

Speckle rate, measured as pixels deviating >60 grey levels from their own 3x3
median:

| source | speckle pixels |
|---|---|
| `rf_anti_uav` | 0.025% |
| `rf_drone_yolov7` | 0.022% |
| **`rf_anti_drone`** | **0.294%** |

For scale: a 9x9 px drone is 0.020% of a 640x640 image. **A typical
`rf_anti_drone` frame contains roughly 15x more speckle pixels than drone
pixels, and the specks are the same spatial scale as the target.**

The worst-performing source is the noisiest one, the noise is at the size of the
object, and it produces exactly the two symptoms observed: false alarms on empty
sky, and boxes that cannot be placed tightly.

### A 3x3 median filter is not the fix

The obvious remedy erases the target along with the noise. Verified visually on
GT crops: at **<=14 px the drone disappears entirely** under a blanket median,
because a distant drone is only 1-2 px across its thin dimension - the same
thing a median filter exists to remove.

### Targeted despeckling keeps them

`tools/despeckle.py` replaces a pixel with the local median **only where it is
both a saturated value and a large deviation** from that median. Salt-and-pepper
is extreme and isolated; a drone is mid-tone and coherent. Verified on the same
crops: drones at 13 px that a blanket median destroys survive this filter while
the specks still go.

Whether it improves accuracy is a separate question from whether it preserves
the target, and is measured separately - see the experiments log.

---

## 5. Are the labels sound?

Partly. Two defects, both smaller than they first look:

**Mislabelled negatives.** All 690 negatives in the dataset come from
`rf_anti_drone`. Ranking them by detector confidence and looking at the top 24:
about 6 contain a clearly airborne drone and are simply wrong; another ~7 show
an RC aircraft held in a hand or resting on grass, which is defensible either
way. Only 42 of 704 negatives draw any detection at all, so the clear-error rate
is around **2%** - real, worth fixing, and **not** the explanation for 204 false
positives (only 3 of them land on negative images).

**Loose positive boxes.** Zooming into small GT boxes, the drone frequently
occupies well under half the box area. This puts a ceiling on achievable IoU
that no amount of training removes.

---

## 5a. The split is leakier than its own report says

Everything above rests on the public split being clean. `split_report.json`
claims `pct_within_1: 0.0, pct_within_3: 0.0`, and the README quotes that as
"with 3,069 test boxes and no leakage, 0.899 is a measurement rather than an
estimate". Recomputed independently, from scratch:

| test vs train, 32x32 thumbnail RMSE | |
|---|---|
| pixel-identical images | **0** (the hash dedup works) |
| within RMSE 1 | **11.9%** (369 images) |
| within RMSE 3 | **23.8%** (734 images) |
| median nearest | 11.7 |

Rendering the closest pairs settles what they are: the same scene, same framing,
same moment, in train and in test as different augmented copies.

### Why the built-in check could not see it

Two independent faults in `verify()`.

**It was tautological.** It was called as `verify(X, where, unit_clusters)` -
the *representative* thumbnails and the *unit* clusters. The union-find directly
above it has already guaranteed that representatives in different clusters are
further apart than `dup_rmse`. So `pct_within_3` with `--dup-rmse 3` was forced
to 0.0 by construction. It restated the threshold; it never tested anything.

**It sampled 16% of train.** `rng.choice(tr, min(4000, len(tr)))` against 24,699
training images, so even on real data it usually missed the true nearest
neighbour. Reproducing that sampling on actual images gives median 20.4 against
the full-comparison 11.7 - close to the 22.2 the report shipped.

### Why the clustering under-merged

The near-duplicate join runs on **one arbitrary augmented copy per source
frame** (`reps = [pairs[units[k][0]][0] ...]`). Roboflow copies differ by
brightness, noise and flips, so the representative of frame A may be a heavily
noised copy and the representative of a genuinely-duplicate frame B a clean one.
Their RMSE exceeds 3, the two never merge, and they are dealt into different
splits - even though other copies of A and B are near-identical.

A smaller, separate defect: `RF_COPY` matched only `_jpg.rf.<hash>`, while
Roboflow stamps the original extension, so `_png.rf.<hash>` copies were never
grouped at all. Only 9 images and 1 train/test-spanning group here, but silent.

Both are fixed in `split_public.py`: `verify()` now compares every held-out
image against the full training set, reports val as well as test, and the regex
accepts any extension.

### How much it inflates the number

This is the part that needed care. A first comparison of near-duplicate vs clean
test images showed +0.248 mAP50 - and was worthless, because the near-duplicate
half was 97% `rf_anti_uav` and the clean half 100% `rf_anti_drone`, two sources
whose recall differs by 25 points on their own. It measured the source mix.

Controlled - one source, small boxes only, matched sample sizes:

| `rf_anti_uav`, small boxes, n=350 each | mAP50 | recall | median side |
|---|---|---|---|
| near-duplicate of train (RMSE <= 1) | 0.9194 | 0.8800 | 11.5 px |
| clean (RMSE >= 10) | 0.8825 | 0.8514 | 13.1 px |
| **difference** | **+0.0368** | **+0.0286** | (near-dup is if anything *harder*) |

**Leakage is worth about +0.037 mAP50 on the images it touches.** With ~12% of
test within RMSE 1, the headline is inflated by roughly **+0.004 to +0.01
mAP50**.

So: the README's "no leakage" claim is not supported by its own evidence, and
0.899 is optimistic - but by well under a point. The split is still far better
than the official random one (4.5% within RMSE 1, 25% within 3, and 23% bit
identical). The grouping was the right idea and it mostly worked; what failed
was the check that was supposed to prove it.

---

## 6. What follows from this

Ranked by evidence, not by convenience:

1. **A P2/4 detection head.** The finest stride in stock YOLO11 is 8, so a 9 px
   drone lands on ~1 cell of the finest feature map, and the DFL regression that
   places the box edges is working in stride units where the whole box is under
   one unit. A stride-4 head doubles the spatial support for the band holding
   the entire deficit, and unlike upscaling it does not move the scales that
   already work. `cfg/yolo11s-p2.yaml` - Ultralytics ships `yolov8-p2.yaml` but
   no YOLO11 equivalent.
2. **Despeckling `rf_anti_drone`, at train and inference alike.**
3. **Not** 960px training, and **not** TTA. Both are measured above as losses on
   this data.
4. **Tiled inference, once it actually magnifies.** Fixed in `tiled_infer.py`
   (`--tile` now defaults to 320 and `--net` controls the network size; the
   no-op case warns instead of silently doing nothing). On 180 val images whose
   GT boxes are all <=16 px, 2x magnified tiling takes recall from **0.439 to
   0.528** for +19 false positives. 3x is worse than 2x on both counts. The
   README's instinct was right; only the implementation was broken.
