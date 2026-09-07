# Drone detection dataset — frames only, NOT yet labeled

## What this is
Frames extracted from 8 handheld videos of a drone in flight, split by video,
manually reviewed so that every frame contains a visible drone.

**There are no labels/annotations yet.** No `labels/` folder, no `data.yaml`.
The next step is drawing bounding boxes.

## Contents

```
dataset/images/train/   198 frames   6 videos
dataset/images/val/      40 frames   1 video
dataset/images/test/     44 frames   1 video
                        ---
                        282 frames
```

Filenames are `<video name>_frame<NNNNNN>.jpg`, so every frame traces back to
its source video and position. Frames are unmodified — original resolution,
no crop, no resize.

## How the splits were made
Split is by **video**, never by frame, so no video contributes frames to more
than one split. Verified: no leakage.

| split | source video |
|-------|--------------|
| train | 18.23.23, 18.27.15, 18.27.17, 18.27.17(1), 18.27.18, 18.27.19 |
| val   | 18.23.15 |
| test  | 18.27.25 |

Val and test are deliberately drawn from two different shooting sessions so
they don't measure the same thing. Train includes footage from both sessions.

## Known limitations — please read before trusting any metric
- **282 frames from 8 videos is small.** Frames within a video are highly
  correlated (sampled at 5 FPS from continuous flight), so the effective
  sample size is much closer to 8 than to 282.
- **Val and test are one video each.** Expect noisy metrics; a handful of
  frames swings a percentage point by 2-3. Treat scores as directional.
- **The drone is small** — roughly 30x15 px in an 848x480 frame in much of the
  footage, sometimes smaller. This is a small-object detection problem.
  Consider training at higher `imgsz` (1280+) and/or tiled inference.
  COCO-pretrained YOLO does not detect this drone at all (confidence 0.000
  on frames where it is plainly visible), so don't use one as a baseline.
- **Single location.** All footage is the same building/courtyard. The model
  will likely not generalize to other backgrounds without more data.

## Frames that were excluded
262 additional frames were reviewed and judged to contain no visible drone.
They are not in this zip but are kept (not deleted) on the source machine
under `dataset/removed/<split>/`, should you want them as background/negative
examples — which can be worth adding later.
