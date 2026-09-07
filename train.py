"""
Drone detection - Experiment V1 (reproducible baseline).

    python train.py

Runs GPU verification, training, validation on the held-out test set, and
inference on test images. Does not touch dataset/images or dataset/labels.
"""
from pathlib import Path

# ---------------------------- V1 configuration ----------------------------
DATA      = "dataset/data.yaml"
MODEL     = "yolo11n.pt"     # pretrained COCO nano - transfer learning, not scratch
EPOCHS    = 100
IMGSZ     = 640
BATCH     = 16               # fixed, not auto: reproducibility (see notes below)
PATIENCE  = 20               # stop if val mAP50-95 stalls for 20 epochs
WORKERS   = 4                # Windows-safe; needs the __main__ guard below
DEVICE    = 0                # RTX 4050
SEED      = 0
PROJECT   = "runs/detect"
NAME      = "drone_v1"

# Moderate augmentation for a ~300 image dataset. Nothing that would make a
# drone unrealistic: no vertical flip, no shear, no mixup, no copy-paste.
AUG = dict(
    hsv_h=0.015,      # tiny hue shift
    hsv_s=0.5,        # saturation - handles sun vs shade
    hsv_v=0.3,        # brightness - the real variation in outdoor footage
    degrees=5.0,      # small rotation; a drone is roughly level in flight
    translate=0.1,    # position variation
    scale=0.4,        # THE important one: simulates distance the data lacks
    shear=0.0,
    perspective=0.0005,
    flipud=0.0,       # drones are not upside down
    fliplr=0.5,       # horizontal flip is free and realistic
    mosaic=1.0,       # 4-image mosaic: big win on small datasets
    close_mosaic=15,  # disable mosaic for the final 15 epochs to settle
    mixup=0.0,
    copy_paste=0.0,
)
# --------------------------------------------------------------------------


def check_gpu():
    import torch
    print("torch:       ", torch.__version__)
    print("torch.cuda:  ", torch.version.cuda)
    print("cuda avail:  ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:      ", torch.cuda.get_device_name(0))
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM:         {vram:.1f} GB")
        return True
    print("\nCUDA IS NOT AVAILABLE - you have the CPU-only torch build.")
    print("Fix before training:")
    print("  pip uninstall -y torch torchvision")
    print("  pip install torch torchvision --index-url "
          "https://download.pytorch.org/whl/cu126")
    return False


def main():
    if not check_gpu():
        raise SystemExit("Refusing to train on CPU. See instructions above.")

    from ultralytics import YOLO
    model = YOLO(MODEL)

    # optimizer='auto' lets Ultralytics pick AdamW and a conservative lr for a
    # dataset this small, rather than us guessing. See notes in the chat.
    results = model.train(
        data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
        patience=PATIENCE, workers=WORKERS, device=DEVICE, seed=SEED,
        pretrained=True, optimizer="auto", project=PROJECT, name=NAME,
        exist_ok=False, plots=True, val=True, deterministic=True, **AUG,
    )

    # Ask the trainer where it actually saved. Reconstructing this path by
    # hand is wrong: Ultralytics may resolve `project` against its own
    # runs_dir setting and nest the folder.
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\nRun directory:   {save_dir.resolve()}")
    print(f"Best checkpoint: {best.resolve()}")
    if not best.exists():
        raise SystemExit(f"best.pt not found at {best}")

    # ---- held-out test set: run ONCE, never tune against it ----
    print("\n=== TEST SET (held out) ===")
    test_model = YOLO(str(best))
    m = test_model.val(data=DATA, split="test", imgsz=IMGSZ, device=DEVICE,
                       project=PROJECT, name=f"{NAME}_test", plots=True)
    print(f"Precision:  {m.box.mp:.4f}")
    print(f"Recall:     {m.box.mr:.4f}")
    print(f"mAP50:      {m.box.map50:.4f}")
    print(f"mAP50-95:   {m.box.map:.4f}")

    # ---- annotated predictions on the test images ----
    test_model.predict(
        source="dataset/images/test", imgsz=IMGSZ, device=DEVICE,
        conf=0.25, save=True, project=PROJECT,
        name=f"{NAME}_test_predictions", exist_ok=True,
    )
    print(f"\nAnnotated predictions: "
          f"{(Path(PROJECT) / (NAME + '_test_predictions')).resolve()}")


if __name__ == "__main__":   # required on Windows when workers > 0
    main()
