"""
Pull public drone-detection datasets off the internet and normalise every one
of them to this project's format: single class `0 = drone`, YOLO txt labels.

The README's own conclusion is that the model learned *this courtyard*, not
*drones* - 0 of 198 training frames have a grass background. The fastest fix
that does not require filming anything is thousands of public drone images
shot over grass, trees, roads and open sky.

    python fetch_public_data.py --list
    python fetch_public_data.py --source roboflow_drone_detection --rf-key KEY
    python fetch_public_data.py --url https://universe.roboflow.com/ds/XXXX?key=KEY --name uni_a
    python fetch_public_data.py --archive C:\\Downloads\\det_fly.zip --name det_fly
    python fetch_public_data.py --dir C:\\Downloads\\some_unzipped_dataset --name dutav

Every source lands in `external/<name>/` as:

    external/<name>/images/*.jpg      original pixels, never resized
    external/<name>/labels/*.txt      class 0 only, clamped to [0,1]
    external/<name>/SOURCE.json       where it came from, and what was dropped

Then run `build_combined_dataset.py` to merge them with dataset/.

What normalisation handles
  - YOLO layouts: images/train, train/images, valid/ vs val/, flat folders
  - a source data.yaml with several classes -> drone-ish names map to 0,
    everything else is DROPPED (an image left with no boxes becomes a
    negative, which is exactly what this project needs more of)
  - COCO instances_*.json
  - Pascal VOC .xml
  - unreadable images, zero-area boxes, boxes outside the frame
  - duplicate images (content hash) inside and across sources
"""
import argparse, hashlib, json, re, shutil, sys, tempfile, zipfile, tarfile
from pathlib import Path

EXT_ROOT = Path("external")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# A class name counts as "drone" if it matches one of these. Deliberately
# narrow: `bird`, `person`, `plane`, `helicopter` are NOT drones, and letting
# them through would poison a single-class detector.
DRONE_WORDS = re.compile(
    r"^(drone|drones|uav|uavs|quadcopter|quadrotor|multirotor|multicopter"
    r"|dron|flying[_ -]?drone|drone[_ -]?object|small[_ -]?uav|suav)$",
    re.IGNORECASE,
)

# Curated sources. `rf` entries download through the roboflow package (needs a
# free API key: roboflow.com -> Settings -> API keys). `manual` entries have no
# stable direct link - fetch them by hand, then point --archive/--dir here.
SOURCES = {
    "roboflow_drone_detection": dict(
        kind="rf", workspace="drone-detection-6nkkw", project="drone-detection-cmnjm",
        version=1,
        note="Roboflow Universe drone detection set. If this slug has moved, "
             "search universe.roboflow.com for 'drone detection', open the "
             "version, choose Download -> YOLOv8 -> 'show download code', and "
             "pass the resulting link to --url instead.",
    ),
    "det_fly": dict(
        kind="manual", home="https://github.com/Jake-WU/Det-Fly",
        note="13k+ images of a quadcopter in flight against sky, urban, field "
             "and mountain backgrounds, shot from another UAV. Exactly the "
             "background variety this dataset is missing. Download the archive "
             "from the repo's link, then: --archive det_fly.zip --name det_fly",
    ),
    "dut_anti_uav": dict(
        kind="manual", home="https://github.com/wangdongdut/DUT-Anti-UAV",
        note="~10k annotated detection images of UAVs over varied backgrounds. "
             "Download, then: --dir DUT-Anti-UAV/detection --name dut_anti_uav",
    ),
    "anti_uav": dict(
        kind="manual", home="https://anti-uav.github.io/",
        note="Anti-UAV RGB+IR tracking benchmark. Large, and its long distances "
             "cover the <32x32 px case this project cannot currently measure. "
             "Registration required.",
    ),
    "drone_vs_bird": dict(
        kind="manual", home="https://wosdetc2023.wordpress.com/",
        note="Drone-vs-Bird challenge. The single best source of hard negatives: "
             "birds at drone-like scale. Requires signing the challenge form.",
    ),
}


# ----------------------------- acquisition -----------------------------

def fetch_roboflow(spec, dest, api_key):
    if not api_key:
        raise SystemExit("This source needs --rf-key (free at roboflow.com).")
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("pip install roboflow")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(spec["workspace"]).project(spec["project"])
    ver = proj.version(spec["version"])
    last = None
    for fmt in ("yolov11", "yolov8"):      # identical txt format; name varies by SDK age
        try:
            ver.download(fmt, location=str(dest))
            return dest
        except Exception as e:
            last = e
    raise SystemExit(f"roboflow download failed: {type(last).__name__}: {last}")


def fetch_url(url, dest):
    import urllib.request
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "_download.bin"
    print(f"  downloading {url.split('?')[0]} ...")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"  {tmp.stat().st_size/1e6:.1f} MB")
    unpack(tmp, dest)
    tmp.unlink()
    return dest


def unpack(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(dest)
    else:
        raise SystemExit(f"Not a zip or tar archive: {archive}")
    return dest


# ----------------------------- conversion ------------------------------

def load_class_names(root):
    """Class id -> name, from a source data.yaml / classes.txt if present."""
    for y in list(root.rglob("data.yaml")) + list(root.rglob("data.yml")):
        try:
            import yaml
            d = yaml.safe_load(y.read_text(encoding="utf-8"))
        except Exception:
            continue
        names = d.get("names") if isinstance(d, dict) else None
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}
    for c in list(root.rglob("classes.txt")) + list(root.rglob("obj.names")):
        lines = [l.strip() for l in c.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            return {i: n for i, n in enumerate(lines)}
    return {}


def drone_ids(names):
    """Which source class ids are drones. No names at all -> assume single class."""
    if not names:
        return None                       # None means "keep every class"
    keep = {i for i, n in names.items() if DRONE_WORDS.match(n.strip())}
    return keep


def find_label_for(img, root):
    """The .txt that belongs to an image, across the usual YOLO layouts."""
    cand = [img.with_suffix(".txt")]
    parts = list(img.parts)
    for i, p in enumerate(parts):
        if p == "images":
            alt = list(parts); alt[i] = "labels"
            cand.append(Path(*alt).with_suffix(".txt"))
    cand.append(img.parent.parent / "labels" / (img.stem + ".txt"))
    for c in cand:
        if c.exists():
            return c
    return None


def yolo_lines(lf, keep):
    """Source YOLO txt -> our single-class lines. Returns (lines, dropped)."""
    out, dropped = [], 0
    for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if len(parts) > 5:                 # segmentation polygon, not a box
            xs, ys = [float(v) for v in parts[1::2]], [float(v) for v in parts[2::2]]
            cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
            w, h = max(xs)-min(xs), max(ys)-min(ys)
        if keep is not None and cid not in keep:
            dropped += 1
            continue
        box = clamp(cx, cy, w, h)
        if box:
            out.append("0 %.6f %.6f %.6f %.6f" % box)
    return out, dropped


def clamp(cx, cy, w, h):
    """Clamp to [0,1] and reject degenerate boxes. Mirrors fix_label_bounds."""
    x1, y1 = max(0.0, cx - w/2), max(0.0, cy - h/2)
    x2, y2 = min(1.0, cx + w/2), min(1.0, cy + h/2)
    if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
        return None
    return ((x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1)


def coco_index(root):
    """image filename -> list of (cid, cx, cy, w, h) from any COCO json found."""
    idx = {}
    for j in root.rglob("*.json"):
        if "annotation" not in j.name.lower() and "instances" not in j.name.lower():
            continue
        try:
            d = json.loads(j.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if not isinstance(d, dict) or "images" not in d or "annotations" not in d:
            continue
        cats = {c["id"]: c.get("name", "") for c in d.get("categories", [])}
        keep = drone_ids(cats) if cats else None
        imgs = {im["id"]: im for im in d["images"]}
        for a in d["annotations"]:
            im = imgs.get(a["image_id"])
            if not im or "bbox" not in a:
                continue
            if keep is not None and a.get("category_id") not in keep:
                continue
            W, H = im.get("width"), im.get("height")
            if not W or not H:
                continue
            x, y, w, h = a["bbox"]
            b = clamp((x+w/2)/W, (y+h/2)/H, w/W, h/H)
            if b:
                idx.setdefault(Path(im["file_name"]).name, []).append(b)
        # every image in the json is known, even those with no drone box
        for im in d["images"]:
            idx.setdefault(Path(im["file_name"]).name, [])
    return idx


def voc_boxes(xml_path):
    import xml.etree.ElementTree as ET
    try:
        r = ET.parse(xml_path).getroot()
    except Exception:
        return None
    size = r.find("size")
    if size is None:
        return None
    W, H = float(size.findtext("width", 0)), float(size.findtext("height", 0))
    if not W or not H:
        return None
    out = []
    for obj in r.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name and not DRONE_WORDS.match(name):
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1, y1 = float(bb.findtext("xmin", 0)), float(bb.findtext("ymin", 0))
        x2, y2 = float(bb.findtext("xmax", 0)), float(bb.findtext("ymax", 0))
        b = clamp((x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H)
        if b:
            out.append(b)
    return out


def normalise(raw, out, name, max_images, keep_negatives):
    """raw (any layout) -> out/images + out/labels, single class, deduped."""
    import cv2
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    names = load_class_names(raw)
    keep = drone_ids(names)
    coco = coco_index(raw)
    print(f"  source classes: {names or '(none declared - keeping all)'}")
    if keep is not None:
        kept_names = [names[i] for i in sorted(keep)] if names else []
        print(f"  treated as drone: {kept_names or '(none matched!)'}")
        if names and not keep:
            print("  ! no class name matched a drone. Check --list notes; this "
                  "source may need DRONE_WORDS extended before it is useful.")

    imgs = sorted(p for p in raw.rglob("*") if p.suffix.lower() in IMG_EXTS)
    seen, n_pos, n_neg, n_box, n_drop, n_bad, n_dup = set(), 0, 0, 0, 0, 0, 0

    for p in imgs:
        if max_images and (n_pos + n_neg) >= max_images:
            break
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        if digest in seen:
            n_dup += 1
            continue
        if cv2.imread(str(p)) is None:
            n_bad += 1
            continue

        lines = None
        lf = find_label_for(p, raw)
        if lf is not None:
            lines, dropped = yolo_lines(lf, keep)
            n_drop += dropped
        elif p.name in coco:
            lines = ["0 %.6f %.6f %.6f %.6f" % b for b in coco[p.name]]
        else:
            xml = p.with_suffix(".xml")
            if not xml.exists():
                xml = p.parent.parent / "annotations" / (p.stem + ".xml")
            if xml.exists():
                b = voc_boxes(xml)
                if b is not None:
                    lines = ["0 %.6f %.6f %.6f %.6f" % x for x in b]
        if lines is None:                  # no annotation of any kind: not usable
            n_bad += 1
            continue
        if not lines and not keep_negatives:
            continue

        seen.add(digest)
        stem = f"{name}_{p.stem}" if lines else f"neg_{name}_{p.stem}"
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)[:120]
        shutil.copy2(p, out / "images" / (stem + p.suffix.lower()))
        (out / "labels" / (stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        if lines:
            n_pos += 1
            n_box += len(lines)
        else:
            n_neg += 1

    (out / "SOURCE.json").write_text(json.dumps(dict(
        name=name, raw_images=len(imgs), positives=n_pos, negatives=n_neg,
        boxes=n_box, dropped_nondrone_boxes=n_drop, unreadable_or_unlabelled=n_bad,
        duplicates=n_dup, source_classes=names,
    ), indent=2), encoding="utf-8")

    print(f"  -> {n_pos} drone images ({n_box} boxes), {n_neg} negatives")
    print(f"     dropped {n_drop} non-drone boxes, skipped {n_bad} unusable, "
          f"{n_dup} duplicates")
    return n_pos, n_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="a name from --list")
    ap.add_argument("--url", help="direct zip/tar URL (e.g. a Roboflow download link)")
    ap.add_argument("--archive", help="a zip/tar you already downloaded")
    ap.add_argument("--dir", help="a folder you already unzipped")
    ap.add_argument("--name", help="folder name under external/ (required with "
                                   "--url/--archive/--dir)")
    ap.add_argument("--rf-key", default=None, help="Roboflow API key")
    ap.add_argument("--max-images", type=int, default=0, help="0 = no cap")
    ap.add_argument("--no-negatives", action="store_true",
                    help="skip images that end up with no drone box "
                         "(by default they are kept - they are useful)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep the unconverted download under external/_raw/")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not (args.source or args.url or args.archive or args.dir):
        print("Curated sources:\n")
        for k, v in SOURCES.items():
            tag = "roboflow" if v["kind"] == "rf" else "MANUAL DOWNLOAD"
            print(f"  {k}  [{tag}]")
            if v.get("home"):
                print(f"      {v['home']}")
            print(f"      {v['note']}\n")
        print("Manual sources: download by hand, then re-run with\n"
              "  --archive <file.zip> --name <name>   or   --dir <folder> --name <name>")
        return

    if args.source:
        spec = SOURCES.get(args.source)
        if not spec:
            raise SystemExit(f"unknown source '{args.source}' - see --list")
        if spec["kind"] == "manual":
            print(f"{args.source} has no direct download link.\n  {spec['home']}\n"
                  f"  {spec['note']}")
            return
        name = args.source
    else:
        name = args.name
        if not name:
            raise SystemExit("--name is required with --url/--archive/--dir")

    out = EXT_ROOT / name
    if out.exists() and any(out.glob("images/*")):
        raise SystemExit(f"{out} already has images - delete it to re-fetch.")

    raw_keep = EXT_ROOT / "_raw" / name
    tmp = None
    if args.dir:
        raw = Path(args.dir)
        if not raw.is_dir():
            raise SystemExit(f"not a folder: {raw}")
    else:
        raw = raw_keep if args.keep_raw else Path(
            tmp := tempfile.mkdtemp(prefix=f"drone_{name}_"))
        raw.mkdir(parents=True, exist_ok=True)
        if args.source:
            fetch_roboflow(SOURCES[args.source], raw, args.rf_key)
        elif args.url:
            fetch_url(args.url, raw)
        else:
            unpack(Path(args.archive), raw)

    print(f"\n{name}: normalising {raw} -> {out}")
    try:
        n_pos, n_neg = normalise(raw, out, name, args.max_images,
                                 keep_negatives=not args.no_negatives)
    finally:
        if tmp and not args.keep_raw:
            shutil.rmtree(tmp, ignore_errors=True)

    if n_pos == 0:
        print("\nNothing usable came out of this source. Open external/"
              f"{name}/SOURCE.json - most likely no class name matched a drone.")
        return
    print(f"\nDone. Next:\n  python build_combined_dataset.py\n"
          f"  python train_v3.py")


if __name__ == "__main__":
    main()
