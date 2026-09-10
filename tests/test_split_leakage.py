"""
Regression tests for the split-integrity check in split_public.py.

Why these exist: the shipped `verify()` reported `pct_within_3: 0.0` on a split
that actually had 23.8% of test images within RMSE 3 of a training image, and
the README quoted that zero as proof the benchmark was clean. It was not a
close call - the function compared cluster representatives, which the
de-duplication had already forced apart by construction, so it could not have
returned anything else. A check that cannot fail is not a check.

test_detects_planted_leak is the one that matters: it puts the same images in
train and test and asserts the function says so. The old implementation would
have passed the "clean" test and failed this one.

    python -m pytest tests/ -q          (or just: python tests/test_split_leakage.py)
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset_public" / "images"

spec = importlib.util.spec_from_file_location("split_public", ROOT / "split_public.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)

pytestmark = pytest.mark.skipif(
    not DATA.exists(), reason="dataset_public/ not built; run split_public.py first")


def _pairs(split, n):
    out = []
    for p in sorted((DATA / split).iterdir())[:n]:
        lf = ROOT / "dataset_public" / "labels" / split / (p.stem + ".txt")
        if lf.exists():
            out.append((p, lf))
    return out


def _singleton_clusters(pairs, boundaries):
    """boundaries: {split: (start, end)} over the pairs list."""
    clusters = [[i] for i in range(len(pairs))]
    where = {}
    for split, (a, b) in boundaries.items():
        for i in range(a, b):
            where[i] = split
    return clusters, where


def test_clean_split_reports_no_near_duplicates():
    tr, te = _pairs("train", 250), _pairs("test", 120)
    pairs = tr + te
    clusters, where = _singleton_clusters(
        pairs, {"train": (0, len(tr)), "test": (len(tr), len(pairs))})
    out = sp.verify(pairs, where, clusters, sample=120)
    r = out["test_vs_train_similarity"]
    assert r["compared"] == len(te)
    assert r["against_train"] == len(tr)
    assert r["pct_within_1"] == 0.0, "distinct images should not look identical"


def test_detects_planted_leak():
    """The regression that motivated this file."""
    tr = _pairs("train", 250)
    leaked = tr[:100]                      # the SAME images, dealt into test
    pairs = tr + leaked
    clusters, where = _singleton_clusters(
        pairs, {"train": (0, len(tr)), "test": (len(tr), len(pairs))})
    out = sp.verify(pairs, where, clusters, sample=100)
    r = out["test_vs_train_similarity"]
    assert r["pct_within_1"] > 99.0, (
        "verify() failed to notice that every test image is also a training "
        f"image (got {r['pct_within_1']}% within RMSE 1)")
    assert r["median_nearest_rmse"] == 0.0


def test_reports_val_as_well_as_test():
    """The original only ever looked at test, so val leakage went unmeasured."""
    tr, te, va = _pairs("train", 200), _pairs("test", 60), _pairs("val", 60)
    pairs = tr + te + va
    clusters, where = _singleton_clusters(
        pairs, {"train": (0, len(tr)), "test": (len(tr), len(tr) + len(te)),
                "val": (len(tr) + len(te), len(pairs))})
    out = sp.verify(pairs, where, clusters, sample=60)
    assert "val_vs_train_similarity" in out
    assert "test_vs_train_similarity" in out


@pytest.mark.parametrize("stem,expected", [
    ("frame_jpg.rf.abc123", "frame"),
    ("frame_png.rf.deadbeef", "frame"),      # the extension that was missed
    ("frame_JPEG.rf.0f0f", "frame"),
    ("frame_webp.rf.99", "frame"),
    ("frame_no_suffix", "frame_no_suffix"),
])
def test_copy_suffix_regex_covers_every_extension(stem, expected):
    """Roboflow stamps the ORIGINAL extension, not always _jpg."""
    assert sp.RF_COPY.sub("", stem) == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
