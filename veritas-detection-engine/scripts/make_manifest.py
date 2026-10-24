"""Build a manifest CSV (image_path, label, source_generator, split) for CIFAKE.

Usage:
    python scripts/make_manifest.py --root /path/to/cifake --out data/cifake_manifest.csv

Expects the official layout: <root>/{train,test}/{REAL,FAKE}/*.jpg
Labels: 0 = real, 1 = AI-generated (matches src/vision/onnx_infer.py).
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image

SOURCES = {"REAL": ("cifar10", 0), "FAKE": ("stable-diffusion-v1.4", 1)}


def build(root: Path):
    rows = []
    for split in ("train", "test"):
        for cls, (source, label) in SOURCES.items():
            for p in sorted((root / split / cls).glob("*.jpg")):
                rows.append({"image_path": str(p.relative_to(root)), "label": label,
                             "source_generator": source, "split": split})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--check-sizes", type=int, default=500, help="sample N images to verify size")
    args = ap.parse_args()

    rows = build(args.root)
    if not rows:
        raise SystemExit(f"No images found under {args.root}. Check the folder layout.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label", "source_generator", "split"])
        w.writeheader()
        w.writerows(rows)

    print("counts:", dict(Counter((r["split"], r["source_generator"]) for r in rows)))
    step = max(1, len(rows) // args.check_sizes)
    sizes = Counter(Image.open(args.root / r["image_path"]).size for r in rows[::step])
    print("sampled image sizes:", dict(sizes))
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
