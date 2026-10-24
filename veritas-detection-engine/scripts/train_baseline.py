"""Train + evaluate the CIFAKE baseline locally.

Run from the repo root:
    python -m scripts.train_baseline --data-root data/cifake

--data-root must be the folder that directly contains train/ and test/.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from src.data.in_memory import GpuLoader, load_images
from src.data.manifest import read_manifest, split_rows
from src.eval.metrics import binary_metrics, expected_calibration_error, roc_auc
from src.training.export import export_onnx
from src.training.train import predict, train
from src.vision.cnn_2d_artifact import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--size", type=int, default=224, help="input resolution; 128 is faster")
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not found. Install the CUDA build of PyTorch (see pytorch.org).")
    device = torch.device("cuda")
    torch.manual_seed(42)
    print("GPU:", torch.cuda.get_device_name(0))

    manifest = Path("data/cifake_manifest.csv")
    if not manifest.exists():
        subprocess.run([sys.executable, "scripts/make_manifest.py", "--root", str(args.data_root),
                        "--out", str(manifest)], check=True)
    train_rows, val_rows, test_rows = split_rows(read_manifest(manifest))
    print("train/val/test:", len(train_rows), len(val_rows), len(test_rows))

    t0 = time.time()
    x_tr, y_tr = load_images(train_rows, args.data_root)
    x_va, y_va = load_images(val_rows, args.data_root)
    x_te, y_te = load_images(test_rows, args.data_root)
    print(f"images loaded in {time.time() - t0:.0f}s")

    bs, size = args.batch_size, args.size
    train_dl = GpuLoader(x_tr, y_tr, device, bs, shuffle=True, augment=True, size=size)
    val_dl = GpuLoader(x_va, y_va, device, bs, size=size)
    test_dl = GpuLoader(x_te, y_te, device, bs, size=size)

    Path("weights").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    ckpt = "weights/baseline_best.pt"
    model = build_model()
    t0 = time.time()
    train(model, train_dl, val_dl, device, epochs=args.epochs, lr=args.lr, ckpt_path=ckpt)
    print(f"training took {(time.time() - t0) / 60:.1f} min")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    probs, labels = predict(model, test_dl, device)
    result = {**binary_metrics(labels, probs), "auc": roc_auc(labels, probs),
              "ece": expected_calibration_error(labels, probs), "input_size": size}
    print(json.dumps(result, indent=2))
    Path("results/cifake_baseline.json").write_text(json.dumps(result, indent=2))
    export_onnx(model, "weights/image_artifact.onnx", size=size)
    print("saved weights/image_artifact.onnx and results/cifake_baseline.json")


if __name__ == "__main__":
    main()
