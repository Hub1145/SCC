"""
Prepare training samples for the DirectL2VAE from reconstructed L2 files.

Handles both lowercase (real/) and capitalised (Real/) Drive folder names.

Usage:
    python scripts2/prepare_l2_training_data.py
    python scripts2/prepare_l2_training_data.py --root /workspace/Synthetic-Data
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

FEATURES  = ['bid_price', 'ask_price', 'bid_size', 'ask_size']
TIMESTEPS = 96


def prepare_data(root: str):
    # Accept both capitalised and lowercase folder names from Drive
    candidates = [
        os.path.join(root, "reconstructed"),
        os.path.join(root, "Reconstructed"),
        os.path.join(root, "real"),
        os.path.join(root, "Real"),
    ]
    bases = [b for b in candidates if os.path.isdir(b)]
    if not bases:
        print(f"[ERROR] No reconstructed/ or real/ directory found under {root}")
        sys.exit(1)

    all_files = []
    for b in bases:
        all_files.extend(glob.glob(os.path.join(b, "**", "*_L2.csv"), recursive=True))
    all_files = [f for f in all_files if "_market_ctx" not in f]

    print(f"Scanning {len(bases)} base(s) → {len(all_files)} L2 files found.")

    all_samples = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            missing = [c for c in FEATURES if c not in df.columns]
            if missing or len(df) < TIMESTEPS:
                continue
            data = df[FEATURES].values[:TIMESTEPS].astype(np.float32)
            mid0 = (data[0, 0] + data[0, 1]) / 2
            if mid0 <= 0:
                continue
            data[:, 0:2] /= mid0
            all_samples.append(data)
        except Exception as e:
            print(f"  [WARN] {f}: {e}")

    if not all_samples:
        print("[ERROR] No valid samples found.")
        sys.exit(1)

    samples = np.array(all_samples)  # [N, 96, 4]
    print(f"Created {len(samples)} training samples. Shape: {samples.shape}")

    B, T, F = samples.shape
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(samples.reshape(-1, F)).reshape(B, T, F)

    data_dir  = os.path.join(root, "data")
    model_dir = os.path.join(root, "models")
    os.makedirs(data_dir,  exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    data_path   = os.path.join(data_dir,  "l2_training_samples.npy")
    scaler_path = os.path.join(model_dir, "l2_scaler.pkl")

    np.save(data_path, scaled)
    joblib.dump(scaler, scaler_path)
    print(f"Saved training data → {data_path}")
    print(f"Saved scaler        → {scaler_path}")


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=_PROJECT_ROOT, help="Project root directory")
    args = parser.parse_args()
    prepare_data(os.path.abspath(args.root))


if __name__ == "__main__":
    main()
