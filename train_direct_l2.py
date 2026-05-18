"""
Train the DirectL2VAE on prepared L2 training samples.

Usage:
    python scripts2/train_direct_l2.py
    python scripts2/train_direct_l2.py --root /workspace/Synthetic-Data
    python scripts2/train_direct_l2.py --epochs 50 --batch-size 64
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Resolve models/ package from project root
SCRIPTS2_DIR = os.path.dirname(os.path.abspath(__file__))


def _setup_path(root: str):
    if root not in sys.path:
        sys.path.insert(0, root)


def train(root: str, epochs: int, batch_size: int, latent_dim: int):
    _setup_path(root)
    from models.direct_l2_vae import DirectL2VAE, vae_loss_function

    data_path  = os.path.join(root, "data",   "l2_training_samples.npy")
    model_path = os.path.join(root, "models", "direct_l2_vae_v1.pth")

    if not os.path.exists(data_path):
        print(f"[ERROR] Training data not found: {data_path}")
        print("        Run prepare_l2_training_data.py first.")
        sys.exit(1)

    data   = np.load(data_path)
    tensor = torch.FloatTensor(data)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training DirectL2VAE on {device}  ({len(data)} samples, {epochs} epochs)")

    model     = DirectL2VAE(timesteps=96, num_features=4, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss_function(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{epochs}  loss={total_loss/len(tensor):.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved → {model_path}")


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",       default=_PROJECT_ROOT, help="Project root directory")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=64)
    args = parser.parse_args()
    train(os.path.abspath(args.root), args.epochs, args.batch_size, args.latent_dim)


if __name__ == "__main__":
    main()
