"""
Train PumpDetectorV3 (dual-stream CNN) on data downloaded from Google Drive.

Handles both lowercase (reconstructed/) and capitalised (Reconstructed/) names.

Usage:
    python scripts2/train_pump_detector.py
    python scripts2/train_pump_detector.py --root /workspace/Synthetic-Data
    python scripts2/train_pump_detector.py --epochs 40 --no-zero-shot
"""

import os
import sys
import re
import json
import glob
import argparse

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_fscore_support

SCRIPTS2_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Constants ──────────────────────────────────────────────────────────────────

TIMESTEPS       = 96
L2_FEATURES     = ['bid_price', 'ask_price', 'bid_size', 'ask_size']
TRADE_FEATURES  = ['buy_ratio', 'aggressor_imbalance']
ALL_FEATURES    = L2_FEATURES + TRADE_FEATURES
NUM_FEATURES    = len(ALL_FEATURES)

TRAIN_EXCHANGES = ['binance', 'kucoin', 'huobi', 'mexc', 'gateio', 'bitget']
TEST_EXCHANGES  = ['bybit', 'okx']

CELL_WEIGHTS = {
    ("pump",    "normal"):    3.0,
    ("pump",    "uncertain"): 2.0,
    ("pump",    "pumped"):    1.0,
    ("control", "normal"):    2.5,
    ("control", "uncertain"): 1.5,
    ("control", "pumped"):    1.0,
}

NEUTRAL_BUY_RATIO = 0.5
NEUTRAL_AGGRESSOR = 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_bases(root: str) -> list[str]:
    """Return existing tier directories, accepting both cases."""
    found = []
    for name in ("reconstructed", "Reconstructed", "synthetic", "Synthetic"):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            found.append(p)
    return found


def _load_trade_features(trades_path: str) -> np.ndarray:
    out = np.full((TIMESTEPS, 2), [NEUTRAL_BUY_RATIO, NEUTRAL_AGGRESSOR], dtype=np.float32)
    try:
        df = pd.read_csv(trades_path)
        if not {"timestamp_idx", "side", "size"}.issubset(df.columns):
            return out
        df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0)
        for idx, grp in df.groupby("timestamp_idx"):
            if not (0 <= idx < TIMESTEPS):
                continue
            total = grp["size"].sum()
            if total <= 0:
                continue
            buy_vol  = grp.loc[grp["side"] == "buy", "size"].sum()
            sell_vol = total - buy_vol
            out[int(idx), 0] = buy_vol / total
            out[int(idx), 1] = (buy_vol - sell_vol) / total
    except Exception:
        pass
    return out


_NEUTRAL_MARKET = None

def _neutral_market() -> np.ndarray:
    global _NEUTRAL_MARKET
    if _NEUTRAL_MARKET is None:
        _NEUTRAL_MARKET = np.column_stack([
            np.ones(TIMESTEPS), np.ones(TIMESTEPS),
            np.ones(TIMESTEPS), np.ones(TIMESTEPS),
            np.full(TIMESTEPS, 0.5), np.zeros(TIMESTEPS),
        ]).astype(np.float32)
    return _NEUTRAL_MARKET.copy()


def _load_market_ctx(l2_path: str) -> np.ndarray:
    ctx = re.sub(r'(_direct_L2|_synthetic_L2|_L2)\.csv$', '_market_ctx.csv', l2_path)
    if not os.path.exists(ctx):
        return _neutral_market()
    try:
        df = pd.read_csv(ctx)
        needed = ['bid_price', 'ask_price', 'bid_size', 'ask_size']
        if not set(needed).issubset(df.columns):
            return _neutral_market()
        data = df[needed].values[:TIMESTEPS].astype(np.float32)
        if len(data) < TIMESTEPS:
            return _neutral_market()
        if 'buy_ratio' in df.columns and 'aggressor_imbalance' in df.columns:
            tf = df[['buy_ratio', 'aggressor_imbalance']].values[:TIMESTEPS].astype(np.float32)
        else:
            tf = np.column_stack([np.full(TIMESTEPS, 0.5), np.zeros(TIMESTEPS)]).astype(np.float32)
        return np.concatenate([data, tf], axis=1)
    except Exception:
        return _neutral_market()


def _cell_weight(l2_path: str, label: int) -> float:
    mp = re.sub(r'(_direct_L2|_synthetic_L2|_L2)\.csv$', '_meta.json', l2_path)
    coin_regime   = "pump" if label == 1 else "control"
    market_regime = "normal"
    if os.path.exists(mp):
        try:
            with open(mp) as f:
                meta = json.load(f)
            coin_regime   = meta.get("coin_regime",   coin_regime)
            market_regime = meta.get("market_regime", "normal")
            if market_regime == "unknown":
                market_regime = "normal"
        except Exception:
            pass
    return CELL_WEIGHTS.get((coin_regime, market_regime), 1.5)


def _normalize(w: np.ndarray) -> np.ndarray:
    w = w.copy().astype(np.float32)
    mid0 = (w[0, 0] + w[0, 1]) / 2.0
    if mid0 > 0:
        w[:, 0:2] /= mid0
    mean_sz = w[:, 2:4].mean()
    if mean_sz > 0:
        w[:, 2:4] /= mean_sz
    return w


def _extract_windows(df: pd.DataFrame, l2_path: str, label: int):
    l2_data  = df[L2_FEATURES].values.astype(np.float32)
    n        = len(l2_data)
    tp       = re.sub(r'(_direct_L2|_synthetic_L2|_L2)\.csv$', '_trades.csv', l2_path)
    tf       = _load_trade_features(tp) if os.path.exists(tp) else \
               np.full((n, 2), [NEUTRAL_BUY_RATIO, NEUTRAL_AGGRESSOR], dtype=np.float32)
    coin_all = np.concatenate([l2_data, tf[:n]], axis=1)
    mkt      = _load_market_ctx(l2_path)
    cw       = _cell_weight(l2_path, label)

    step     = max(1, TIMESTEPS // 2)
    coins, mkts, wts = [], [], []
    for s in range(0, n - TIMESTEPS + 1, step):
        wc = coin_all[s:s + TIMESTEPS]
        if len(wc) != TIMESTEPS or not np.isfinite(wc).all():
            continue
        wm = mkt[s:s + TIMESTEPS] if len(mkt) > TIMESTEPS else mkt
        coins.append(_normalize(wc))
        mkts.append(_normalize(wm))
        wts.append(cw)
    return coins, mkts, wts


# ── Dataset ────────────────────────────────────────────────────────────────────

class PumpDataset(Dataset):
    def __init__(self, X_coin, X_mkt, y, weights):
        self.xc = torch.FloatTensor(X_coin)
        self.xm = torch.FloatTensor(X_mkt)
        self.y  = torch.FloatTensor(y)
        self.w  = weights

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.xc[i], self.xm[i], self.y[i]


def _make_loader(ds: PumpDataset, bs: int, shuffle: bool) -> DataLoader:
    if shuffle:
        y       = ds.y.numpy().astype(int)
        counts  = np.bincount(y, minlength=2).clip(min=1)
        cls_w   = (1.0 / counts)[y]
        sampler = WeightedRandomSampler(cls_w * ds.w, num_samples=len(ds))
        return DataLoader(ds, batch_size=bs, sampler=sampler)
    return DataLoader(ds, batch_size=bs, shuffle=False)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset(exchanges: list, bases: list):
    Xc, Xm, y, w = [], [], [], []
    for base in bases:
        for exchange in exchanges:
            exch_dir = os.path.join(base, exchange)
            if not os.path.isdir(exch_dir):
                continue
            for regime_dir in os.listdir(exch_dir):
                label = 1 if regime_dir == "pumps" else 0
                rp = os.path.join(exch_dir, regime_dir)
                if not os.path.isdir(rp):
                    continue
                files = [p for p in glob.glob(os.path.join(rp, "**", "*_L2.csv"), recursive=True)
                         if "_market_ctx" not in p]
                n_win = 0
                for fp in files:
                    try:
                        df = pd.read_csv(fp)
                        miss = [c for c in L2_FEATURES if c not in df.columns]
                        if miss or len(df) < TIMESTEPS:
                            continue
                        df = df.dropna(subset=L2_FEATURES)
                        coins, mkts, wts = _extract_windows(df, fp, label)
                        Xc.extend(coins); Xm.extend(mkts)
                        y.extend([label] * len(coins))
                        w.extend(wts)
                        n_win += len(coins)
                    except Exception as e:
                        print(f"  [WARN] {fp}: {e}")
                if files:
                    print(f"  {exchange}/{regime_dir}: {len(files)} files → {n_win} windows")

    empty = np.empty((0, TIMESTEPS, NUM_FEATURES), dtype=np.float32)
    if not Xc:
        return empty, empty, np.empty(0, np.float32), np.empty(0, np.float32)
    return (np.array(Xc, np.float32), np.array(Xm, np.float32),
            np.array(y,  np.float32), np.array(w,  np.float32))


# ── Training ───────────────────────────────────────────────────────────────────

def train(root: str, epochs: int, batch_size: int, no_zero_shot: bool):
    if root not in sys.path:
        sys.path.insert(0, root)
    from models.pump_detector import PumpDetectorV3

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(root, "models", "pump_detector_v3.pth")
    bases      = _find_bases(root)

    if not bases:
        print(f"[ERROR] No reconstructed/ or synthetic/ directory found under {root}")
        sys.exit(1)

    print("=" * 65)
    print("  PumpDetectorV3  (dual-stream: coin + market context)")
    print(f"  Train: {TRAIN_EXCHANGES}")
    print(f"  Test : {TEST_EXCHANGES}  (zero-shot)")
    print(f"  Device: {device}  |  Epochs: {epochs}")
    print("=" * 65)

    print("\nLoading training data ...")
    Xc_tr, Xm_tr, y_tr, w_tr = load_dataset(TRAIN_EXCHANGES, bases)
    if len(Xc_tr) == 0:
        print("[ERROR] No training windows found.")
        sys.exit(1)
    print(f"\n  Train: {len(Xc_tr)} windows  "
          f"({int(y_tr.sum())} pump, {int((y_tr==0).sum())} control)")

    if not no_zero_shot:
        print("\nLoading zero-shot test data ...")
        Xc_te, Xm_te, y_te, w_te = load_dataset(TEST_EXCHANGES, bases)
        if len(Xc_te):
            print(f"  Test : {len(Xc_te)} windows  "
                  f"({int(y_te.sum())} pump, {int((y_te==0).sum())} control)")
    else:
        Xc_te = np.empty((0,))

    idx   = np.random.permutation(len(Xc_tr))
    split = int(0.8 * len(idx))
    tr_i, val_i = idx[:split], idx[split:]

    tr_ds  = PumpDataset(Xc_tr[tr_i],  Xm_tr[tr_i],  y_tr[tr_i],  w_tr[tr_i])
    val_ds = PumpDataset(Xc_tr[val_i], Xm_tr[val_i], y_tr[val_i], w_tr[val_i])

    tr_loader  = _make_loader(tr_ds,  batch_size, shuffle=True)
    val_loader = _make_loader(val_ds, batch_size, shuffle=False)

    model     = PumpDetectorV3(NUM_FEATURES, NUM_FEATURES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=6, factor=0.5)
    criterion = nn.BCELoss()

    best_auc  = 0.0
    epoch_log = []
    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val AUC':>9}")
    print("-" * 34)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xc, xm, yb in tr_loader:
            xc, xm, yb = xc.to(device), xm.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xc, xm), yb)
            loss.backward(); optimizer.step()
            total_loss += loss.item()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xc, xm, yb in val_loader:
                preds.extend(model(xc.to(device), xm.to(device)).cpu().numpy())
                labels.extend(yb.numpy())

        avg_loss = total_loss / len(tr_loader)
        val_auc  = roc_auc_score(labels, preds)
        scheduler.step(val_auc)
        epoch_log.append({"epoch": epoch, "train_loss": round(avg_loss, 6),
                           "val_auc": round(val_auc, 6)})
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {avg_loss:>12.4f}  {val_auc:>9.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

    print("-" * 34)
    print(f"  Best val AUC : {best_auc:.4f}")
    print(f"  Saved        : {model_path}")

    # ── JSON output ────────────────────────────────────────────────────────────
    json_result = {
        "config": {
            "epochs":           epochs,
            "batch_size":       batch_size,
            "train_exchanges":  TRAIN_EXCHANGES,
            "test_exchanges":   TEST_EXCHANGES,
            "device":           str(device),
            "timesteps":        TIMESTEPS,
            "features":         ALL_FEATURES,
        },
        "dataset": {
            "train_windows":   int(len(Xc_tr)),
            "train_pump":      int(y_tr.sum()),
            "train_control":   int((y_tr == 0).sum()),
        },
        "training": {
            "best_val_auc": round(best_auc, 6),
            "model_path":   model_path,
            "epochs":       epoch_log,
        },
        "zero_shot": None,
    }

    # Zero-shot eval
    if not no_zero_shot and len(Xc_te):
        print("\n" + "=" * 65)
        print("  Zero-Shot Test  (Bybit + OKX)")
        print("=" * 65)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        te_ds  = PumpDataset(Xc_te, Xm_te, y_te, w_te)
        te_ldr = _make_loader(te_ds, batch_size, shuffle=False)
        preds, labels = [], []
        with torch.no_grad():
            for xc, xm, yb in te_ldr:
                preds.extend(model(xc.to(device), xm.to(device)).cpu().numpy())
                labels.extend(yb.numpy())
        preds  = np.array(preds); labels = np.array(labels)
        te_auc = roc_auc_score(labels, preds)
        binary = (preds >= 0.5).astype(int)
        print(f"\n  Zero-shot AUC : {te_auc:.4f}\n")
        print(classification_report(labels, binary,
                                    target_names=["control", "pump"], zero_division=0))
        grade = ("Strong" if te_auc >= 0.80 else
                 "Moderate" if te_auc >= 0.65 else "Weak")
        print(f"  RESULT: {grade} cross-exchange generalisation.")

        p, r, f, s = precision_recall_fscore_support(labels, binary, zero_division=0)
        json_result["zero_shot"] = {
            "auc":          round(float(te_auc), 6),
            "grade":        grade,
            "test_windows": int(len(Xc_te)),
            "test_pump":    int(y_te.sum()),
            "test_control": int((y_te == 0).sum()),
            "per_class": {
                "control": {"precision": round(float(p[0]), 4), "recall": round(float(r[0]), 4),
                            "f1": round(float(f[0]), 4), "support": int(s[0])},
                "pump":    {"precision": round(float(p[1]), 4), "recall": round(float(r[1]), 4),
                            "f1": round(float(f[1]), 4), "support": int(s[1])},
            },
        }

    json_path = os.path.join(root, "output", "pump_detector_training.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(json_result, f, indent=2)
    print(f"\n  JSON results saved to: {json_path}")


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",         default=_PROJECT_ROOT, help="Project root directory")
    parser.add_argument("--epochs",       type=int, default=60)
    parser.add_argument("--batch-size",   type=int, default=64)
    parser.add_argument("--no-zero-shot", action="store_true",
                        help="Skip zero-shot eval (if Bybit/OKX data unavailable)")
    args = parser.parse_args()
    train(os.path.abspath(args.root), args.epochs, args.batch_size, args.no_zero_shot)


if __name__ == "__main__":
    main()
