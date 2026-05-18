"""
Improved peak estimation for L2 orderbook files.

Fixes over scripts/tag_peak_buckets.py:

  Fix 1 — Smoothed peak detection
    Raw argmax picks the single noisiest bar as the "peak". This version
    applies a rolling mean before argmax so the detected peak represents a
    sustained price move, not a one-bar spike.

  Fix 2 — Peak-at-window-edge handling
    The old script discarded any event where the peak fell in the last 2 bars
    (couldn't measure retracement). This version always computes retracement
    with whatever post-peak bars exist, and sets a `peak_truncated` flag in
    the meta.json so downstream code can down-weight these samples rather than
    silently drop them.

Usage:
    python scripts2/tag_peak_buckets.py                    # default: data/ dir
    python scripts2/tag_peak_buckets.py --root /my/data   # custom root
    python scripts2/tag_peak_buckets.py --overwrite        # re-tag existing
"""

import os
import sys
import re
import json
import glob
import argparse

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ─────────────────────────────────────────────────────────────────────

SCAN_BASES = ["real", "reconstructed", "synthetic"]

MIN_INCREASE          = 0.05   # 5% minimum price rise to be considered
RETRACEMENT_THRESHOLD = 0.30   # 30% of the gain must be given back

# Fix 1: rolling mean window for smoothing before argmax
SMOOTH_WINDOW = 5   # bars; adaptive — capped at len(mid)//5

# Fix 2: flag events where fewer than this many bars exist after the peak
MIN_POST_PEAK_BARS = 5

PEAK_BUCKETS = [
    ("micro",   0.05, 0.10),
    ("small",   0.10, 0.20),
    ("medium",  0.20, 0.30),
    ("large",   0.30, 0.40),
    ("major",   0.40, 0.50),
    ("extreme", 0.50, float("inf")),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_peak_bucket(increase: float) -> str:
    for name, lo, hi in PEAK_BUCKETS:
        if lo <= increase < hi:
            return name
    return "extreme"


def meta_path_for(l2_path: str) -> str:
    return re.sub(r'(_direct_L2|_synthetic_L2|_L2)\.csv$', '_meta.json', l2_path)


def regime_from_path(l2_path: str) -> str:
    parts = l2_path.replace("\\", "/").split("/")
    for p in parts:
        if p == "pumps":
            return "pump"
        if p in ("control", "normal", "uncertain"):
            return "control"
    return "unknown"


def _smooth(mid: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling mean with edge-padding so the output length matches the input.
    Uses 'edge' padding (repeats first/last value) to avoid boundary artifacts.
    """
    if window < 2 or len(mid) < window:
        return mid.copy()
    half = window // 2
    padded = np.pad(mid, half, mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(mid)]


def compute_peak(l2_path: str) -> dict:
    """
    Returns a dict with keys:
      increase        float   fractional price rise from bar 0 to peak
      retracement     float   fraction of gain given back after peak
      peak_idx        int     bar index of the detected peak
      peak_truncated  bool    True if fewer than MIN_POST_PEAK_BARS exist after peak
    """
    try:
        df = pd.read_csv(l2_path, usecols=["bid_price", "ask_price"])
        if len(df) < 5:
            return _null_peak()

        mid = ((df["bid_price"] + df["ask_price"]) / 2).values
        p0  = float(mid[0])
        if p0 <= 0:
            return _null_peak()

        # ── Fix 1: smooth before argmax ──────────────────────────────────────
        win      = max(3, min(SMOOTH_WINDOW, len(mid) // 5))
        smoothed = _smooth(mid, win)
        peak_idx = int(np.argmax(smoothed))

        # Use the raw mid value at the smoothed-identified peak index
        peak_val = float(mid[peak_idx])
        increase = (peak_val - p0) / p0

        if increase < MIN_INCREASE:
            return {"increase": increase, "retracement": 0.0,
                    "peak_idx": peak_idx, "peak_truncated": False}

        # ── Fix 2: always compute retracement with available post-peak data ──
        after          = mid[peak_idx:]
        peak_truncated = len(after) < MIN_POST_PEAK_BARS

        if len(after) < 2:
            # Peak is literally the last bar — no dump data at all
            return {"increase": increase, "retracement": 0.0,
                    "peak_idx": peak_idx, "peak_truncated": True}

        min_after   = float(after.min())
        price_range = peak_val - p0
        retracement = (peak_val - min_after) / price_range if price_range > 0 else 0.0

        return {
            "increase":       increase,
            "retracement":    retracement,
            "peak_idx":       peak_idx,
            "peak_truncated": peak_truncated,
        }

    except Exception:
        return _null_peak()


def _null_peak() -> dict:
    return {"increase": 0.0, "retracement": 0.0, "peak_idx": 0, "peak_truncated": False}


# ── Main ───────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Tag L2 files with peak metadata (improved)")
    parser.add_argument("--root",      default=_PROJECT_ROOT,
                        help=f"Project root directory (default: {_PROJECT_ROOT})")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-tag files that already have a _meta.json")
    args = parser.parse_args()

    root = os.path.abspath(args.root)

    tagged   = 0
    skipped  = 0
    failed   = 0
    truncated_flagged = 0

    for base_name in SCAN_BASES:
        base = os.path.join(root, base_name)
        if not os.path.isdir(base):
            print(f"  [SKIP] {base_name}/ not found under {root}")
            continue

        l2_files = [
            p for p in glob.glob(os.path.join(base, "**", "*_L2.csv"), recursive=True)
            if "_market_ctx" not in p
        ]

        print(f"\n{base_name}/ — {len(l2_files)} L2 files")

        for l2_path in l2_files:
            mp = meta_path_for(l2_path)

            if os.path.exists(mp) and not args.overwrite:
                skipped += 1
                continue

            if "_direct_L2.csv" in l2_path and os.path.exists(mp) and not args.overwrite:
                skipped += 1
                continue

            coin_regime = regime_from_path(l2_path)
            result      = compute_peak(l2_path)

            increase       = result["increase"]
            retracement    = result["retracement"]
            peak_idx       = result["peak_idx"]
            peak_truncated = result["peak_truncated"]

            is_pump = (
                coin_regime == "pump"
                and increase    >= MIN_INCREASE
                and retracement >= RETRACEMENT_THRESHOLD
            )

            meta = {
                "coin_regime":    coin_regime,
                "market_regime":  "unknown",
                "peak_pct":       round(increase * 100, 2) if coin_regime == "pump" else None,
                "peak_bucket":    get_peak_bucket(increase) if is_pump else None,
                "peak_idx":       int(peak_idx) if coin_regime == "pump" else None,
                "peak_truncated": peak_truncated,   # Fix 2: preserved in meta
            }

            try:
                with open(mp, "w") as f:
                    json.dump(meta, f, indent=2)
                tagged += 1
                if peak_truncated:
                    truncated_flagged += 1
            except Exception as e:
                print(f"  [ERR] {l2_path}: {e}")
                failed += 1

    print(f"\n{'='*60}")
    print(f"  Tagged              : {tagged}")
    print(f"  Truncated (flagged) : {truncated_flagged}  (peak near window edge)")
    print(f"  Skipped (have meta) : {skipped}")
    print(f"  Failed              : {failed}")
    print(f"{'='*60}")
    print("  Run fetch_market_context.py next to fill in market_regime.")


if __name__ == "__main__":
    main()
