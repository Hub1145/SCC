"""
Run peak estimation on downloaded data and save results to a CSV report.

Steps:
  1. Scans real/, reconstructed/, synthetic/ for *_L2.csv files
  2. Calls the improved compute_peak() (smoothed argmax + edge-safe retracement)
  3. Writes results to output/peak_estimation_results.csv
  4. Prints a summary table by exchange and regime

Usage:
    python scripts2/run_peak_estimation.py
    python scripts2/run_peak_estimation.py --root /workspace/Synthetic-Data
    python scripts2/run_peak_estimation.py --overwrite   # re-compute existing
"""

import os
import sys
import glob
import json
import argparse

import pandas as pd

# Re-use the improved helpers from tag_peak_buckets in the same folder
sys.path.insert(0, os.path.dirname(__file__))
from tag_peak_buckets import (
    compute_peak,
    compute_background_volatility,
    get_peak_bucket,
    meta_path_for,
    regime_from_path,
    MIN_INCREASE,
    RETRACEMENT_THRESHOLD,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCAN_BASES   = ["real", "reconstructed", "synthetic"]
OUTPUT_FILE  = "output/peak_estimation_results.csv"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Helpers ────────────────────────────────────────────────────────────────────

def exchange_from_path(path: str) -> str:
    """Extract exchange name from directory path."""
    parts = path.replace("\\", "/").split("/")
    known = {"binance","bybit","kucoin","okx","huobi","mexc","gateio","bitget"}
    for p in parts:
        if p.lower() in known:
            return p.lower()
    # Fallback: first segment after the tier
    for i, p in enumerate(parts):
        if p in SCAN_BASES and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def load_existing_meta(meta_path: str) -> dict:
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return {}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Peak estimation report")
    parser.add_argument("--root",      default=_PROJECT_ROOT,
                        help=f"Project root (default: {_PROJECT_ROOT})")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-compute peaks even if meta.json already exists")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_path = os.path.join(root, OUTPUT_FILE)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    records = []

    for base_name in SCAN_BASES:
        base = os.path.join(root, base_name)
        if not os.path.isdir(base):
            print(f"  [SKIP] {base_name}/ not found")
            continue

        l2_files = [
            p for p in glob.glob(os.path.join(base, "**", "*_L2.csv"), recursive=True)
            if "_market_ctx" not in p
        ]

        print(f"\nScanning {base_name}/ — {len(l2_files)} L2 files ...")

        for i, l2_path in enumerate(l2_files, 1):
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(l2_files)}")

            mp          = meta_path_for(l2_path)
            coin_regime = regime_from_path(l2_path)
            exchange    = exchange_from_path(l2_path)

            # Use cached meta if available and not overwriting
            if os.path.exists(mp) and not args.overwrite:
                meta           = load_existing_meta(mp)
                increase       = (meta.get("peak_pct") or 0) / 100
                retracement    = meta.get("retracement", 0.0)
                peak_idx       = meta.get("peak_idx") or 0
                peak_truncated = meta.get("peak_truncated", False)
                market_regime  = meta.get("market_regime", "unknown")
                bg_volatility  = meta.get("background_volatility", "unknown")
            else:
                result         = compute_peak(l2_path)
                increase       = result["increase"]
                retracement    = result["retracement"]
                peak_idx       = result["peak_idx"]
                peak_truncated = result["peak_truncated"]
                market_regime  = "unknown"
                bg_volatility  = compute_background_volatility(l2_path)

                is_pump = (
                    coin_regime == "pump"
                    and increase    >= MIN_INCREASE
                    and retracement >= RETRACEMENT_THRESHOLD
                )
                existing = load_existing_meta(mp)
                existing.update({
                    "coin_regime":          coin_regime,
                    "background_volatility": bg_volatility,
                    "peak_pct":             round(increase * 100, 2) if coin_regime == "pump" else None,
                    "peak_bucket":          get_peak_bucket(increase) if is_pump else None,
                    "peak_idx":             int(peak_idx) if coin_regime == "pump" else None,
                    "peak_truncated":       peak_truncated,
                })
                try:
                    with open(mp, "w") as f:
                        import json as _json
                        _json.dump(existing, f, indent=2)
                except Exception:
                    pass

            is_pump = (
                coin_regime == "pump"
                and increase    >= MIN_INCREASE
                and retracement >= RETRACEMENT_THRESHOLD
            )

            records.append({
                "tier":                 base_name,
                "exchange":             exchange,
                "coin_regime":          coin_regime,
                "market_regime":        market_regime,
                "background_volatility": bg_volatility,
                "increase_pct":         round(increase * 100, 2),
                "retracement_pct":      round(retracement * 100, 2),
                "peak_idx":             peak_idx,
                "peak_bucket":          get_peak_bucket(increase) if is_pump else "",
                "peak_truncated":       peak_truncated,
                "confirmed_pump":       is_pump,
                "file":                 os.path.relpath(l2_path, root),
            })

    if not records:
        print("\nNo L2 files found — nothing to report.")
        return

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"\n\nResults saved to: {out_path}")
    print(f"Total files processed: {len(df)}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  PEAK ESTIMATION SUMMARY")
    print("="*70)

    pumps = df[df["confirmed_pump"]]
    print(f"\n  Confirmed pumps (≥{int(MIN_INCREASE*100)}% rise, ≥{int(RETRACEMENT_THRESHOLD*100)}% retracement): {len(pumps)}")
    print(f"  Truncated events (peak near window edge):  {df['peak_truncated'].sum()}")
    print(f"  Control events:                            {len(df) - len(df[df['coin_regime']=='pump'])}")

    print("\n  By exchange and tier (confirmed pumps):")
    print(f"  {'Exchange':<12} {'Tier':<15} {'Pumps':>7} {'Mean rise%':>12} {'Mean retracement%':>18}")
    print("  " + "-"*64)

    by_exchange = []
    for (exch, tier), grp in pumps.groupby(["exchange", "tier"]):
        print(f"  {exch:<12} {tier:<15} {len(grp):>7} "
              f"{grp['increase_pct'].mean():>12.1f} "
              f"{grp['retracement_pct'].mean():>18.1f}")
        by_exchange.append({
            "exchange":          exch,
            "tier":              tier,
            "confirmed_pumps":   int(len(grp)),
            "mean_increase_pct": round(float(grp["increase_pct"].mean()), 2),
            "mean_retracement_pct": round(float(grp["retracement_pct"].mean()), 2),
        })

    print("\n  Peak bucket distribution (confirmed pumps):")
    bucket_counts = pumps["peak_bucket"].value_counts()
    bucket_dist = {}
    for bucket, count in bucket_counts.items():
        print(f"    {bucket:<10} {count:>6}")
        bucket_dist[bucket] = int(count)

    print("\n" + "="*70)

    # ── JSON summary ───────────────────────────────────────────────────────────
    import json as _json
    print("\n  Background volatility distribution (all files):")
    vol_counts = df["background_volatility"].value_counts()
    vol_dist   = {}
    for vol, count in vol_counts.items():
        print(f"    {vol:<10} {count:>6}")
        vol_dist[vol] = int(count)

    json_path = out_path.replace(".csv", "_summary.json")
    summary = {
        "total_files":       len(df),
        "confirmed_pumps":   int(len(pumps)),
        "truncated_events":  int(df["peak_truncated"].sum()),
        "control_events":    int(len(df) - len(df[df["coin_regime"] == "pump"])),
        "thresholds": {
            "min_increase_pct":    int(MIN_INCREASE * 100),
            "min_retracement_pct": int(RETRACEMENT_THRESHOLD * 100),
        },
        "by_exchange":              by_exchange,
        "bucket_distribution":      bucket_dist,
        "volatility_distribution":  vol_dist,
    }
    with open(json_path, "w") as f:
        _json.dump(summary, f, indent=2)
    print(f"  JSON summary saved to: {json_path}")


if __name__ == "__main__":
    main()
