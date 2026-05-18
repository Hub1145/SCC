"""
scripts2/main.py — full pipeline: Drive download → peak estimation → model training.

Steps:
  1. Download data from Google Drive (rclone)
  2. Tag L2 files with improved peak metadata (smoothed peak + edge-safe retracement)
  3. Prepare DirectL2VAE training data
  4. Train DirectL2VAE (Type-B synthetic generator)
  5. Train PumpDetectorV3 (dual-stream CNN)
  6. Run peak estimation report

Usage:
    python scripts2/main.py
    python scripts2/main.py --root /workspace/Synthetic-Data
    python scripts2/main.py --skip-download          # data already on disk
    python scripts2/main.py --skip-vae               # skip VAE training
    python scripts2/main.py --train-only             # skip download + peak tag
    python scripts2/main.py --no-zero-shot           # skip Bybit/OKX eval
    python scripts2/main.py --epochs 40
    python scripts2/main.py --continue-on-error      # don't halt on step failure
"""

import os
import sys
import time
import argparse
import subprocess

SCRIPTS2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS2_DIR)   # parent of scripts2/


# ── Step runner ────────────────────────────────────────────────────────────────

def _run(step_num: int, label: str, script: str,
         extra_args: list, root: str, continue_on_error: bool) -> bool:
    print(f"\n{'='*68}")
    print(f"  Step {step_num:02d}  —  {label}")
    print(f"{'='*68}")
    t0 = time.time()

    cmd = [sys.executable, os.path.join(SCRIPTS2_DIR, script),
           "--root", root] + extra_args

    result  = subprocess.run(cmd)
    elapsed = time.time() - t0
    ok      = result.returncode == 0
    status  = "OK" if ok else "FAILED"

    print(f"\n  [{status}]  {label}  ({elapsed:.1f}s)")

    if not ok and not continue_on_error:
        print("  Pipeline halted. Use --continue-on-error to keep going.")
        sys.exit(1)

    return ok


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="scripts2 full pipeline: Drive → peaks → VAE → PumpDetectorV3")
    parser.add_argument("--root",             default=PROJECT_ROOT,
                        help=f"Project root (default: {PROJECT_ROOT})")
    parser.add_argument("--skip-download",    action="store_true",
                        help="Skip Google Drive download (data already on disk)")
    parser.add_argument("--skip-peak-tag",    action="store_true",
                        help="Skip peak tagging (meta.json already present)")
    parser.add_argument("--skip-vae",         action="store_true",
                        help="Skip DirectL2VAE training")
    parser.add_argument("--train-only",       action="store_true",
                        help="Skip download + peak tag; go straight to training")
    parser.add_argument("--epochs",           type=int, default=60,
                        help="Epochs for PumpDetectorV3 (default 60)")
    parser.add_argument("--vae-epochs",       type=int, default=100,
                        help="Epochs for DirectL2VAE (default 100)")
    parser.add_argument("--batch-size",       type=int, default=64)
    parser.add_argument("--no-zero-shot",     action="store_true",
                        help="Skip Bybit/OKX zero-shot eval")
    parser.add_argument("--overwrite",        action="store_true",
                        help="Re-compute peaks even if meta.json exists")
    parser.add_argument("--dry-run",          action="store_true",
                        help="Dry-run the download step")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue pipeline even if a step fails")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    t_total = time.time()

    print(f"\n{'='*68}")
    print(f"  SCRIPTS2 PIPELINE")
    print(f"  Root   : {root}")
    print(f"  Epochs : {args.epochs} (detector) / {args.vae_epochs} (VAE)")
    print(f"{'='*68}")

    step = 1

    # Step 1 — download
    if not args.skip_download and not args.train_only:
        dl_args = ["--dry-run"] if args.dry_run else []
        _run(step, "Download from Google Drive", "fetch_from_drive.py",
             dl_args, root, args.continue_on_error)
    else:
        print(f"\n  [SKIP] Step {step:02d} — Download")
    step += 1

    # Step 2 — peak tagging
    if not args.skip_peak_tag and not args.train_only:
        tag_args = ["--overwrite"] if args.overwrite else []
        _run(step, "Tag peak buckets (smoothed + edge-safe)", "tag_peak_buckets.py",
             tag_args, root, args.continue_on_error)
    else:
        print(f"\n  [SKIP] Step {step:02d} — Peak tagging")
    step += 1

    # Step 3 — prepare VAE data
    if not args.skip_vae:
        _run(step, "Prepare DirectL2VAE training data", "prepare_l2_training_data.py",
             [], root, args.continue_on_error)
    else:
        print(f"\n  [SKIP] Step {step:02d} — Prepare VAE data")
    step += 1

    # Step 4 — train VAE
    if not args.skip_vae:
        vae_args = [f"--epochs={args.vae_epochs}"]
        _run(step, "Train DirectL2VAE", "train_direct_l2.py",
             vae_args, root, args.continue_on_error)
    else:
        print(f"\n  [SKIP] Step {step:02d} — Train DirectL2VAE")
    step += 1

    # Step 5 — train PumpDetectorV3
    det_args = [f"--epochs={args.epochs}", f"--batch-size={args.batch_size}"]
    if args.no_zero_shot:
        det_args.append("--no-zero-shot")
    _run(step, "Train PumpDetectorV3", "train_pump_detector.py",
         det_args, root, args.continue_on_error)
    step += 1

    # Step 6 — peak estimation report
    rpt_args = ["--overwrite"] if args.overwrite else []
    _run(step, "Peak estimation report", "run_peak_estimation.py",
         rpt_args, root, args.continue_on_error)

    elapsed = time.time() - t_total
    h, m    = divmod(int(elapsed), 3600)
    m, s    = divmod(m, 60)
    print(f"\n{'='*68}")
    print(f"  Pipeline complete  ({h}h {m}m {s}s)")
    print(f"  Results : {os.path.join(root, 'output', 'peak_estimation_results.csv')}")
    print(f"  Model   : {os.path.join(root, 'models', 'pump_detector_v3.pth')}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
