# scripts2 — Drive-Based Peak Estimation & Model Training Pipeline

`scripts2/` is a self-contained pipeline that downloads the pump-and-dump dataset
from Google Drive, runs improved peak estimation on all L2 orderbook files, trains
the DirectL2VAE generator and the PumpDetectorV3 classifier, and writes a full
results report. It is the successor to `scripts/` with two key improvements:
smoother peak detection and edge-safe retracement measurement.

---

## Quick Start

```bash
# 1. Install all Python dependencies
pip install -r scripts2/requirements.txt

# 2. Run the full pipeline from inside scripts2/
cd scripts2
python main.py

# Or from the project root
python scripts2/main.py
```

No API key setup required — a key is already embedded in `fetch_from_drive.py`.
If you ever need to use your own key, pass it with `--key AIza...` or set the
`GOOGLE_API_KEY` environment variable (the script checks both).

**The `--root` flag is optional.** Every script in `scripts2/` automatically
resolves the project root as its own parent directory, so `python main.py` from
inside `scripts2/` and `python scripts2/main.py` from the project root both point
to the same correct paths. You only need `--root` if you are running with data
stored somewhere other than the default location.

---

## Pipeline Steps

```
Step 01  fetch_from_drive.py          Download data from Google Drive (API v3)
Step 02  tag_peak_buckets.py          Tag each L2 file with improved peak metadata
Step 03  prepare_l2_training_data.py  Build VAE training samples + fit scaler
Step 04  train_direct_l2.py           Train DirectL2VAE (Type-B synthetic generator)
Step 05  train_pump_detector.py       Train PumpDetectorV3 (dual-stream CNN)
Step 06  run_peak_estimation.py       Write full peak estimation report
```

All six steps run in sequence when you call `python main.py`. Use flags to skip
steps you have already completed (see the Flags Reference section below).

---

## Google Drive Download (`fetch_from_drive.py`)

### Why not gdown

Google changed their internal Drive page format in 2024. The JavaScript variable
`window['_DRIVE_ivd']` that gdown parses now contains `null` instead of the folder
listing JSON. Both gdown v4 and v5 crash with `JSONDecodeError` on every public
folder regardless of version. This script uses the official **Google Drive API v3**
via `google-api-python-client` which is stable and fully pip-installable.

### Setup

**pip install** (already covered by requirements.txt — nothing else needed):

```bash
pip install google-api-python-client
```

No OAuth, no browser login, no system binary to install, no API key to create.
A working API key is already embedded in `fetch_from_drive.py`. If you want to
use your own key instead, pass it at runtime or set an environment variable:

```bash
python scripts2/fetch_from_drive.py --key AIza...          # inline
export GOOGLE_API_KEY="AIza..."                            # Linux/Mac env
$env:GOOGLE_API_KEY="AIza..."                              # Windows PowerShell
```

### Capitalised folder names

The Google Drive folder uses **capital names** (`Real/`, `Reconstructed/`,
`Synthetic/`). The download script maps each to its **lowercase local name**
(`real/`, `reconstructed/`, `synthetic/`) automatically. All scripts2 scripts
also accept both cases when scanning directories at runtime.

### Running the download standalone

```bash
python scripts2/fetch_from_drive.py                     # all tiers
python scripts2/fetch_from_drive.py --tier Real         # one tier only
python scripts2/fetch_from_drive.py --dry-run           # list files without downloading
python scripts2/fetch_from_drive.py --key AIza...       # pass API key inline
python scripts2/fetch_from_drive.py --root /my/data     # custom output root
```

---

## Peak Estimation

### Improvements over `scripts/tag_peak_buckets.py`

The original script had two limitations that caused it to misidentify peaks and
silently discard real pump events. Both are fixed in `scripts2`.

---

**Fix 1 — Smoothed peak detection**

*Problem:* The original used `numpy.argmax` directly on the raw mid-price series.
A single noisy bar (wide spread, bad tick) could be picked as the "peak" even if
the actual pump peak was several bars earlier at a lower but more sustained price.

*Fix:* A **rolling mean** (window = 5 bars, adaptive — capped at `len(mid) // 5`)
is applied to the mid-price curve before `argmax`. The detected peak index now
points at a sustained price move rather than a one-bar artefact. The raw mid-price
value at that index is used for the actual percentage calculation, so the smoothing
changes which bar is selected but does not distort the numbers.

---

**Fix 2 — Peak-at-window-edge handling**

*Problem:* The original skipped any event where `peak_idx >= len(mid) - 2` and
set `retracement = 0.0`. This silently discarded real pump events where the window
happened to end before the dump fully played out.

*Fix:* Retracement is always computed using whatever post-peak bars exist, even if
only 2–4 bars are available. When fewer than 5 bars exist after the peak, the file
gets `"peak_truncated": true` written to its `_meta.json`. Training code can use
this flag to down-weight those samples rather than silently drop them.

---

### Peak buckets

Events are classified into six magnitude buckets based on the price increase from
the first bar to the detected peak:

| Bucket  | Rise range |
|---------|------------|
| micro   | 5% – 10%   |
| small   | 10% – 20%  |
| medium  | 20% – 30%  |
| large   | 30% – 40%  |
| major   | 40% – 50%  |
| extreme | 50%+       |

### What gets written: `_meta.json`

Every L2 file gets a paired `_meta.json` written alongside it in the same
directory. The file is created if it does not exist, or updated if it does.

```json
{
  "coin_regime":    "pump",
  "market_regime":  "normal",
  "peak_pct":       23.45,
  "peak_bucket":    "medium",
  "peak_idx":       47,
  "peak_truncated": false
}
```

| Field            | Description |
|------------------|-------------|
| `coin_regime`    | `"pump"` or `"control"` — inferred from the directory name (`pumps/` → pump) |
| `market_regime`  | BTC context at the time of the event: `"normal"`, `"uncertain"`, `"pumped"`, or `"unknown"` |
| `peak_pct`       | Price increase from bar 0 to the detected peak, as a percentage. `null` for control files |
| `peak_bucket`    | Magnitude category. `null` for control files or unconfirmed pumps |
| `peak_idx`       | Bar index of the detected peak. `null` for control files |
| `peak_truncated` | `true` if fewer than 5 bars exist after the peak — retracement may be underestimated |

Control files (`normal/`, `uncertain/`) get `peak_pct: null`, `peak_bucket: null`,
`peak_idx: null` because peak statistics are not meaningful for non-pump events.

---

## Model Training

### Step 3 — Prepare training data (`prepare_l2_training_data.py`)

Scans `reconstructed/` and `real/` (both cases accepted), collects every
`*_L2.csv` file, takes the first 96 bars from each, normalises prices relative to
the first mid-price of each window, fits a `StandardScaler`, and saves the result.

**Saved outputs:**

| File | Description |
|------|-------------|
| `data/l2_training_samples.npy` | Normalised training array `[N, 96, 4]` |
| `models/l2_scaler.pkl` | Fitted StandardScaler used by DirectL2VAE at inference |

---

### Step 4 — DirectL2VAE (`train_direct_l2.py`)

A Conv1D variational autoencoder that learns the distribution of real L2 orderbook
sequences. Once trained it can generate new realistic synthetic orderbook windows
for data augmentation.

**Architecture:**

```
Input  [batch, 96, 4]
  └─ Encoder: Conv1d(4→32) → Conv1d(32→64) → Conv1d(64→128) → Linear(256) → μ, log σ²
  └─ Reparameterise: z = μ + ε·σ   (latent dim = 64)
  └─ Decoder: Linear → ConvTranspose1d(128→64) → ConvTranspose1d(64→32) → ConvTranspose1d(32→4)
Output [batch, 96, 4]
```

Features: `bid_price`, `ask_price`, `bid_size`, `ask_size`.

**Saved outputs:**

| File | Description |
|------|-------------|
| `models/direct_l2_vae_v1.pth` | Trained VAE weights (saved at final epoch) |

**Configuration:**

```bash
python scripts2/train_direct_l2.py --epochs 100 --batch-size 32 --latent-dim 64
```

---

### Step 5 — PumpDetectorV3 (`train_pump_detector.py`)

A dual-stream 1D-CNN that classifies L2 orderbook windows as pump or control.
It uses two separate CNN towers — one for the coin being inspected, one for the
BTC market context — and combines their outputs to detect pumps that are
anomalous relative to general market conditions.

**Architecture:**

```
x_coin   [96, 6]  ──►  3-block Conv1D tower  ──►  [128]  ─┐
                                                             ├─► concat [256] → Linear(64) → Sigmoid → pump prob
x_market [96, 6]  ──►  3-block Conv1D tower  ──►  [128]  ─┘
```

Each Conv1D tower:
- Block 1: Conv1d(k=5) → BatchNorm → ReLU → Conv1d(k=3) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.1)
- Block 2: same structure, wider channels
- Block 3: Conv1d(k=3) → BatchNorm → ReLU → AdaptiveAvgPool(1) → Flatten → `[128]`

**6 features per stream:**

| # | Feature | Source |
|---|---------|--------|
| 1 | `bid_price` | L2 orderbook |
| 2 | `ask_price` | L2 orderbook |
| 3 | `bid_size` | L2 orderbook |
| 4 | `ask_size` | L2 orderbook |
| 5 | `buy_ratio` | `_trades.csv` if present, otherwise neutral fill (0.5) |
| 6 | `aggressor_imbalance` | `_trades.csv` if present, otherwise neutral fill (0.0) |

**3×3 regime matrix and sample weights:**

Each training window is weighted by the combination of its coin label and the BTC
market state at the time. The most suspicious combination — coin pumping while BTC
is flat — receives the highest training signal so the model learns to distinguish
true coordinated pumps from general market moves.

| Coin label | BTC regime | Sample weight | Rationale |
|------------|------------|:---:|-----------|
| pump       | normal     | **3.0** | Most suspicious — coin up, market flat |
| pump       | uncertain  | 2.0     | Ambiguous — could be partial correlation |
| pump       | pumped     | 1.0     | Least suspicious — general market up |
| control    | normal     | **2.5** | Hard negative — market flat, coin flat |
| control    | uncertain  | 1.5     | Medium negative |
| control    | pumped     | 1.0     | Easy negative — general up market |

**Train / test exchange split:**

| Role | Exchanges |
|------|-----------|
| Training (seen) | Binance, KuCoin, Huobi, MEXC, Gate.io, Bitget |
| Zero-shot test (unseen) | Bybit, OKX |

The zero-shot AUC on Bybit + OKX is the primary generalisation metric. These
exchanges are never shown to the model during training.

**Saved outputs:**

| File | Description |
|------|-------------|
| `models/pump_detector_v3.pth` | Best weights by validation AUC (saved during training) |

**Configuration:**

```bash
python scripts2/train_pump_detector.py \
    --epochs 60 \
    --batch-size 64 \
    --no-zero-shot     # use this flag if Bybit/OKX data is not available
```

**Interpreting the training output:**

```
================================================================
  PumpDetectorV3  (dual-stream: coin + market context)
  Train: ['binance', 'kucoin', 'huobi', 'mexc', 'gateio', 'bitget']
  Test : ['bybit', 'okx']  (zero-shot)
================================================================

 Epoch    Train Loss    Val AUC
     1        0.6832     0.5211
     5        0.4901     0.7043
    10        0.3812     0.7834
    ...
  Best validation AUC : 0.8549
  Model saved         : models/pump_detector_v3.pth

================================================================
  Zero-Shot Test  (Bybit + OKX — never seen during training)
================================================================
  Zero-shot AUC : 0.7273

              precision    recall  f1-score
     control       0.91      0.81      0.86
        pump       0.62      0.80      0.70

  RESULT: Moderate cross-exchange generalisation.
```

**Reading the AUC numbers:**

| Val AUC | Meaning |
|---------|---------|
| > 0.85 | Strong fit on training exchanges |
| 0.75 – 0.85 | Good — model is learning the pattern |
| < 0.70 | Weak — check data quality or add more samples |

| Zero-shot AUC | Meaning |
|---------------|---------|
| > 0.75 | Strong cross-exchange generalisation |
| 0.65 – 0.75 | Moderate — acceptable for most use cases |
| < 0.65 | Weak transfer — consider adding Bybit/OKX to training |

**Threshold guidance for live use:**

| Threshold | Effect |
|-----------|--------|
| 0.30 | High recall — catches most pumps, more false alarms |
| 0.50 | Balanced (default) |
| 0.65 – 0.75 | Recommended for live alerting |
| 0.85+ | Very conservative — flags only the most obvious events |

---

## Peak Estimation Report (`run_peak_estimation.py`)

Scans all three tiers (`real/`, `reconstructed/`, `synthetic/`) and writes a flat
CSV with one row per L2 file, combining peak metrics with exchange, regime, and
bucket labels.

**Output file:** `output/peak_estimation_results.csv`

### Columns

| Column | Description |
|--------|-------------|
| `tier` | `real`, `reconstructed`, or `synthetic` |
| `exchange` | Exchange name (binance, bybit, kucoin, …) |
| `coin_regime` | `pump` or `control` |
| `market_regime` | `normal`, `uncertain`, `pumped`, or `unknown` |
| `increase_pct` | Price rise from bar 0 to detected peak (%) |
| `retracement_pct` | Fraction of the gain given back after the peak (%) |
| `peak_idx` | Bar index of the detected peak |
| `peak_bucket` | micro / small / medium / large / major / extreme |
| `peak_truncated` | `True` if fewer than 5 bars existed after the peak |
| `confirmed_pump` | `True` if rise ≥ 5% AND retracement ≥ 30% |
| `file` | Relative path to the L2 file from the project root |

### Console summary printed at the end of the run

```
======================================================================
  PEAK ESTIMATION SUMMARY
======================================================================

  Confirmed pumps (≥5% rise, ≥30% retracement): 2,007
  Truncated events (peak near window edge):        142
  Control events:                                2,923

  By exchange and tier (confirmed pumps):
  Exchange     Tier              Pumps   Mean rise%  Mean retracement%
  ------------ --------------- ------- ------------ ------------------
  binance      reconstructed       313         14.2               61.3
  gateio       reconstructed     1,168         11.8               58.7
  kucoin       reconstructed        24         18.4               64.1
  ...

  Peak bucket distribution (confirmed pumps):
    micro          412
    small          891
    medium         503
    large          148
    major           38
    extreme         15
```

---

## Output Directory Layout

After a full pipeline run the project root will contain:

```
output/
  peak_estimation_results.csv      ← per-file peak metrics (one row per L2 file)

models/
  direct_l2_vae_v1.pth             ← DirectL2VAE weights
  l2_scaler.pkl                    ← StandardScaler for VAE input normalisation
  pump_detector_v3.pth             ← PumpDetectorV3 weights (best validation AUC)

data/
  l2_training_samples.npy          ← VAE training array [N, 96, 4]

real/                              ← downloaded from Drive (Real/)
  [exchange]/pumps/[symbol]/
    [symbol]_[date]_L2.csv
    [symbol]_[date]_meta.json      ← written by tag_peak_buckets.py
    [symbol]_[date]_trades.csv     ← taker flow (if available)
    [symbol]_[date]_market_ctx.csv ← BTC context (if available)
  [exchange]/normal/[symbol]/
  [exchange]/uncertain/[symbol]/

reconstructed/                     ← downloaded from Drive (Reconstructed/)
  (same structure as real/)

synthetic/                         ← downloaded from Drive (Synthetic/)
  (same structure as real/)
```

---

## `main.py` Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--root` | auto (parent of `scripts2/`) | Project root directory |
| `--skip-download` | off | Skip Drive download — data already on disk |
| `--skip-peak-tag` | off | Skip peak tagging — `_meta.json` already present |
| `--skip-vae` | off | Skip DirectL2VAE training |
| `--train-only` | off | Skip download and peak tag; go straight to model training |
| `--epochs` | 60 | PumpDetectorV3 training epochs |
| `--vae-epochs` | 100 | DirectL2VAE training epochs |
| `--batch-size` | 64 | Batch size for PumpDetectorV3 |
| `--no-zero-shot` | off | Skip Bybit/OKX zero-shot evaluation |
| `--overwrite` | off | Re-compute peaks even if `_meta.json` already exists |
| `--dry-run` | off | Preview Drive download without downloading anything |
| `--continue-on-error` | off | Continue to the next step even if a step fails |

**Common run patterns:**

```bash
# Full pipeline (download + peak tag + train)
python main.py

# Data already on disk
python main.py --skip-download

# Data on disk and peaks already tagged
python main.py --train-only

# Fast experiment: fewer epochs, skip zero-shot eval
python main.py --train-only --epochs 20 --vae-epochs 30 --no-zero-shot

# Re-compute peaks even if meta.json exists, then retrain
python main.py --skip-download --overwrite

# Don't stop if one step fails
python main.py --continue-on-error
```

---

## Dependencies

**Python packages** (`pip install -r scripts2/requirements.txt`):

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
torch>=2.0.0
google-api-python-client>=2.0.0
requests>=2.28.0
```

Everything installs via pip — no system binaries required.

**Project files** required at the project root:

- `models/direct_l2_vae.py` — DirectL2VAE architecture
- `models/pump_detector.py` — PumpDetectorV3 architecture
- `models/__init__.py` — makes `models/` a Python package

The training scripts add the project root to `sys.path` automatically, so you do
not need to set `PYTHONPATH` manually.
