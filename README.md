# MPS – Multi-scale Price-movement Similarity (qlib edition)

Adaptation of [ECNU-CILAB/MPS](https://github.com/ECNU-CILAB/MPS) for local
qlib data on macOS, with five standard financial evaluation metrics.

---

## Architecture

```
Stage 1 – Co-movement pre-training (MultiTask)
──────────────────────────────────────────────
  Stock A window (seq_len × n_features)      Stock B window
         │                                          │
  ┌──────┴──────────────────────────────────────────┘
  │        Shared Encoder (3 × TransformerEncoder)
  │    short-scale      middle-scale      long-scale
  │    (k=1 days)       (k=5 days)       (k=20 days)
  └────────────────────────────────────────────────────┐
       Cross-attention + 3-class head per scale
       Predicts co-movement direction of (A, B)

Stage 2 – Return-ranking fine-tuning (GRU_Predict)
──────────────────────────────────────────────────
  Stock window → frozen Encoder → s/m/l encodings
       └──── attention bottleneck ── BiGRU ── Linear ── Sigmoid ──► score
  Ranked cross-sectionally; all 5 metrics reported on test set.
```

---

## File overview

| File | Purpose |
|---|---|
| `config.py` | All paths, dates, hyperparameters, metric settings |
| `model.py` | `MultiTask` (stage-1) and `GRU_Predict` (stage-2) |
| `dataset.py` | qlib data loading, window builder, `PairDataset`, `StockDataset` |
| `train.py` | Two-stage training loop (no validation, fixed epochs) |
| `evaluate.py` | MSE / Accuracy / IC / ICIR / Sharpe Ratio |
| `requirements.txt` | Python dependencies |

---

## Setup (macOS)

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3/M4):**
> PyTorch uses the MPS backend automatically.
> Device selection in `config.py`: CUDA → MPS → CPU.

### 2. Qlib data

Place your data at `~/Desktop/my_qlib_data/` (or edit `QLIB_DATA_PATH` in
`config.py`).  Expected layout:

```
~/Desktop/my_qlib_data/
├── calendars/
│   └── day.txt              # one trading date per line (YYYY-MM-DD)
├── features/
│   └── sh600000/
│       ├── close.day.bin
│       ├── open.day.bin
│       ├── high.day.bin
│       ├── low.day.bin
│       ├── volume.day.bin
│       ├── vwap.day.bin
│       └── ...
└── instruments/
    ├── csi300.txt           # format: <stock>\t<start>\t<end>
    └── ...
```

Date coverage: **2020-01-02 → 2026-03-20**.

---

## Data split (no validation set)

| Split | Date range | Share |
|-------|------------|-------|
| **Train** | 2020-01-02 → 2024-12-31 | ~80 % |
| **Test**  | 2025-01-01 → 2026-03-20 | ~20 % |

There is no separate validation set, following the experimental setup in the
MPS paper.

---

## Training

```bash
python train.py
```

**Stage 1** (25 epochs, fixed):
- Trains the shared encoder on stock-pair co-movement classification.
- Reports cross-entropy loss and classification accuracy each epoch.
- Saves checkpoint → `saved_models/encoder.pt`.

**Stage 2** (25 epochs, fixed):
- Freezes the encoder; trains `GRU_Predict` on cross-sectional return ranking.
- Reports MSE loss each epoch.
- Saves checkpoint → `saved_models/predictor.pt`.

**Final test evaluation** – all five metrics printed at the end.

---

## Evaluation

```bash
python evaluate.py     # re-evaluate saved models on the test set
```

Example output:

```
────────────────────────────────────────────────────
  Test Metrics
────────────────────────────────────────────────────
  MSE      (↓ better) : +0.082134
  Accuracy (↑ better) : +0.5841
  IC       (↑ better) : +0.0423
  ICIR     (↑ better) : +0.5817
  Sharpe   (↑ better) : +1.2341
────────────────────────────────────────────────────
```

---

## Metric definitions

| Metric | Definition |
|--------|------------|
| **MSE** | Mean Squared Error between predicted cross-sectional rank and actual rank |
| **Accuracy** | Fraction of stocks correctly placed in top-half vs bottom-half by predicted score |
| **IC** | Daily mean Pearson correlation between predicted scores and actual next-day returns |
| **ICIR** | `IC.mean() / IC.std()` – risk-adjusted consistency of the signal |
| **Sharpe** | Annualised Sharpe Ratio of the long-top-10% portfolio (daily rebalancing) |

Practical benchmarks: IC > 0.05 is considered good; ICIR > 0.5 is useful;
Sharpe > 1.0 is a common target in quantitative strategies.

---

## Hyperparameters (paper Section 3)

| Parameter | Value | Source |
|-----------|-------|--------|
| Look-back window `q` | 20 | paper |
| Co-movement horizons `k` | 1 / 5 / 20 days | paper |
| Correlation boundaries (short) | r₁ = 2/3, r₂ = −1/3 | paper |
| Correlation boundaries (mid/long) | r₁ = 0.5, r₂ = −0.5 | paper |
| Attention heads `h` | 1 | paper |
| Loss weights λ, γ, μ | 1/3 each | paper |
| Learning rate | 1e-4 | notebook |
| Batch size | 256 | notebook |
| GRU hidden size | 30 | notebook |
| GRU layers | 2 | notebook |
| Dropout | 0.3 | notebook |
| Epochs (each stage) | 25 | notebook |

---

## Features

Six normalised daily features computed via qlib expressions:

| Feature | Expression |
|---------|------------|
| `open_ret`  | `open / prev_close − 1` |
| `high_ret`  | `high / prev_close − 1` |
| `low_ret`   | `low / prev_close − 1`  |
| `close_ret` | `close / prev_close − 1`|
| `vol_chg`   | `log(volume / prev_volume + ε)` |
| `vwap_ret`  | `vwap / prev_close − 1` |

Per-stock z-score normalisation, clipped at ±3σ.

---

## Reference

> **MPS: Multi-scale Time Based Stock Appreciation Ranking Prediction via
> Price Co-movement Discrimination**
> ECNU-CILAB – https://github.com/ECNU-CILAB/MPS
