# MPS – Multi-scale Price-movement Similarity

[ECNU-CILAB/MPS](https://github.com/ECNU-CILAB/MPS) 

---

## Architecture

```
Stage 1 – Co-movement pre-training (MultiTask)
─────────────────────────────────────────────
  Stock A window (seq_len × n_features)          Stock B window
         │                                               │
  ┌──────┴──────────────────────────────────────────────┘
  │           Shared Encoder (3 × TransformerEncoder)
  │   short-scale   middle-scale   long-scale
  └───────────────────────────────────────────────────────┐
          Cross-attention + classification head (3 classes)
          Predicts correlation class of A & B at each scale

Stage 2 – Return-ranking fine-tuning (GRU_Predict)
───────────────────────────────────────────────────
  Single stock window  →  frozen Encoder  →  s/m/l encodings
          └──────────────── GRU ────── Linear ── Sigmoid ──► score [0,1]
  Ranked cross-sectionally; evaluated with IC and IR.
```

---

## File overview

| File | Purpose |
|---|---|
| `config.py` | All paths, dates, hyper-parameters |
| `model.py` | `MultiTask` (stage-1) and `GRU_Predict` (stage-2) |
| `dataset.py` | qlib data loading, window builder, `PairDataset`, `StockDataset` |
| `train.py` | Two-stage training loop with early stopping |
| `evaluate.py` | IC / IR computation; standalone eval script |
| `requirements.txt` | Python dependencies |

---

## Setup (macOS)

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** PyTorch automatically uses the MPS backend
> (`torch.backends.mps`).  No extra steps are needed.  The device selection
> in `config.py` falls through: CUDA → MPS → CPU.

### 2. Qlib data

Place your local qlib data at `~/Desktop/my_qlib_data/` (or edit
`QLIB_DATA_PATH` in `config.py`).  Expected layout:

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
    ├── all.txt
    ├── csi300.txt           # format: <stock>\t<start>\t<end>
    ├── csi500.txt
    └── ...
```

Date coverage used: **2020-01-02 → 2026-03-20**.

---

## Data split

| Split | Date range | Approx. share |
|-------|------------|---------------|
| Train | 2020-01-02 → 2024-06-30 | ~72 % |
| Valid | 2024-07-01 → 2024-12-31 | ~8 % |
| Test  | 2025-01-01 → 2026-03-20 | ~20 % |

---

## Training

```bash
python train.py
```

The script runs both stages end-to-end:

1. **Stage 1** – trains the shared encoder on co-movement pairs.
   Best checkpoint saved to `saved_models/encoder.pt`.
2. **Stage 2** – freezes the encoder and trains `GRU_Predict`.
   Best checkpoint saved to `saved_models/predictor.pt`.
   Validation IC / IR are reported each epoch.

---

## Evaluation

Run standalone evaluation on the test set:

```bash
python evaluate.py
```

Output example:

```
──────────────────────────────────────────────────
  Test Results
──────────────────────────────────────────────────
  Pearson  IC : +0.0423   IR : +0.5817
  Spearman IC : +0.0398   IR : +0.5231
──────────────────────────────────────────────────
```

### Metric definitions

**IC (Information Coefficient)**
Pearson (or Spearman) correlation between the model's predicted score and the
actual next-day return, computed cross-sectionally across all stocks on each
trading day.  Higher is better; a value of ~0.05 is considered good in practice.

**IR (Information Ratio)**
`IC.mean() / IC.std()` over all trading days.  Measures consistency of the
signal.  IR > 0.5 is generally considered useful.

---

## Configuration

Key settings in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `UNIVERSE` | `"csi300"` | Instrument pool |
| `SEQ_LEN` | `20` | Look-back window (trading days) |
| `N_FEATURES` | `6` | Features per timestep |
| `HID_DIM` | `6` | Transformer hidden size |
| `N_HEADS` | `1` | Attention heads |
| `N_LAYERS` | `1` | Transformer layers |
| `GRU_HIDDEN` | `32` | GRU hidden size |
| `BATCH_SIZE` | `256` | Mini-batch size |
| `LR` | `1e-4` | Learning rate |
| `N_EPOCHS_ENC` | `25` | Stage-1 max epochs |
| `N_EPOCHS_PRE` | `25` | Stage-2 max epochs |
| `PATIENCE` | `5` | Early-stopping patience |

---

## Features

Six normalised daily features are computed from raw OHLCV data via qlib:

| Feature | Expression |
|---------|------------|
| `open_ret` | `open / prev_close - 1` |
| `high_ret` | `high / prev_close - 1` |
| `low_ret` | `low / prev_close - 1` |
| `close_ret` | `close / prev_close - 1` |
| `vol_chg` | `log(volume / prev_volume + ε)` |
| `vwap_ret` | `vwap / prev_close - 1` |

Each feature is z-score normalised per stock (across its full available
history) and clipped at ±3σ.

---

## Reference

> **MPS: Multi-scale Time Based Stock Appreciation Ranking Prediction via
> Price Co-movement Discrimination**
> ECNU-CILAB – https://github.com/ECNU-CILAB/MPS
