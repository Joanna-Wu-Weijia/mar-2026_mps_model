"""
MPS (Multi-scale Price-movement Similarity) configuration.
Adapted from https://github.com/ECNU-CILAB/MPS for qlib + macOS.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
QLIB_DATA_PATH = os.path.expanduser("~/Desktop/my_qlib_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

ENCODER_SAVE_PATH   = os.path.join(MODEL_DIR, "encoder.pt")
PREDICTOR_SAVE_PATH = os.path.join(MODEL_DIR, "predictor.pt")

# ---------------------------------------------------------------------------
# Data  –  strict 80 / 20 chronological split, NO validation set
# ---------------------------------------------------------------------------
UNIVERSE    = "csi300"       # instrument pool  (csi300 / csi500 / all)

TRAIN_START = "2020-01-02"
TRAIN_END   = "2024-12-31"   # ~80 % of all trading days

TEST_START  = "2025-01-01"   # ~20 % held-out test
TEST_END    = "2026-03-20"

# Feature set – raw OHLCV, matching build_data.ipynb exactly.
# Original: ["open", "close", "high", "low", "turnover", "volume"]
# qlib fields: $amount corresponds to "turnover" (traded value in yuan).
# Global 3-sigma clip + z-score normalisation (train stats applied to test).
FEATURE_FIELDS = ["$open", "$close", "$high", "$low", "$amount", "$volume"]
FEATURE_NAMES  = ["open", "close", "high", "low", "turnover", "volume"]

# Label: next-day close return used to rank stocks (stage-2 target)
LABEL_FIELD = "Ref($close,-1)/$close-1"   # 1-day forward return
LABEL_NAME  = "label"

# Future close ratios for co-movement labels (stage-1 pretext task).
# Ref($close,-i)/$close = future close / today's close, for i = 1..20
FUTURE_CLOSE_FIELDS = [f"Ref($close,{-i})/$close" for i in range(1, 21)]
FUTURE_CLOSE_NAMES  = [f"fwd_close_{i}" for i in range(1, 21)]

# ---------------------------------------------------------------------------
# Window / co-movement parameters  (from paper Section 3)
# ---------------------------------------------------------------------------
SEQ_LEN    = 20    # look-back window q
N_FEATURES = 6

# k_short=1, k_mid=5, k_long=20  (paper Section 3)
# Correlation boundaries: r1=2/3, r2=-1/3 (short); r1=0.5, r2=-0.5 (mid/long)
# Classes: 0 = uncorrelated, 1 = positively correlated, 2 = negatively correlated
CORR_SHORT  = 1
CORR_MID    = 5
CORR_LONG   = 20

# Number of stock pairs sampled per date during stage-1 training
PAIRS_PER_DATE = 200

# ---------------------------------------------------------------------------
# Model architecture  (from paper / original notebook)
# ---------------------------------------------------------------------------
HID_DIM  = 6     # transformer hidden dim == N_FEATURES (no input projection)
N_LAYERS = 1
N_HEADS  = 1     # h = 1  (paper Section 3)
PF_DIM   = 30
DROPOUT  = 0.3

GRU_HIDDEN = 30   # original notebook: hidden_size=30
GRU_LAYERS = 2    # original notebook: num_layers=2

# ---------------------------------------------------------------------------
# Training  –  fixed epochs, no early stopping (no validation set)
# ---------------------------------------------------------------------------
BATCH_SIZE   = 256
LR           = 1e-4
N_EPOCHS_ENC = 25    # stage-1 encoder training epochs
N_EPOCHS_PRE = 25    # stage-2 predictor training epochs

# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
# Sharpe Ratio: annualised Sharpe of the long-top-K portfolio on test days
TOP_K_PCT        = 0.10    # select top 10 % of stocks by predicted score
ANNUAL_FACTOR    = 252     # trading days per year for annualisation
RISK_FREE_RATE   = 0.0     # daily risk-free rate (set to 0 for simplicity)

# ---------------------------------------------------------------------------
# Device  (CUDA > Apple-Silicon MPS > CPU)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
SEED = 42
