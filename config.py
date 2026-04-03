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

ENCODER_SAVE_PATH = os.path.join(MODEL_DIR, "encoder.pt")
PREDICTOR_SAVE_PATH = os.path.join(MODEL_DIR, "predictor.pt")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
UNIVERSE = "csi300"          # instrument pool (csi300 / csi500 / all)
START_DATE = "2020-01-02"
END_DATE   = "2026-03-20"

# 80 / 20 chronological split
TRAIN_START = "2020-01-02"
TRAIN_END   = "2024-06-30"   # ~80 % of all trading days
VALID_START = "2024-07-01"   # held-out validation (part of the 80 % block)
VALID_END   = "2024-12-31"
TEST_START  = "2025-01-01"   # ~20 % test
TEST_END    = "2026-03-20"

# Feature set (6 daily features, all normalised relative to previous close)
FEATURE_FIELDS = [
    "$open/Ref($close,1)-1",
    "$high/Ref($close,1)-1",
    "$low/Ref($close,1)-1",
    "$close/Ref($close,1)-1",
    "Log($volume/Ref($volume,1)+1e-8)",
    "$vwap/Ref($close,1)-1",
]
FEATURE_NAMES = ["open_ret", "high_ret", "low_ret", "close_ret", "vol_chg", "vwap_ret"]

# Label: next-day close return used to rank stocks
LABEL_FIELD = "Ref($close,-1)/$close-1"   # 1-day forward return
LABEL_NAME  = "label"

# ---------------------------------------------------------------------------
# Window sizes
# ---------------------------------------------------------------------------
SEQ_LEN    = 20    # look-back window (trading days)
N_FEATURES = 6     # number of features per timestep

# Correlation horizons for co-movement labels (stage-1 pre-training)
CORR_SHORT  = 5
CORR_MID    = 10
CORR_LONG   = 20   # same as SEQ_LEN

# Number of stock pairs sampled per date during stage-1 training
PAIRS_PER_DATE = 200

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
HID_DIM  = 6     # == N_FEATURES (no projection; transformer works in feature space)
N_LAYERS = 1
N_HEADS  = 1
PF_DIM   = 30
DROPOUT  = 0.3

GRU_HIDDEN  = 32
GRU_LAYERS  = 1

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE   = 256
LR           = 1e-4
N_EPOCHS_ENC = 25    # stage-1 (encoder / co-movement)
N_EPOCHS_PRE = 25    # stage-2 (predictor / return ranking)
PATIENCE     = 5     # early-stopping patience (epochs)

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
