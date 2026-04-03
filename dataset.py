"""
Data loading and preprocessing for the MPS model using local qlib data.

Data directory layout (set QLIB_DATA_PATH in config.py):
    my_qlib_data/
    ├── calendars/day.txt
    ├── features/<stock>/close.day.bin ...
    └── instruments/csi300.txt ...

Two Dataset classes are produced:
  PairDataset   – used in stage-1 (MultiTask co-movement training)
  StockDataset  – used in stage-2 (GRU_Predict return-ranking training/eval)
"""

from __future__ import annotations

import random
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import qlib
from qlib.data import D

import config


# ---------------------------------------------------------------------------
# Qlib initialisation
# ---------------------------------------------------------------------------

def init_qlib() -> None:
    """Initialise qlib with the local data provider."""
    qlib.init(provider_uri=config.QLIB_DATA_PATH, region="cn")


# ---------------------------------------------------------------------------
# Raw feature loading
# ---------------------------------------------------------------------------

def load_raw_features(start: str, end: str) -> pd.DataFrame:
    """
    Load 6 normalised daily features for all instruments in the universe.

    Returns
    -------
    df : pd.DataFrame
        MultiIndex (instrument, datetime), columns = FEATURE_NAMES + ['label']
        label = next-day close return (used for ranking in stage-2).
    """
    instruments = D.instruments(config.UNIVERSE)

    fields = config.FEATURE_FIELDS + [config.LABEL_FIELD]
    names  = config.FEATURE_NAMES  + [config.LABEL_NAME]

    # Add enough history so we can build SEQ_LEN-day windows from start
    from qlib.utils.time import Freq  # noqa: F401 – side-effects only
    df = D.features(
        instruments,
        fields,
        start_time=start,
        end_time=end,
        freq="day",
    )
    df.columns = names
    df.index.names = ["instrument", "datetime"]
    return df


# ---------------------------------------------------------------------------
# Normalisation  (z-score per feature, clip at 3σ)
# ---------------------------------------------------------------------------

def _zscore_clip(x: np.ndarray) -> np.ndarray:
    """Row-wise (feature-wise) z-score normalisation with 3-sigma clip."""
    mu  = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x,  axis=0, keepdims=True) + 1e-8
    x   = (x - mu) / std
    return np.clip(x, -3.0, 3.0)


# ---------------------------------------------------------------------------
# Window builder
# ---------------------------------------------------------------------------

def _build_windows(
    df: pd.DataFrame,
    stocks: List[str],
    seq_len: int,
) -> dict[str, dict[pd.Timestamp, np.ndarray]]:
    """
    For each stock build a mapping  date → feature_window.

    feature_window : np.ndarray of shape (seq_len, n_features)
    Only dates that have a complete (no-NaN) window of length seq_len are kept.
    """
    feat_cols = config.FEATURE_NAMES
    windows: dict[str, dict[pd.Timestamp, np.ndarray]] = {}

    for stock in stocks:
        if stock not in df.index.get_level_values("instrument"):
            continue

        sub = (
            df.loc[stock, feat_cols]
            .sort_index()
            .dropna(how="any")
        )

        stock_wins: dict[pd.Timestamp, np.ndarray] = {}
        dates = sub.index.tolist()
        vals  = sub.values.astype(np.float32)

        # Normalise per-stock across its whole available history
        vals = _zscore_clip(vals).astype(np.float32)

        for i in range(seq_len - 1, len(dates)):
            win = vals[i - seq_len + 1 : i + 1]   # (seq_len, n_features)
            if win.shape[0] == seq_len:
                stock_wins[dates[i]] = win

        if stock_wins:
            windows[stock] = stock_wins

    return windows


# ---------------------------------------------------------------------------
# Co-movement label helpers (stage-1)
# ---------------------------------------------------------------------------

def _corr_label(ret_a: np.ndarray, ret_b: np.ndarray) -> int:
    """
    Pearson correlation between two return series, binned into 3 classes:
      0  = negative   (corr < -1/3)
      1  = neutral    (-1/3 ≤ corr ≤ 1/3)
      2  = positive   (corr > 1/3)
    Returns 1 when one or both series have zero variance.
    """
    if ret_a.std() < 1e-8 or ret_b.std() < 1e-8:
        return 1
    corr = float(np.corrcoef(ret_a, ret_b)[0, 1])
    if corr < -1 / 3:
        return 0
    elif corr > 1 / 3:
        return 2
    return 1


def _corr_labels_for_pair(
    win_a: np.ndarray, win_b: np.ndarray
) -> Tuple[int, int, int]:
    """
    Compute co-movement labels at three temporal scales using the close-return
    column (index 3) inside the feature window.

    Returns (corr_short_label, corr_mid_label, corr_long_label).
    """
    close_idx = config.FEATURE_NAMES.index("close_ret")
    ret_a = win_a[:, close_idx]
    ret_b = win_b[:, close_idx]

    c_short = _corr_label(ret_a[-config.CORR_SHORT:],  ret_b[-config.CORR_SHORT:])
    c_mid   = _corr_label(ret_a[-config.CORR_MID:],    ret_b[-config.CORR_MID:])
    c_long  = _corr_label(ret_a,                        ret_b)
    return c_short, c_mid, c_long


# ---------------------------------------------------------------------------
# PairDataset – stage-1
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """
    Each sample: (window_A, window_B, label_short, label_mid, label_long)

    windows_A/B : float32 tensors of shape (seq_len, n_features)
    labels      : int64 tensors  (3 classes each)
    """

    def __init__(
        self,
        windows: dict[str, dict[pd.Timestamp, np.ndarray]],
        dates: List[pd.Timestamp],
        pairs_per_date: int = config.PAIRS_PER_DATE,
        seed: int = config.SEED,
    ):
        rng    = random.Random(seed)
        stocks = list(windows.keys())

        self.X_A: List[np.ndarray] = []
        self.X_B: List[np.ndarray] = []
        self.y1:  List[int] = []
        self.y2:  List[int] = []
        self.y3:  List[int] = []

        for date in dates:
            # stocks that have a window on this date
            avail = [s for s in stocks if date in windows[s]]
            if len(avail) < 2:
                continue
            n = min(pairs_per_date, len(avail) * (len(avail) - 1) // 2)
            for _ in range(n):
                a, b = rng.sample(avail, 2)
                win_a = windows[a][date]
                win_b = windows[b][date]
                c1, c2, c3 = _corr_labels_for_pair(win_a, win_b)
                self.X_A.append(win_a)
                self.X_B.append(win_b)
                self.y1.append(c1)
                self.y2.append(c2)
                self.y3.append(c3)

    def __len__(self) -> int:
        return len(self.y1)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X_A[idx]),
            torch.from_numpy(self.X_B[idx]),
            torch.tensor(self.y1[idx], dtype=torch.long),
            torch.tensor(self.y2[idx], dtype=torch.long),
            torch.tensor(self.y3[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# StockDataset – stage-2
# ---------------------------------------------------------------------------

class StockDataset(Dataset):
    """
    Each sample: (window, label_rank)

    window     : float32 tensor  (seq_len, n_features)
    label_rank : float32 scalar  – cross-sectional rank (0..1) of forward return
    """

    def __init__(
        self,
        windows:  dict[str, dict[pd.Timestamp, np.ndarray]],
        labels_df: pd.DataFrame,   # MultiIndex (instrument, datetime), col 'label'
        dates:    List[pd.Timestamp],
    ):
        self.windows:  List[np.ndarray] = []
        self.labels:   List[float]      = []

        stocks = list(windows.keys())

        for date in dates:
            # forward-return for all available stocks on this date
            day_labels: dict[str, float] = {}
            for stock in stocks:
                try:
                    val = labels_df.loc[(stock, date), config.LABEL_NAME]
                    if not np.isnan(val):
                        day_labels[stock] = float(val)
                except KeyError:
                    pass

            if len(day_labels) < 2:
                continue

            # Convert to cross-sectional rank (0 = worst, 1 = best)
            sorted_stocks = sorted(day_labels, key=day_labels.get)  # type: ignore
            n = len(sorted_stocks)
            rank_map = {s: i / (n - 1) for i, s in enumerate(sorted_stocks)}

            for stock, rank in rank_map.items():
                if date in windows.get(stock, {}):
                    self.windows.append(windows[stock][date])
                    self.labels.append(rank)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.windows[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# High-level builder  (called from train.py)
# ---------------------------------------------------------------------------

def build_datasets(
    seq_len: int = config.SEQ_LEN,
) -> Tuple[
    PairDataset, PairDataset,           # enc_train, enc_valid
    StockDataset, StockDataset, StockDataset,  # pred_train, pred_valid, pred_test
    pd.DataFrame,                       # full labels df (for evaluate.py)
]:
    """
    Load qlib data and construct all datasets needed for both training stages.
    """
    init_qlib()

    print("Loading features from qlib …")
    # Load enough history before TRAIN_START to fill the initial window
    from datetime import datetime, timedelta
    extra_days = seq_len * 2  # conservative buffer for weekends/holidays
    history_start = (
        datetime.strptime(config.TRAIN_START, "%Y-%m-%d")
        - timedelta(days=extra_days)
    ).strftime("%Y-%m-%d")

    df = load_raw_features(history_start, config.TEST_END)

    # --- dates per split ---
    cal = df.index.get_level_values("datetime").unique().sort_values()

    def _dates(start: str, end: str) -> List[pd.Timestamp]:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        return [d for d in cal if s <= d <= e]

    train_dates = _dates(config.TRAIN_START, config.TRAIN_END)
    valid_dates = _dates(config.VALID_START, config.VALID_END)
    test_dates  = _dates(config.TEST_START,  config.TEST_END)

    stocks = df.index.get_level_values("instrument").unique().tolist()

    print(f"  Stocks in universe : {len(stocks)}")
    print(f"  Train dates        : {len(train_dates)} ({train_dates[0].date()} → {train_dates[-1].date()})")
    print(f"  Valid dates        : {len(valid_dates)} ({valid_dates[0].date()} → {valid_dates[-1].date()})")
    print(f"  Test  dates        : {len(test_dates)} ({test_dates[0].date()} → {test_dates[-1].date()})")

    # --- build feature windows (covers the full date range) ---
    print("Building feature windows …")
    all_dates = _dates(history_start, config.TEST_END)
    windows   = _build_windows(df, stocks, seq_len)

    # --- stage-1 datasets ---
    print("Building PairDatasets (stage-1) …")
    enc_train = PairDataset(windows, train_dates)
    enc_valid = PairDataset(windows, valid_dates)

    print(f"  enc_train pairs : {len(enc_train):,}")
    print(f"  enc_valid pairs : {len(enc_valid):,}")

    # --- stage-2 datasets ---
    print("Building StockDatasets (stage-2) …")
    pred_train = StockDataset(windows, df, train_dates)
    pred_valid = StockDataset(windows, df, valid_dates)
    pred_test  = StockDataset(windows, df, test_dates)

    print(f"  pred_train samples : {len(pred_train):,}")
    print(f"  pred_valid samples : {len(pred_valid):,}")
    print(f"  pred_test  samples : {len(pred_test):,}")

    return enc_train, enc_valid, pred_train, pred_valid, pred_test, df
