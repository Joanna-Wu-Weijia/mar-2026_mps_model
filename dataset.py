"""
Data loading and preprocessing for the MPS model using local qlib data.

Data directory layout (set QLIB_DATA_PATH in config.py):
    my_qlib_data/
    ├── calendars/day.txt
    ├── features/<stock>/close.day.bin ...
    └── instruments/csi300.txt ...

Two Dataset classes:
  PairDataset   – stage-1 (MultiTask co-movement pre-training)
  StockDataset  – stage-2 (GRU_Predict return-ranking training / eval)

Co-movement labels follow the original MPS build_data.ipynb:
  correlation is computed on FUTURE close price sequences, giving a
  supervision signal that teaches the encoder about price co-movement.

  Label encoding (0 / 1 / 2):
    corr_short  (next 1 trading day)  : corr >= 2/3 → 1, corr <= -1/3 → 2, else 0
    corr_mid    (next 5 trading days) : corr >= 0.5 → 1, corr <= -0.5 → 2, else 0
    corr_long   (next 20 trading days): corr >= 0.5 → 1, corr <= -0.5 → 2, else 0
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

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
    qlib.init(provider_uri=config.QLIB_DATA_PATH, region="cn")


# ---------------------------------------------------------------------------
# Raw feature + label loading
# ---------------------------------------------------------------------------

def load_raw_data(start: str, end: str) -> pd.DataFrame:
    """
    Load 6 historical features  +  1-day forward return label
    +  20 future close ratios (for co-movement labels in stage-1).

    Returns a DataFrame with MultiIndex (instrument, datetime).
    """
    instruments = D.instruments(config.UNIVERSE)

    fields = (config.FEATURE_FIELDS
              + [config.LABEL_FIELD]
              + config.FUTURE_CLOSE_FIELDS)
    names  = (config.FEATURE_NAMES
              + [config.LABEL_NAME]
              + config.FUTURE_CLOSE_NAMES)

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
# Per-stock z-score normalisation with 3-sigma clip
# ---------------------------------------------------------------------------

def _zscore_clip(x: np.ndarray) -> np.ndarray:
    mu  = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x,  axis=0, keepdims=True) + 1e-8
    z   = (x - mu) / std
    return np.clip(z, -3.0, 3.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Window builder
# ---------------------------------------------------------------------------

def _build_windows(
    df: pd.DataFrame,
    stocks: List[str],
    seq_len: int,
) -> dict[str, dict[pd.Timestamp, np.ndarray]]:
    """
    For each stock build:  date → feature_window (seq_len, n_features).
    Only dates with a complete NaN-free window are kept.
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
        if len(sub) < seq_len:
            continue

        vals  = _zscore_clip(sub.values)
        dates = sub.index.tolist()

        stock_wins: dict[pd.Timestamp, np.ndarray] = {}
        for i in range(seq_len - 1, len(dates)):
            win = vals[i - seq_len + 1 : i + 1]
            if win.shape[0] == seq_len:
                stock_wins[dates[i]] = win

        if stock_wins:
            windows[stock] = stock_wins

    return windows


# ---------------------------------------------------------------------------
# Co-movement correlation helpers (original MPS convention)
# ---------------------------------------------------------------------------

def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; returns 0.0 when one series is constant."""
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    denom = a.std() * b.std() * len(a)
    if denom < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _corr_short_label(corr: float) -> int:
    """1-day horizon: corr >= 2/3 → 1 (positive), <= -1/3 → 2 (negative), else 0."""
    if corr >= 2 / 3:
        return 1
    if corr <= -1 / 3:
        return 2
    return 0


def _corr_mid_long_label(corr: float) -> int:
    """5-/20-day horizon: corr >= 0.5 → 1, <= -0.5 → 2, else 0."""
    if corr >= 0.5:
        return 1
    if corr <= -0.5:
        return 2
    return 0


def _future_corr_labels(
    df: pd.DataFrame,
    stock_a: str,
    stock_b: str,
    date: pd.Timestamp,
) -> Tuple[int, int, int] | None:
    """
    Compute co-movement labels using FUTURE close price ratios.

    Future close ratio for day i:  Ref($close,-i)/$close
    Arrays span [close_today, close_t1, ..., close_t20] for each stock.

    Returns (label_short, label_mid, label_long) or None if data missing.
    """
    fwd_cols = config.FUTURE_CLOSE_NAMES   # fwd_close_1 … fwd_close_20

    try:
        row_a = df.loc[(stock_a, date), fwd_cols].values.astype(float)
        row_b = df.loc[(stock_b, date), fwd_cols].values.astype(float)
    except KeyError:
        return None

    if np.any(np.isnan(row_a)) or np.any(np.isnan(row_b)):
        return None

    # short: next 1 day  → 2 points: [fwd_close_1] for each stock
    arr_short_a = np.array([1.0, row_a[0]])   # today=1.0, t+1
    arr_short_b = np.array([1.0, row_b[0]])
    c_short = _pearson_corr(arr_short_a, arr_short_b)
    l_short = _corr_short_label(c_short)

    # mid: next 5 days → 6 points: [1.0, fwd_1..5]
    arr_mid_a = np.concatenate([[1.0], row_a[:5]])
    arr_mid_b = np.concatenate([[1.0], row_b[:5]])
    c_mid  = _pearson_corr(arr_mid_a, arr_mid_b)
    l_mid  = _corr_mid_long_label(c_mid)

    # long: next 20 days → 21 points: [1.0, fwd_1..20]
    arr_long_a = np.concatenate([[1.0], row_a])
    arr_long_b = np.concatenate([[1.0], row_b])
    c_long = _pearson_corr(arr_long_a, arr_long_b)
    l_long = _corr_mid_long_label(c_long)

    return l_short, l_mid, l_long


# ---------------------------------------------------------------------------
# PairDataset – stage-1
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """
    Each sample: (window_A, window_B, label_short, label_mid, label_long)
      windows : float32  (seq_len, n_features)
      labels  : int64    {0, 1, 2}
    """

    def __init__(
        self,
        windows:       dict[str, dict[pd.Timestamp, np.ndarray]],
        df:            pd.DataFrame,    # full data df with future close cols
        dates:         List[pd.Timestamp],
        pairs_per_date: int = config.PAIRS_PER_DATE,
        seed:          int  = config.SEED,
    ):
        rng    = random.Random(seed)
        stocks = list(windows.keys())

        self.X_A: List[np.ndarray] = []
        self.X_B: List[np.ndarray] = []
        self.y1:  List[int] = []
        self.y2:  List[int] = []
        self.y3:  List[int] = []

        for date in dates:
            avail = [s for s in stocks if date in windows.get(s, {})]
            if len(avail) < 2:
                continue

            n = min(pairs_per_date, len(avail) * (len(avail) - 1) // 2)
            sampled = set()
            attempts = 0
            while len(self.X_A) - (len(self.X_A) - len(self.y1)) < n and attempts < n * 5:
                attempts += 1
                a, b = rng.sample(avail, 2)
                key  = (min(a, b), max(a, b))
                if key in sampled:
                    continue
                sampled.add(key)

                labels = _future_corr_labels(df, a, b, date)
                if labels is None:
                    continue

                self.X_A.append(windows[a][date])
                self.X_B.append(windows[b][date])
                self.y1.append(labels[0])
                self.y2.append(labels[1])
                self.y3.append(labels[2])

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
    Each sample: (window, rank_label)
      window     : float32  (seq_len, n_features)
      rank_label : float32  cross-sectional rank in [0, 1]
    """

    def __init__(
        self,
        windows:   dict[str, dict[pd.Timestamp, np.ndarray]],
        df:        pd.DataFrame,   # needs 'label' column (1-day fwd return)
        dates:     List[pd.Timestamp],
    ):
        self.X: List[np.ndarray] = []
        self.y: List[float]      = []

        stocks = list(windows.keys())

        for date in dates:
            # gather forward returns for all stocks available on this date
            day_rets: dict[str, float] = {}
            for stock in stocks:
                if date not in windows.get(stock, {}):
                    continue
                try:
                    val = df.loc[(stock, date), config.LABEL_NAME]
                    if not math.isnan(float(val)):
                        day_rets[stock] = float(val)
                except KeyError:
                    pass

            if len(day_rets) < 2:
                continue

            # cross-sectional rank: 0 = worst, 1 = best
            sorted_stocks = sorted(day_rets, key=day_rets.get)  # type: ignore
            n = len(sorted_stocks)
            rank_map = {s: i / (n - 1) for i, s in enumerate(sorted_stocks)}

            for stock, rank in rank_map.items():
                self.X.append(windows[stock][date])
                self.y.append(rank)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# High-level builder (called from train.py / evaluate.py)
# ---------------------------------------------------------------------------

def build_datasets(
    seq_len: int = config.SEQ_LEN,
) -> Tuple[
    PairDataset, PairDataset,
    StockDataset, StockDataset, StockDataset,
    pd.DataFrame,
]:
    """Load qlib data and construct all datasets for both training stages."""
    init_qlib()

    print("Loading features from qlib …")
    # Load a buffer of extra history before TRAIN_START so the first window
    # on TRAIN_START has enough look-back data.
    from datetime import datetime as _dt, timedelta
    history_start = (
        _dt.strptime(config.TRAIN_START, "%Y-%m-%d")
        - timedelta(days=seq_len * 2)
    ).strftime("%Y-%m-%d")

    df = load_raw_data(history_start, config.TEST_END)

    cal = df.index.get_level_values("datetime").unique().sort_values()

    def _dates(start: str, end: str) -> List[pd.Timestamp]:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return [d for d in cal if s <= d <= e]

    train_dates = _dates(config.TRAIN_START, config.TRAIN_END)
    valid_dates = _dates(config.VALID_START, config.VALID_END)
    test_dates  = _dates(config.TEST_START,  config.TEST_END)

    stocks = df.index.get_level_values("instrument").unique().tolist()

    print(f"  Stocks in universe : {len(stocks)}")
    print(f"  Train dates        : {len(train_dates)} "
          f"({train_dates[0].date()} → {train_dates[-1].date()})")
    print(f"  Valid dates        : {len(valid_dates)} "
          f"({valid_dates[0].date()} → {valid_dates[-1].date()})")
    print(f"  Test  dates        : {len(test_dates)} "
          f"({test_dates[0].date()} → {test_dates[-1].date()})")

    print("Building feature windows …")
    windows = _build_windows(df, stocks, seq_len)

    print("Building PairDatasets (stage-1) …")
    enc_train = PairDataset(windows, df, train_dates)
    enc_valid = PairDataset(windows, df, valid_dates)
    print(f"  enc_train pairs : {len(enc_train):,}")
    print(f"  enc_valid pairs : {len(enc_valid):,}")

    print("Building StockDatasets (stage-2) …")
    pred_train = StockDataset(windows, df, train_dates)
    pred_valid = StockDataset(windows, df, valid_dates)
    pred_test  = StockDataset(windows, df, test_dates)
    print(f"  pred_train samples : {len(pred_train):,}")
    print(f"  pred_valid samples : {len(pred_valid):,}")
    print(f"  pred_test  samples : {len(pred_test):,}")

    return enc_train, enc_valid, pred_train, pred_valid, pred_test, df
