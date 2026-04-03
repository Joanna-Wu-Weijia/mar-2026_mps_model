"""
Data loading and preprocessing for the MPS model using local qlib data.

Data directory layout (QLIB_DATA_PATH in config.py):
    my_qlib_data/
    ├── calendars/day.txt
    ├── features/<stock>/*.day.bin
    └── instruments/csi300.txt ...

Datasets produced:
  PairDataset   – stage-1 (MultiTask co-movement pre-training)
  StockDataset  – stage-2 (GRU_Predict return-ranking training / evaluation)

Co-movement labels (paper Section 3, original build_data.ipynb):
  Pearson correlation of FUTURE close price sequences between stock pairs.
  k_short=1, k_mid=5, k_long=20 trading days.
  Label encoding: 0=uncorrelated, 1=positive co-movement, 2=negative co-movement
  Boundaries: short → r1=2/3, r2=-1/3 | mid/long → r1=0.5, r2=-0.5
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
# Raw data loading
# ---------------------------------------------------------------------------

def load_raw_data(start: str, end: str) -> pd.DataFrame:
    """
    Load 6 historical features + 1-day forward return label
    + 20 future close ratios (for co-movement labels in stage-1).

    Returns DataFrame with MultiIndex (instrument, datetime).
    """
    instruments = D.instruments(config.UNIVERSE)

    fields = (config.FEATURE_FIELDS
              + [config.LABEL_FIELD]
              + config.FUTURE_CLOSE_FIELDS)
    names  = (config.FEATURE_NAMES
              + [config.LABEL_NAME]
              + config.FUTURE_CLOSE_NAMES)

    df = D.features(instruments, fields,
                    start_time=start, end_time=end, freq="day")
    df.columns = names
    df.index.names = ["instrument", "datetime"]
    return df


# ---------------------------------------------------------------------------
# Per-stock z-score normalisation with 3-sigma clip
# ---------------------------------------------------------------------------

def _zscore_clip(x: np.ndarray) -> np.ndarray:
    mu  = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x,  axis=0, keepdims=True) + 1e-8
    return np.clip((x - mu) / std, -3.0, 3.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Window builder  →  {stock: {date: np.ndarray(seq_len, n_features)}}
# ---------------------------------------------------------------------------

def _build_windows(
    df: pd.DataFrame,
    stocks: List[str],
    seq_len: int,
) -> dict[str, dict[pd.Timestamp, np.ndarray]]:
    feat_cols = config.FEATURE_NAMES
    windows: dict[str, dict[pd.Timestamp, np.ndarray]] = {}

    for stock in stocks:
        if stock not in df.index.get_level_values("instrument"):
            continue

        sub = df.loc[stock, feat_cols].sort_index().dropna(how="any")
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
# Co-movement label helpers  (paper Section 3)
# ---------------------------------------------------------------------------

def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _short_label(corr: float) -> int:
    """k=1  boundary r1=2/3, r2=-1/3"""
    if corr >= 2 / 3:
        return 1
    if corr <= -1 / 3:
        return 2
    return 0


def _mid_long_label(corr: float) -> int:
    """k=5,20  boundary r1=0.5, r2=-0.5"""
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
    Compute co-movement labels using future close price ratios.
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

    # short (k=1): 2-point sequence [today=1.0, t+1]
    a_short = np.array([1.0, row_a[0]])
    b_short = np.array([1.0, row_b[0]])
    l_short = _short_label(_pearson_corr(a_short, b_short))

    # mid (k=5): 6-point sequence [1.0, fwd_1..5]
    a_mid = np.concatenate([[1.0], row_a[:5]])
    b_mid = np.concatenate([[1.0], row_b[:5]])
    l_mid = _mid_long_label(_pearson_corr(a_mid, b_mid))

    # long (k=20): 21-point sequence [1.0, fwd_1..20]
    a_long = np.concatenate([[1.0], row_a])
    b_long = np.concatenate([[1.0], row_b])
    l_long = _mid_long_label(_pearson_corr(a_long, b_long))

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
        windows:        dict[str, dict[pd.Timestamp, np.ndarray]],
        df:             pd.DataFrame,
        dates:          List[pd.Timestamp],
        pairs_per_date: int = config.PAIRS_PER_DATE,
        seed:           int = config.SEED,
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

            n        = min(pairs_per_date, len(avail) * (len(avail) - 1) // 2)
            sampled  = set()
            attempts = 0
            added    = 0
            while added < n and attempts < n * 5:
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
                added += 1

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
    Each sample: (window, rank_label, raw_return)
      window      : float32  (seq_len, n_features)
      rank_label  : float32  cross-sectional rank in [0, 1]  (training target)
      raw_return  : float32  actual 1-day forward return     (for Sharpe/IC)

    Also exposes:
      self.dates         : List[pd.Timestamp]  – date for each sample
      self.raw_returns   : List[float]         – raw forward return per sample
    """

    def __init__(
        self,
        windows: dict[str, dict[pd.Timestamp, np.ndarray]],
        df:      pd.DataFrame,
        dates:   List[pd.Timestamp],
    ):
        self.X:           List[np.ndarray]   = []
        self.ranks:       List[float]        = []
        self.raw_returns: List[float]        = []
        self.dates:       List[pd.Timestamp] = []

        stocks = list(windows.keys())

        for date in dates:
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

            # cross-sectional rank: 0 = worst return, 1 = best return
            sorted_s = sorted(day_rets, key=day_rets.get)  # type: ignore
            n        = len(sorted_s)
            rank_map = {s: i / (n - 1) for i, s in enumerate(sorted_s)}

            for stock, rank in rank_map.items():
                self.X.append(windows[stock][date])
                self.ranks.append(rank)
                self.raw_returns.append(day_rets[stock])
                self.dates.append(date)

    def __len__(self) -> int:
        return len(self.ranks)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.ranks[idx],       dtype=torch.float32),
            torch.tensor(self.raw_returns[idx], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------

def build_datasets(
    seq_len: int = config.SEQ_LEN,
) -> Tuple[
    PairDataset,    # enc_train
    StockDataset,   # pred_train
    StockDataset,   # pred_test
    pd.DataFrame,   # full df (for any post-hoc analysis)
]:
    """Load qlib data and build all datasets. No validation split."""
    init_qlib()

    print("Loading features from qlib …")
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
    test_dates  = _dates(config.TEST_START,  config.TEST_END)

    stocks = df.index.get_level_values("instrument").unique().tolist()

    print(f"  Stocks : {len(stocks)}")
    print(f"  Train  : {len(train_dates)} days "
          f"({train_dates[0].date()} → {train_dates[-1].date()})")
    print(f"  Test   : {len(test_dates)} days "
          f"({test_dates[0].date()} → {test_dates[-1].date()})")

    print("Building feature windows …")
    windows = _build_windows(df, stocks, seq_len)

    print("Building PairDataset (stage-1 train) …")
    enc_train = PairDataset(windows, df, train_dates)
    print(f"  pairs : {len(enc_train):,}")

    print("Building StockDatasets (stage-2) …")
    pred_train = StockDataset(windows, df, train_dates)
    pred_test  = StockDataset(windows, df, test_dates)
    print(f"  pred_train : {len(pred_train):,}  |  pred_test : {len(pred_test):,}")

    return enc_train, pred_train, pred_test, df
