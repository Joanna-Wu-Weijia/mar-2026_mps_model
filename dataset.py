"""
Data loading and preprocessing for the MPS model using local qlib data.

Processing pipeline mirrors build_data.ipynb from ECNU-CILAB/MPS, except
that raw data comes from a local qlib store instead of qsdata.

Key design decisions that match the original exactly:
  1. Features    : raw OHLCV ["open","close","high","low","turnover","volume"]
  2. Zero-filter : drop rows where volume=0 or turnover=0  (original: same)
  3. Normalisation:
       - Training slice → 3-sigma clip, then z-score (fit on train)
       - Test slice     → apply the SAME clip bounds + mean/std from train
                          (no data leakage, matches ds_filter_extreme_3sigma
                          + ds_standardize_zscore in the original)
  4. Windows     : 20-day rolling windows, newest-first  (matches original)
  5. Co-movement labels (stage-1 pretext task):
       corr_short  (k=1) : Pearson of [today, t+1] close prices
                           → 1 if ≥ 2/3, 2 if ≤ −1/3, else 0
       corr_mid    (k=5) : Pearson of [today, t+1..5]
                           → 1 if ≥ 0.5, 2 if ≤ −0.5, else 0
       corr_long   (k=20): Pearson of [today, t+1..20]
                           → same thresholds as corr_mid

What differs from the original (intentional):
  - We sample up to PAIRS_PER_DATE pairs per date rather than one fixed pair
    per stock; this gives the encoder more diverse co-movement signal per pass.
  - Data source is qlib D.features() instead of qsdata.get_price().
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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
    Load raw OHLCV features + 1-day forward return label
    + 20 future close ratios (for co-movement labels in stage-1).

    Returns DataFrame with MultiIndex (instrument, datetime).
    Rows where volume=0 or turnover=0 are dropped (same as original).
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

    # Drop illiquid days: volume=0 or turnover=0 (build_data.ipynb line 1)
    df = df[(df["volume"] > 0) & (df["turnover"] > 0)]
    return df


# ---------------------------------------------------------------------------
# Global normalisation  (mirrors build_data.ipynb normalisation block)
# ---------------------------------------------------------------------------

class GlobalNormaliser:
    """
    Fits 3-sigma clip bounds + z-score (mean, std) on training data,
    then transforms both train and test slices using those same parameters.

    Mirrors ds_filter_extreme_3sigma() + ds_standardize_zscore() from the
    original, applied globally (not per-stock) for each feature column.
    """

    def __init__(self, n_sigma: float = 3.0):
        self.n_sigma = n_sigma
        self.params: Dict[str, Dict] = {}   # {col: {min, max, mean, std}}

    def fit_transform(
        self, df: pd.DataFrame, cols: List[str]
    ) -> pd.DataFrame:
        """Fit on df and return the clipped+normalised copy."""
        df = df.copy()
        for col in cols:
            s    = df[col].dropna()
            mean = float(s.mean())
            std  = float(s.std())
            lo   = mean - self.n_sigma * std
            hi   = mean + self.n_sigma * std

            df[col] = df[col].clip(lo, hi)
            df[col] = (df[col] - mean) / (std + 1e-8)

            self.params[col] = dict(mean=mean, std=std, lo=lo, hi=hi)

        return df

    def transform(
        self, df: pd.DataFrame, cols: List[str]
    ) -> pd.DataFrame:
        """Apply previously fitted parameters to a new slice."""
        df = df.copy()
        for col in cols:
            p    = self.params[col]
            # same clip range from train, same mean/std from train
            df[col] = df[col].clip(p["lo"], p["hi"])
            df[col] = (df[col] - p["mean"]) / (p["std"] + 1e-8)
        return df


def normalise_split(
    df: pd.DataFrame,
    train_end: str,
    cols: List[str],
) -> Tuple[pd.DataFrame, GlobalNormaliser]:
    """
    Fit normaliser on rows up to train_end, transform the whole df.
    Returns the normalised df and the fitted normaliser.
    """
    split = pd.Timestamp(train_end)
    dates = df.index.get_level_values("datetime")

    train_mask = dates <= split
    test_mask  = dates >  split

    norm = GlobalNormaliser()

    train_part = norm.fit_transform(df[train_mask], cols)
    test_part  = norm.transform(df[test_mask], cols)

    return pd.concat([train_part, test_part]).sort_index(), norm


# ---------------------------------------------------------------------------
# Window builder  →  {stock: {date: np.ndarray(seq_len, n_features)}}
#
# Window order: newest-first  [today, t-1, …, t-19]
# This matches the original:
#   daily20_features = [features, features-1, …, features-19]
# ---------------------------------------------------------------------------

def _build_windows(
    df: pd.DataFrame,
    stocks: List[str],
    seq_len: int,
) -> Dict[str, Dict[pd.Timestamp, np.ndarray]]:
    feat_cols = config.FEATURE_NAMES
    windows: Dict[str, Dict[pd.Timestamp, np.ndarray]] = {}

    for stock in stocks:
        if stock not in df.index.get_level_values("instrument"):
            continue

        sub = df.loc[stock, feat_cols].sort_index().dropna(how="any")
        if len(sub) < seq_len:
            continue

        vals  = sub.values.astype(np.float32)
        dates = sub.index.tolist()

        stock_wins: Dict[pd.Timestamp, np.ndarray] = {}
        for i in range(seq_len - 1, len(dates)):
            # newest-first: [today, t-1, …, t-seq_len+1]
            win = vals[i : i - seq_len : -1]   # shape (seq_len, n_features)
            if win.shape[0] == seq_len:
                stock_wins[dates[i]] = win

        if stock_wins:
            windows[stock] = stock_wins

    return windows


# ---------------------------------------------------------------------------
# Co-movement label helpers  (paper Section 3 / build_data.ipynb)
# ---------------------------------------------------------------------------

def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _short_label(corr: float) -> int:
    """k=1: r1=2/3, r2=-1/3"""
    if corr >= 2 / 3:
        return 1
    if corr <= -1 / 3:
        return 2
    return 0


def _mid_long_label(corr: float) -> int:
    """k=5,20: r1=0.5, r2=-0.5"""
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
) -> Optional[Tuple[int, int, int]]:
    """
    Compute co-movement labels from FUTURE close price ratios.
    fwd_close_i = Ref($close,-i)/$close  → normalised future close sequence.

    Returns (label_short, label_mid, label_long) or None if data is missing.
    """
    fwd_cols = config.FUTURE_CLOSE_NAMES
    try:
        row_a = df.loc[(stock_a, date), fwd_cols].values.astype(float)
        row_b = df.loc[(stock_b, date), fwd_cols].values.astype(float)
    except KeyError:
        return None

    if np.any(np.isnan(row_a)) or np.any(np.isnan(row_b)):
        return None

    # short (k=1): 2 points [today=1.0, t+1]
    l_short = _short_label(
        _pearson_corr(np.array([1.0, row_a[0]]),
                      np.array([1.0, row_b[0]])))

    # mid (k=5): 6 points [1.0, fwd_1..5]
    l_mid = _mid_long_label(
        _pearson_corr(np.concatenate([[1.0], row_a[:5]]),
                      np.concatenate([[1.0], row_b[:5]])))

    # long (k=20): 21 points [1.0, fwd_1..20]
    l_long = _mid_long_label(
        _pearson_corr(np.concatenate([[1.0], row_a]),
                      np.concatenate([[1.0], row_b])))

    return l_short, l_mid, l_long


# ---------------------------------------------------------------------------
# PairDataset – stage-1
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """
    Each sample: (window_A, window_B, label_short, label_mid, label_long)
      windows : float32  (seq_len, n_features)   newest-first
      labels  : int64    {0=uncorrelated, 1=positive, 2=negative}
    """

    def __init__(
        self,
        windows:        Dict[str, Dict[pd.Timestamp, np.ndarray]],
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
            added    = 0
            attempts = 0
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
      window      : float32  (seq_len, n_features)  newest-first
      rank_label  : float32  cross-sectional rank [0,1]  (training target)
      raw_return  : float32  actual 1-day forward return (for IC / Sharpe)

    Attributes:
      self.dates       : List[pd.Timestamp]   date for each sample
      self.raw_returns : List[float]          raw return for each sample
    """

    def __init__(
        self,
        windows: Dict[str, Dict[pd.Timestamp, np.ndarray]],
        df:      pd.DataFrame,
        dates:   List[pd.Timestamp],
    ):
        self.X:           List[np.ndarray]   = []
        self.ranks:       List[float]        = []
        self.raw_returns: List[float]        = []
        self.dates:       List[pd.Timestamp] = []

        stocks = list(windows.keys())

        for date in dates:
            day_rets: Dict[str, float] = {}
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
            # matches groupby('dt')['close_rtn'].rank() / max(rank) in original
            sorted_s = sorted(day_rets, key=day_rets.get)   # type: ignore
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
    PairDataset,
    StockDataset,
    StockDataset,
    pd.DataFrame,
]:
    """
    Load qlib data, normalise (train stats → test), build all datasets.
    No validation split.
    """
    init_qlib()

    print("Loading features from qlib …")
    history_start = (
        datetime.strptime(config.TRAIN_START, "%Y-%m-%d")
        - timedelta(days=seq_len * 2)
    ).strftime("%Y-%m-%d")

    df = load_raw_data(history_start, config.TEST_END)

    # ------------------------------------------------------------------ #
    # Global normalisation – fit on training slice, apply same params     #
    # to test slice.  Matches build_data.ipynb normalisation block.       #
    # ------------------------------------------------------------------ #
    print("Normalising (train-stats applied to test) …")
    df, _ = normalise_split(df, config.TRAIN_END, config.FEATURE_NAMES)

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

    print("Building feature windows (newest-first, 20-day) …")
    windows = _build_windows(df, stocks, seq_len)

    print("Building PairDataset (stage-1 train) …")
    enc_train = PairDataset(windows, df, train_dates)
    print(f"  pairs : {len(enc_train):,}")

    print("Building StockDatasets (stage-2) …")
    pred_train = StockDataset(windows, df, train_dates)
    pred_test  = StockDataset(windows, df, test_dates)
    print(f"  pred_train : {len(pred_train):,}  |  pred_test : {len(pred_test):,}")

    return enc_train, pred_train, pred_test, df
