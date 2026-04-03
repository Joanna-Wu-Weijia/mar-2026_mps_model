"""
Evaluation for the MPS stock-prediction model.

Metrics reported
────────────────
  MSE       Mean Squared Error between predicted rank and actual rank
  Accuracy  Directional accuracy: % of stocks correctly placed in top/bottom half
  IC        Information Coefficient – daily mean Pearson correlation between
            predicted scores and actual next-day returns
  ICIR      IC / std(IC)  (risk-adjusted consistency of the signal)
  Sharpe    Annualised Sharpe Ratio of the long-top-K portfolio
              (daily returns of the top TOP_K_PCT stocks by predicted score)

    python evaluate.py    
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import StockDataset
from model import MultiTask, GRU_Predict


# ---------------------------------------------------------------------------
# Inference pass – collect predictions, ranks, raw returns, and dates
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_predictions(
    predictor: GRU_Predict,
    encoder:   MultiTask,
    dataset:   StockDataset,
    device:    torch.device,
    batch_size: int = config.BATCH_SIZE,
) -> dict:
    """
    Run predictor on every sample in dataset (preserving order).
    Returns dict with keys: scores, ranks, raw_returns, dates.
    """
    predictor.eval()
    encoder.eval()

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    scores_all: list[float] = []
    ranks_all:  list[float] = []
    rets_all:   list[float] = []

    for X, rank, raw_ret in loader:
        X = X.to(device)
        s_enc, m_enc, l_enc = encoder.encoder(X)
        score = predictor(s_enc, m_enc, l_enc).squeeze(1)
        scores_all.extend(score.cpu().numpy().tolist())
        ranks_all.extend(rank.numpy().tolist())
        rets_all.extend(raw_ret.numpy().tolist())

    return {
        "scores":      np.array(scores_all),
        "ranks":       np.array(ranks_all),
        "raw_returns": np.array(rets_all),
        "dates":       dataset.dates,          # List[pd.Timestamp]
    }


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def _mse(scores: np.ndarray, ranks: np.ndarray) -> float:
    return float(np.mean((scores - ranks) ** 2))


def _accuracy(scores: np.ndarray, ranks: np.ndarray) -> float:
    """
    Directional accuracy: fraction of stocks correctly predicted as
    top-half (rank >= 0.5) vs bottom-half (rank < 0.5).
    """
    pred_top   = scores >= 0.5
    actual_top = ranks  >= 0.5
    return float(np.mean(pred_top == actual_top))


def _daily_ic(data: dict, use_spearman: bool = False) -> np.ndarray:
    """Compute per-day IC series between predicted scores and raw returns."""
    from scipy.stats import spearmanr

    by_date: dict[object, list] = defaultdict(list)
    for score, ret, date in zip(data["scores"], data["raw_returns"], data["dates"]):
        by_date[date].append((score, ret))

    ic_series = []
    for date in sorted(by_date):
        pairs = by_date[date]
        if len(pairs) < 2:
            continue
        s = np.array([p[0] for p in pairs])
        r = np.array([p[1] for p in pairs])
        if use_spearman:
            ic, _ = spearmanr(s, r)
        else:
            ic = float(np.corrcoef(s, r)[0, 1])
        if not np.isnan(ic):
            ic_series.append(ic)

    return np.array(ic_series)


def _sharpe(data: dict) -> float:
    """
    Long-top-K daily portfolio Sharpe Ratio (annualised).
    On each day, invest equally in the top TOP_K_PCT stocks by predicted score.
    Portfolio daily return = average raw return of selected stocks.
    """
    by_date: dict[object, list] = defaultdict(list)
    for score, ret, date in zip(data["scores"], data["raw_returns"], data["dates"]):
        by_date[date].append((score, ret))

    port_rets = []
    k_pct = config.TOP_K_PCT

    for date in sorted(by_date):
        pairs = sorted(by_date[date], key=lambda x: x[0], reverse=True)
        k = max(1, int(len(pairs) * k_pct))
        top_rets = [p[1] for p in pairs[:k]]
        port_rets.append(float(np.mean(top_rets)))

    if len(port_rets) < 2:
        return float("nan")

    port_rets = np.array(port_rets) - config.RISK_FREE_RATE
    mean_r = float(np.mean(port_rets))
    std_r  = float(np.std(port_rets, ddof=1))
    if std_r < 1e-8:
        return float("nan")

    return float(mean_r / std_r * np.sqrt(config.ANNUAL_FACTOR))


# ---------------------------------------------------------------------------
# All-in-one evaluation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    predictor:   GRU_Predict,
    encoder:     MultiTask,
    dataset:     StockDataset,
    device:      torch.device,
    use_spearman: bool = False,
) -> Dict[str, float]:
    """
    Compute MSE, Accuracy, IC, ICIR, Sharpe Ratio in one pass.

    Parameters
    ----------
    use_spearman : bool
        If True, use Spearman rank IC (RankIC). Default: Pearson IC.
    """
    data = _collect_predictions(predictor, encoder, dataset, device)

    ic_series = _daily_ic(data, use_spearman=use_spearman)
    mean_ic   = float(np.mean(ic_series))
    icir      = float(mean_ic / (np.std(ic_series, ddof=1) + 1e-8))

    return {
        "MSE":      _mse(data["scores"], data["ranks"]),
        "Accuracy": _accuracy(data["scores"], data["ranks"]),
        "IC":       mean_ic,
        "ICIR":     icir,
        "Sharpe":   _sharpe(data),
    }


def print_metrics(metrics: Dict[str, float], split: str = "Test") -> None:
    print(f"\n{'─'*52}")
    print(f"  {split} Metrics")
    print(f"{'─'*52}")
    print(f"  MSE      (↓ better) : {metrics['MSE']:+.6f}")
    print(f"  Accuracy (↑ better) : {metrics['Accuracy']:+.4f}")
    print(f"  IC       (↑ better) : {metrics['IC']:+.4f}")
    print(f"  ICIR     (↑ better) : {metrics['ICIR']:+.4f}")
    print(f"  Sharpe   (↑ better) : {metrics['Sharpe']:+.4f}")
    print(f"{'─'*52}\n")


# ---------------------------------------------------------------------------
# Stand-alone evaluation entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device = config.DEVICE
    print(f"Device: {device}\n")

    from dataset import build_datasets
    _, _, pred_test, _ = build_datasets()

    encoder = MultiTask(
        seq_len  = config.SEQ_LEN,
        hid_dim  = config.HID_DIM,
        n_layers = config.N_LAYERS,
        n_heads  = config.N_HEADS,
        pf_dim   = config.PF_DIM,
        dropout  = config.DROPOUT,
        device   = device,
    ).to(device)
    encoder.load_state_dict(
        torch.load(config.ENCODER_SAVE_PATH, map_location=device))
    encoder.eval()

    predictor = GRU_Predict(
        seq_len    = config.SEQ_LEN,
        hid_dim    = config.HID_DIM,
        gru_hidden = config.GRU_HIDDEN,
        gru_layers = config.GRU_LAYERS,
    ).to(device)
    predictor.load_state_dict(
        torch.load(config.PREDICTOR_SAVE_PATH, map_location=device))
    predictor.eval()

    metrics = compute_all_metrics(predictor, encoder, pred_test, device)
    print_metrics(metrics, split="Test")


if __name__ == "__main__":
    main()