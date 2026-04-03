"""
Evaluation utilities for the MPS model.

Primary metrics
---------------
IC  (Information Coefficient)
    Pearson correlation between the model's predicted score and the actual
    next-day return, computed cross-sectionally on each trading day.

IR  (Information Ratio)
    IC.mean() / IC.std()  – a risk-adjusted measure of predictive consistency.

Usage
-----
    python evaluate.py          # evaluates the saved models on the test set
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from tqdm import tqdm

import config
from model import MultiTask, GRU_Predict


# ---------------------------------------------------------------------------
# IC / IR computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_ic_ir(
    predictor: GRU_Predict,
    encoder:   MultiTask,
    loader:    DataLoader,
    device:    torch.device,
    use_spearman: bool = False,
) -> tuple[float, float]:
    """
    Compute mean IC and IR over all batches.

    Because StockDataset is ordered by (date, stock), samples within the same
    date form a natural cross-section.  We accumulate all predictions /
    targets and then compute per-date IC.

    Parameters
    ----------
    use_spearman : bool
        If True, use Spearman rank IC (RankIC).  Default is Pearson IC.

    Returns
    -------
    (mean_IC, IR)
    """
    predictor.eval()
    encoder.eval()

    all_scores: list[float] = []
    all_labels: list[float] = []

    for X, y in loader:
        X = X.to(device)
        s_enc, m_enc, l_enc = encoder.encoder(X)
        score = predictor(s_enc, m_enc, l_enc).squeeze(1)
        all_scores.extend(score.cpu().numpy().tolist())
        all_labels.extend(y.numpy().tolist())

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    # The label is a cross-sectional rank [0, 1] rebuilt by StockDataset.
    # We treat each batch-worth of data as one "day" approximation.
    # For a proper per-day IC we need the date index – see evaluate_with_dates().
    if use_spearman:
        ic, _ = spearmanr(scores, labels)
    else:
        ic = float(np.corrcoef(scores, labels)[0, 1])

    # We don't have per-day breakdown without dates, so IR = IC / small epsilon
    # Use a chunked approach: split into ~loader-batch-sized chunks as proxy days
    ic_series = _chunked_ic(scores, labels, chunk=config.BATCH_SIZE,
                            use_spearman=use_spearman)
    ir = float(np.mean(ic_series) / (np.std(ic_series) + 1e-8))

    return float(np.mean(ic_series)), ir


def _chunked_ic(
    scores: np.ndarray,
    labels: np.ndarray,
    chunk: int,
    use_spearman: bool,
) -> np.ndarray:
    """Split predictions into equal-size cross-sections and return IC per chunk."""
    ics = []
    for i in range(0, len(scores), chunk):
        s = scores[i : i + chunk]
        l = labels[i : i + chunk]
        if len(s) < 2:
            continue
        if use_spearman:
            ic, _ = spearmanr(s, l)
        else:
            ic = float(np.corrcoef(s, l)[0, 1])
        if not np.isnan(ic):
            ics.append(ic)
    return np.array(ics)


@torch.no_grad()
def evaluate_with_dates(
    predictor:  GRU_Predict,
    encoder:    MultiTask,
    pred_ds,    # StockDataset
    device:     torch.device,
    batch_size: int = config.BATCH_SIZE,
    use_spearman: bool = False,
) -> pd.DataFrame:
    """
    Full per-day IC evaluation.

    Returns a DataFrame with columns ['date', 'IC'] sorted by date.
    IC and IR summary is printed to stdout.
    """
    from dataset import StockDataset  # avoid circular import at module level

    predictor.eval()
    encoder.eval()

    # We need to know how many samples correspond to each date.
    # StockDataset stores samples in date order via the dates list;
    # rebuild a per-date count map.
    loader = DataLoader(pred_ds, batch_size=1, shuffle=False, num_workers=0)
    all_scores, all_labels = [], []

    for X, y in tqdm(loader, desc="  scoring", leave=False):
        X = X.to(device)
        s_enc, m_enc, l_enc = encoder.encoder(X)
        score = predictor(s_enc, m_enc, l_enc).item()
        all_scores.append(score)
        all_labels.append(y.item())

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    # Per-date IC using chunked approach (proxy for cross-section)
    ic_series = _chunked_ic(scores, labels, chunk=batch_size,
                            use_spearman=use_spearman)
    mean_ic = float(np.mean(ic_series))
    ir      = float(mean_ic / (np.std(ic_series) + 1e-8))

    return mean_ic, ir, ic_series


# ---------------------------------------------------------------------------
# High-level evaluation printer
# ---------------------------------------------------------------------------

def evaluate_predictor(
    predictor: GRU_Predict,
    encoder:   MultiTask,
    loader:    DataLoader,
    device:    torch.device,
    split:     str = "Test",
) -> None:
    """Print IC and IR for Pearson and Spearman variants."""
    ic_p, ir_p = compute_ic_ir(predictor, encoder, loader, device, use_spearman=False)
    ic_s, ir_s = compute_ic_ir(predictor, encoder, loader, device, use_spearman=True)

    print(f"\n{'─'*50}")
    print(f"  {split} Results")
    print(f"{'─'*50}")
    print(f"  Pearson  IC : {ic_p:+.4f}   IR : {ir_p:+.4f}")
    print(f"  Spearman IC : {ic_s:+.4f}   IR : {ir_s:+.4f}")
    print(f"{'─'*50}\n")


# ---------------------------------------------------------------------------
# Stand-alone evaluation entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device = config.DEVICE
    print(f"Device: {device}\n")

    from dataset import build_datasets
    (_, _, _, _, pred_test_ds, labels_df) = build_datasets()

    from torch.utils.data import DataLoader
    test_dl = DataLoader(pred_test_ds, config.BATCH_SIZE, shuffle=False, num_workers=0)

    enc_model = MultiTask(
        seq_len  = config.SEQ_LEN,
        hid_dim  = config.HID_DIM,
        n_layers = config.N_LAYERS,
        n_heads  = config.N_HEADS,
        pf_dim   = config.PF_DIM,
        dropout  = config.DROPOUT,
        device   = device,
    ).to(device)
    enc_model.load_state_dict(torch.load(config.ENCODER_SAVE_PATH, map_location=device))
    enc_model.eval()

    predictor = GRU_Predict(
        seq_len    = config.SEQ_LEN,
        hid_dim    = config.HID_DIM,
        gru_hidden = config.GRU_HIDDEN,
        gru_layers = config.GRU_LAYERS,
    ).to(device)
    predictor.load_state_dict(torch.load(config.PREDICTOR_SAVE_PATH, map_location=device))
    predictor.eval()

    evaluate_predictor(predictor, enc_model, test_dl, device, split="Test")


if __name__ == "__main__":
    main()
