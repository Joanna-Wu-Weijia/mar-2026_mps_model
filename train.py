"""
Two-stage MPS training pipeline (no validation set).

Stage 1 – Encoder pre-training (MultiTask co-movement discrimination).
          Runs for N_EPOCHS_ENC epochs; reports loss and classification accuracy.

Stage 2 – Predictor fine-tuning (GRU_Predict return ranking) with frozen encoder.
          Runs for N_EPOCHS_PRE epochs; reports MSE loss.

Final test evaluation reports all five metrics:
  MSE, Accuracy, IC, ICIR, Sharpe Ratio.

Run:
    python train.py
"""

from __future__ import annotations

import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model import MultiTask, GRU_Predict
from dataset import build_datasets
from evaluate import compute_all_metrics, print_metrics


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_time(start: float, end: float):
    e = end - start
    return int(e // 60), int(e % 60)


# ---------------------------------------------------------------------------
# Stage-1 helpers
# ---------------------------------------------------------------------------

def train_encoder_epoch(
    model:     MultiTask,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> dict:
    model.train()
    losses, accs = [], []

    for X_A, X_B, y1, y2, y3 in tqdm(loader, desc="  enc-train", leave=False):
        X_A, X_B = X_A.to(device), X_B.to(device)
        y1, y2, y3 = y1.to(device), y2.to(device), y3.to(device)

        optimizer.zero_grad()
        s_score, m_score, l_score = model(X_A, X_B)

        # equal loss weights λ=γ=μ=1/3  (paper Section 3)
        loss = (criterion(s_score, y1) +
                criterion(m_score, y2) +
                criterion(l_score, y3)) / 3.0
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = torch.cat([s_score, m_score, l_score]).argmax(dim=1).cpu().numpy()
        truth = torch.cat([y1, y2, y3]).cpu().numpy()
        accs.append(float((preds == truth).mean()))

    return {"loss": float(np.mean(losses)), "acc": float(np.mean(accs))}


# ---------------------------------------------------------------------------
# Stage-2 helpers  (DataLoader yields (X, rank, raw_return))
# ---------------------------------------------------------------------------

def train_predictor_epoch(
    predictor:  GRU_Predict,
    encoder:    MultiTask,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
) -> dict:
    predictor.train()
    encoder.eval()     # encoder weights are frozen
    losses = []

    for X, rank, _raw in loader:
        X, rank = X.to(device), rank.to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            s_enc, m_enc, l_enc = encoder.encoder(X)

        score = predictor(s_enc, m_enc, l_enc)
        loss  = criterion(score, rank.unsqueeze(1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return {"loss": float(np.mean(losses))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed()
    device = config.DEVICE
    print(f"Device : {device}")
    print(f"Train  : {config.TRAIN_START} → {config.TRAIN_END}")
    print(f"Test   : {config.TEST_START}  → {config.TEST_END}\n")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    enc_train_ds, pred_train_ds, pred_test_ds, _ = build_datasets()

    # Stage-1 loader
    enc_train_dl = DataLoader(enc_train_ds,  config.BATCH_SIZE,
                              shuffle=True,  drop_last=False, num_workers=0)
    # Stage-2 loaders
    pred_train_dl = DataLoader(pred_train_ds, config.BATCH_SIZE,
                               shuffle=True,  drop_last=False, num_workers=0)

    # -----------------------------------------------------------------------
    # Stage-1: Encoder pre-training
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("STAGE 1 – Encoder pre-training (co-movement discrimination)")
    print("=" * 60)

    enc_model = MultiTask(
        seq_len  = config.SEQ_LEN,
        hid_dim  = config.HID_DIM,
        n_layers = config.N_LAYERS,
        n_heads  = config.N_HEADS,
        pf_dim   = config.PF_DIM,
        dropout  = config.DROPOUT,
        device   = device,
    ).to(device)

    enc_criterion = nn.CrossEntropyLoss()
    enc_optimizer = torch.optim.Adam(enc_model.parameters(), lr=config.LR)

    for epoch in range(1, config.N_EPOCHS_ENC + 1):
        t0  = time.time()
        log = train_encoder_epoch(enc_model, enc_train_dl,
                                  enc_optimizer, enc_criterion, device)
        mins, secs = epoch_time(t0, time.time())
        print(f"Epoch {epoch:02d}/{config.N_EPOCHS_ENC}  [{mins}m{secs:02d}s]"
              f"  loss={log['loss']:.4f}  acc={log['acc']:.4f}")

    torch.save(enc_model.state_dict(), config.ENCODER_SAVE_PATH)
    print(f"\nEncoder saved → {config.ENCODER_SAVE_PATH}")

    # Freeze encoder
    for p in enc_model.encoder.parameters():
        p.requires_grad = False

    # -----------------------------------------------------------------------
    # Stage-2: Predictor fine-tuning
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2 – Predictor fine-tuning (return ranking)")
    print("=" * 60)

    predictor = GRU_Predict(
        seq_len    = config.SEQ_LEN,
        hid_dim    = config.HID_DIM,
        gru_hidden = config.GRU_HIDDEN,
        gru_layers = config.GRU_LAYERS,
    ).to(device)

    pre_criterion = nn.MSELoss()
    pre_optimizer = torch.optim.Adam(predictor.parameters(), lr=config.LR)

    for epoch in range(1, config.N_EPOCHS_PRE + 1):
        t0  = time.time()
        log = train_predictor_epoch(predictor, enc_model, pred_train_dl,
                                    pre_optimizer, pre_criterion, device)
        mins, secs = epoch_time(t0, time.time())
        print(f"Epoch {epoch:02d}/{config.N_EPOCHS_PRE}  [{mins}m{secs:02d}s]"
              f"  MSE={log['loss']:.6f}")

    torch.save(predictor.state_dict(), config.PREDICTOR_SAVE_PATH)
    print(f"\nPredictor saved → {config.PREDICTOR_SAVE_PATH}")

    # -----------------------------------------------------------------------
    # Final evaluation on held-out test set
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    metrics = compute_all_metrics(predictor, enc_model, pred_test_ds, device)
    print_metrics(metrics, split="Test")


if __name__ == "__main__":
    main()
