"""
Two-stage MPS training pipeline.

Stage 1 – Encoder pre-training (MultiTask co-movement discrimination).
Stage 2 – Predictor fine-tuning (GRU_Predict return ranking) with frozen encoder.

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
from evaluate import compute_ic_ir, evaluate_predictor


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def epoch_time(start: float, end: float):
    elapsed = end - start
    return int(elapsed // 60), int(elapsed % 60)


# ---------------------------------------------------------------------------
# Stage-1 helpers
# ---------------------------------------------------------------------------

def train_encoder_epoch(
    model: MultiTask,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.train()
    losses, accs = [], []

    for X_A, X_B, y1, y2, y3 in tqdm(loader, desc="  train-enc", leave=False):
        X_A = X_A.to(device)
        X_B = X_B.to(device)
        y1  = y1.to(device)
        y2  = y2.to(device)
        y3  = y3.to(device)

        optimizer.zero_grad()
        s_score, m_score, l_score = model(X_A, X_B)

        loss = (criterion(s_score, y1) +
                criterion(m_score, y2) +
                criterion(l_score, y3)) / 3.0
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # accuracy across all three heads
        preds = torch.cat([s_score, m_score, l_score]).argmax(dim=1).cpu().numpy()
        truth = torch.cat([y1, y2, y3]).cpu().numpy()
        accs.append((preds == truth).mean())

    return {"loss": float(np.mean(losses)), "acc": float(np.mean(accs))}


@torch.no_grad()
def eval_encoder_epoch(
    model: MultiTask,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    losses, accs = [], []

    for X_A, X_B, y1, y2, y3 in tqdm(loader, desc="  eval-enc ", leave=False):
        X_A = X_A.to(device)
        X_B = X_B.to(device)
        y1  = y1.to(device)
        y2  = y2.to(device)
        y3  = y3.to(device)

        s_score, m_score, l_score = model(X_A, X_B)

        loss = (criterion(s_score, y1) +
                criterion(m_score, y2) +
                criterion(l_score, y3)) / 3.0

        losses.append(loss.item())
        preds = torch.cat([s_score, m_score, l_score]).argmax(dim=1).cpu().numpy()
        truth = torch.cat([y1, y2, y3]).cpu().numpy()
        accs.append((preds == truth).mean())

    return {"loss": float(np.mean(losses)), "acc": float(np.mean(accs))}


# ---------------------------------------------------------------------------
# Stage-2 helpers
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
    encoder.eval()          # encoder weights are frozen
    losses = []

    for X, y in tqdm(loader, desc="  train-pre", leave=False):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            s_enc, m_enc, l_enc = encoder.encoder(X)

        score = predictor(s_enc, m_enc, l_enc)
        loss  = criterion(score, y.unsqueeze(1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return {"loss": float(np.mean(losses))}


@torch.no_grad()
def eval_predictor_epoch(
    predictor: GRU_Predict,
    encoder:   MultiTask,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> dict:
    predictor.eval()
    encoder.eval()
    losses = []

    for X, y in tqdm(loader, desc="  eval-pre ", leave=False):
        X = X.to(device)
        y = y.to(device)

        s_enc, m_enc, l_enc = encoder.encoder(X)
        score = predictor(s_enc, m_enc, l_enc)
        loss  = criterion(score, y.unsqueeze(1))
        losses.append(loss.item())

    return {"loss": float(np.mean(losses))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed()
    device = config.DEVICE
    print(f"Device: {device}\n")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    (enc_train_ds, enc_valid_ds,
     pred_train_ds, pred_valid_ds, pred_test_ds,
     labels_df) = build_datasets()

    enc_train_dl  = DataLoader(enc_train_ds,  config.BATCH_SIZE, shuffle=True,  drop_last=False, num_workers=0)
    enc_valid_dl  = DataLoader(enc_valid_ds,  config.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
    pred_train_dl = DataLoader(pred_train_ds, config.BATCH_SIZE, shuffle=True,  drop_last=False, num_workers=0)
    pred_valid_dl = DataLoader(pred_valid_ds, config.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
    pred_test_dl  = DataLoader(pred_test_ds,  config.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)

    # -----------------------------------------------------------------------
    # Stage-1 : Encoder / co-movement training
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

    best_enc_loss = float("inf")
    patience_ctr  = 0

    for epoch in range(1, config.N_EPOCHS_ENC + 1):
        t0 = time.time()
        tr = train_encoder_epoch(enc_model, enc_train_dl, enc_optimizer, enc_criterion, device)
        va = eval_encoder_epoch(enc_model, enc_valid_dl, enc_criterion, device)
        t1 = time.time()
        mins, secs = epoch_time(t0, t1)

        print(f"Epoch {epoch:02d}/{config.N_EPOCHS_ENC}  [{mins}m{secs:02d}s]"
              f"  train_loss={tr['loss']:.4f}  train_acc={tr['acc']:.3f}"
              f"  val_loss={va['loss']:.4f}  val_acc={va['acc']:.3f}", end="")

        if va["loss"] < best_enc_loss:
            best_enc_loss = va["loss"]
            torch.save(enc_model.state_dict(), config.ENCODER_SAVE_PATH)
            print("  ✓ saved", flush=True)
            patience_ctr = 0
        else:
            patience_ctr += 1
            print(flush=True)
            if patience_ctr >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    # Reload best encoder weights
    enc_model.load_state_dict(torch.load(config.ENCODER_SAVE_PATH, map_location=device))

    # Freeze encoder
    for p in enc_model.encoder.parameters():
        p.requires_grad = False

    # -----------------------------------------------------------------------
    # Stage-2 : Predictor training
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

    best_pre_loss = float("inf")
    patience_ctr  = 0

    for epoch in range(1, config.N_EPOCHS_PRE + 1):
        t0 = time.time()
        tr = train_predictor_epoch(predictor, enc_model, pred_train_dl,
                                   pre_optimizer, pre_criterion, device)
        va = eval_predictor_epoch(predictor, enc_model, pred_valid_dl,
                                  pre_criterion, device)

        # Compute IC on validation set each epoch
        val_ic, val_ir = compute_ic_ir(predictor, enc_model, pred_valid_dl, device)

        t1 = time.time()
        mins, secs = epoch_time(t0, t1)

        print(f"Epoch {epoch:02d}/{config.N_EPOCHS_PRE}  [{mins}m{secs:02d}s]"
              f"  train_loss={tr['loss']:.4f}"
              f"  val_loss={va['loss']:.4f}"
              f"  val_IC={val_ic:.4f}  val_IR={val_ir:.4f}", end="")

        if va["loss"] < best_pre_loss:
            best_pre_loss = va["loss"]
            torch.save(predictor.state_dict(), config.PREDICTOR_SAVE_PATH)
            print("  ✓ saved", flush=True)
            patience_ctr = 0
        else:
            patience_ctr += 1
            print(flush=True)
            if patience_ctr >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    # -----------------------------------------------------------------------
    # Final evaluation on held-out test set
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    predictor.load_state_dict(torch.load(config.PREDICTOR_SAVE_PATH, map_location=device))
    evaluate_predictor(predictor, enc_model, pred_test_dl, device, split="Test")


if __name__ == "__main__":
    main()
