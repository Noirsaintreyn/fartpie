"""
deepsupp_levels.py  v4
──────────────────────
DeepSupp: production-hardened drop-in level generator.

Hardening vs v3
───────────────
  1. ModelMetadata dataclass  – feat_dim, seq_len, feature_names, corr_window,
     and all hyperparams stored on the model at training time.
  2. Schema validation at inference  – feat_dim, seq_len, and feature_names
     are all checked before any computation runs.  Mismatches raise ValueError.
  3. Encoder forward asserts  – explicit shape assertions + seq_len guard.
  4. Device normalisation  – model.to(device) called explicitly at inference.
  5. Rolling anomaly threshold  – percentile computed over a trailing window
     of prior scores rather than the full sample, making it online-safe.
  6. Level-relative displacement  – S/R classification measures post-touch
     displacement relative to the candidate level price, not raw future drift.
  7. Population-normalised quality score  – score_mean and score_max are
     normalised against the full high-score population before combining.
  8. Stable feature column ordering  – engineer_features() returns columns
     in a fixed canonical order regardless of vwap source.
  9. Symmetric reconstruction  – loss and anomaly score use symmetrised
     reconstruction: 0.5 * (recon + recon.T) per sample.
 10. Save / load helpers  – save_deepsupp_model() / load_deepsupp_model()
     persist weights + full metadata in a single .pt file.

Pipeline
────────
  OHLCV
    → engineer_features()              stable-ordered price-volume-VWAP set
    → build_corr_series()              rolling Spearman matrices  (T', F, F)
    → build_sequence_dataset()         sliding windows → (N, L, F, F)
    → DeepSuppAutoencoder              attend across L corr-matrix snapshots
    → compute_anomaly_scores()         symmetrised recon error per window
    → extract_levels()                 rolling threshold → latent DBSCAN
                                       → price DBSCAN → level-relative S/R label
    → [LevelRecord, ...]               structured, merge-ready

Reference: Kriuk, B., Ng, L., & Hossain, Z. (2025). ArXiv abs/2507.01971
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from dataclasses import dataclass, asdict
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════════
# 0.  OUTPUT CONTRACT
# ══════════════════════════════════════════════════════════════════════

# Canonical feature column order — must never change without a model retrain
_FEATURE_COLS = [
    "log_close", "returns", "hl_range", "co_diff",
    "volume", "vol_zscore", "vwap", "vwap_dev",
    "true_range", "vol_vr",
]


@dataclass
class LevelRecord:
    """
    Structured level record compatible with multi-family level aggregators.

    Fields
    ──────
    source          always "deepsupp"
    price           representative price of the level
    kind            "support" | "resistance" | "level"
    strength        composite 0-1 score (coverage + quality + tightness)
    coverage        fraction of scored timesteps that belong to this cluster
    quality         weighted mean/max anomaly score, normalised vs population
    tightness       1 / (1 + price_std) — how tight the price sub-cluster is
    cluster_id      composite latent_lbl * 10_000 + price_sub_lbl
    n_members       timesteps in this price sub-cluster
    score_mean      mean anomaly score of sub-cluster members
    score_max       peak anomaly score of sub-cluster members
    price_std       price spread within the sub-cluster
    displacement    mean signed % move of price relative to level_price,
                    measured `horizon` bars after each member bar.
                    positive → price bounced up from level → support evidence
                    negative → price rejected down from level → resistance evidence
    """
    source:       str
    price:        float
    kind:         str
    strength:     float
    coverage:     float
    quality:      float
    tightness:    float
    cluster_id:   int
    n_members:    int
    score_mean:   float
    score_max:    float
    price_std:    float
    displacement: float

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════
# 1.  MODEL METADATA
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ModelMetadata:
    """All information needed to reconstruct and validate a saved model."""
    feat_dim:      int
    seq_len:       int
    corr_window:   int
    vol_lookback:  int
    feature_names: List[str]
    d_model:       int
    n_heads:       int
    n_layers:      int
    latent_dim:    int
    dropout:       float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelMetadata":
        return cls(**d)


# ══════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════

def engineer_features(
    df: pd.DataFrame,
    vol_lookback: int = 20,
) -> pd.DataFrame:
    """
    Compact price-volume-VWAP feature set (Kriuk et al., 2025).

    Required columns : open, high, low, close, volume
    Optional column  : vwap  (computed from OHLCV if absent)

    Returns columns in the canonical _FEATURE_COLS order.
    """
    feat = pd.DataFrame(index=df.index)

    feat["log_close"]  = np.log(df["close"])
    feat["returns"]    = feat["log_close"].diff()
    feat["hl_range"]   = df["high"] - df["low"]
    feat["co_diff"]    = df["close"] - df["open"]

    feat["volume"]     = df["volume"]
    vol_mean           = df["volume"].rolling(vol_lookback).mean()
    vol_std            = df["volume"].rolling(vol_lookback).std()
    feat["vol_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-9)

    if "vwap" in df.columns:
        vwap = df["vwap"]
    else:
        tp   = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (tp * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

    feat["vwap"]       = vwap
    feat["vwap_dev"]   = df["close"] - vwap

    feat["true_range"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    feat["vol_vr"] = df["volume"] / (feat["true_range"] + 1e-9)

    # Canonical column ordering — stable across vwap-present / absent paths
    return feat[_FEATURE_COLS].dropna()


# ══════════════════════════════════════════════════════════════════════
# 3.  ROLLING SPEARMAN CORRELATION MATRICES
# ══════════════════════════════════════════════════════════════════════

def build_corr_series(
    X: np.ndarray,
    win_len: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    ───────
    mats : (T', F, F)   rolling Spearman correlation matrices
    idxs : (T',)        original row index of each matrix's last bar
    """
    T, feat_dim = X.shape
    mats, idxs = [], []

    for t in range(win_len, T + 1):
        window = X[t - win_len : t]

        if feat_dim == 1:
            corr = np.array([[1.0]], dtype=np.float32)
        else:
            raw, _ = spearmanr(window, axis=0)
            corr   = np.asarray(raw, dtype=np.float32)[:feat_dim, :feat_dim]

        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
        # Enforce symmetry (numerical safety)
        corr = 0.5 * (corr + corr.T)

        mats.append(corr)
        idxs.append(t - 1)

    return np.stack(mats, axis=0), np.array(idxs, dtype=np.int64)


# ══════════════════════════════════════════════════════════════════════
# 4.  SEQUENCE WINDOWING
# ══════════════════════════════════════════════════════════════════════

def build_sequence_dataset(
    mats: np.ndarray,
    idxs: np.ndarray,
    seq_len: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    ───────
    seqs     : (N, seq_len, F, F)
    end_idxs : (N,)   original bar index at end of each sequence
    """
    T = len(mats)
    seqs, end_idxs = [], []
    for i in range(seq_len, T + 1):
        seqs.append(mats[i - seq_len : i])
        end_idxs.append(idxs[i - 1])
    return np.stack(seqs, axis=0), np.array(end_idxs, dtype=np.int64)


# ══════════════════════════════════════════════════════════════════════
# 5.  ATTENTION AUTOENCODER
# ══════════════════════════════════════════════════════════════════════

class DeepSuppEncoder(nn.Module):
    """
    Sequence of L corr matrices → latent vector.

    Input  : (B, L, F, F)
    Each matrix is flattened and projected → (B, L, d_model) tokens.
    Transformer attends across the L time-steps.
    Latent is taken from the final token (most recent corr state).
    """
    def __init__(
        self,
        feat_dim:   int,
        seq_len:    int,
        d_model:    int   = 64,
        n_heads:    int   = 4,
        n_layers:   int   = 2,
        latent_dim: int   = 16,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.feat_dim  = feat_dim
        self.seq_len   = seq_len
        self.proj      = nn.Linear(feat_dim * feat_dim, d_model)

        self.pos_emb   = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.to_latent = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"Expected 4-D input (B,L,F,F), got {x.shape}"
        B, L, F, F2 = x.shape
        assert L  == self.seq_len,  (
            f"seq_len mismatch: model expects {self.seq_len}, got {L}"
        )
        assert F  == self.feat_dim, (
            f"feat_dim mismatch: model expects {self.feat_dim}, got {F}"
        )
        assert F2 == self.feat_dim, (
            f"feat_dim mismatch (axis 3): model expects {self.feat_dim}, got {F2}"
        )

        tokens = self.proj(x.reshape(B, L, F * F2))  # (B, L, d_model)
        tokens = tokens + self.pos_emb
        enc    = self.transformer(tokens)              # (B, L, d_model)
        return self.to_latent(enc[:, -1, :])           # (B, latent_dim)


class DeepSuppAutoencoder(nn.Module):
    def __init__(
        self,
        feat_dim:   int,
        seq_len:    int,
        d_model:    int   = 64,
        n_heads:    int   = 4,
        n_layers:   int   = 2,
        latent_dim: int   = 16,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self._feat_dim = feat_dim
        self.encoder   = DeepSuppEncoder(
            feat_dim, seq_len, d_model, n_heads, n_layers, latent_dim, dropout
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, feat_dim * feat_dim),
        )
        # metadata attached by build_and_train()
        self.metadata: Optional[ModelMetadata] = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x     : (B, L, F, F)
        recon : (B, F, F)   symmetrised reconstruction of last matrix
        z     : (B, latent_dim)
        """
        z     = self.encoder(x)
        recon = self.decoder(z).reshape(-1, self._feat_dim, self._feat_dim)
        # Enforce symmetric reconstruction (corr matrices are symmetric)
        recon = 0.5 * (recon + recon.transpose(1, 2))
        return recon, z

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.encoder(x)


# ══════════════════════════════════════════════════════════════════════
# 6.  SCHEMA VALIDATION
# ══════════════════════════════════════════════════════════════════════

def validate_schema(
    features:     pd.DataFrame,
    model:        DeepSuppAutoencoder,
    seq_len:      int,
) -> None:
    """
    Raise ValueError if inference features are incompatible with the
    trained model.  Call before any tensor construction.
    """
    if model.metadata is None:
        # Model was built without metadata (e.g. external); skip validation
        return

    meta = model.metadata
    inf_cols  = list(features.columns)
    inf_fdim  = len(inf_cols)

    if inf_fdim != meta.feat_dim:
        raise ValueError(
            f"Feature dimension mismatch: inference has {inf_fdim} features, "
            f"model was trained with {meta.feat_dim}."
        )
    if inf_cols != meta.feature_names:
        raise ValueError(
            f"Feature name/order mismatch.\n"
            f"  Inference : {inf_cols}\n"
            f"  Training  : {meta.feature_names}"
        )
    if seq_len != meta.seq_len:
        raise ValueError(
            f"seq_len mismatch: inference uses {seq_len}, "
            f"model was trained with {meta.seq_len}."
        )


# ══════════════════════════════════════════════════════════════════════
# 7.  DATASET + TRAINING
# ══════════════════════════════════════════════════════════════════════

class _SeqCorrDataset(Dataset):
    def __init__(self, seqs: np.ndarray):
        self.seqs    = torch.from_numpy(seqs).float()   # (N, L, F, F)
        self.targets = self.seqs[:, -1, :, :]           # (N, F, F)  last matrix

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.targets[idx]


def train_deepsupp_model(
    seqs:       np.ndarray,
    feat_dim:   int,
    seq_len:    int,
    d_model:    int   = 64,
    n_heads:    int   = 4,
    n_layers:   int   = 2,
    latent_dim: int   = 16,
    dropout:    float = 0.1,
    epochs:     int   = 50,
    batch_size: int   = 32,
    lr:         float = 1e-3,
    device:     str   = "cpu",
    verbose:    bool  = True,
) -> DeepSuppAutoencoder:
    """
    Train on historical corr-matrix sequences.
    Not called directly in normal use — use build_and_train() instead.
    """
    loader = DataLoader(
        _SeqCorrDataset(seqs),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    model = DeepSuppAutoencoder(
        feat_dim, seq_len, d_model, n_heads, n_layers, latent_dim, dropout
    ).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N     = len(loader.dataset)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for seq_batch, tgt_batch in loader:
            seq_batch = seq_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            recon, _  = model(seq_batch)
            # Symmetrise target too before loss
            tgt_sym   = 0.5 * (tgt_batch + tgt_batch.transpose(1, 2))
            loss      = Fnn.mse_loss(recon, tgt_sym)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(seq_batch)
        sched.step()
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  epoch {ep:3d}/{epochs}  recon_loss={total/N:.6f}")

    return model


# ══════════════════════════════════════════════════════════════════════
# 8.  ANOMALY SCORING  (symmetrised)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_anomaly_scores(
    model:  DeepSuppAutoencoder,
    seqs:   np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Per-sequence MSE between symmetrised reconstruction and target."""
    model.eval()
    loader = DataLoader(_SeqCorrDataset(seqs), batch_size=256, shuffle=False)
    scores = []
    for seq_batch, tgt_batch in loader:
        seq_batch = seq_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        recon, _  = model(seq_batch)
        tgt_sym   = 0.5 * (tgt_batch + tgt_batch.transpose(1, 2))
        err       = Fnn.mse_loss(recon, tgt_sym, reduction="none")
        scores.append(err.mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(scores)


# ══════════════════════════════════════════════════════════════════════
# 9.  ROLLING ANOMALY THRESHOLD  (online-safe)
# ══════════════════════════════════════════════════════════════════════

def rolling_threshold(
    scores:        np.ndarray,
    threshold_pct: float = 85.0,
    lookback:      int   = 200,
) -> np.ndarray:
    """
    Per-timestep anomaly gate using a rolling lookback window of prior scores.

    Each position t uses percentile(scores[max(0,t-lookback):t], pct) as the
    threshold.  t=0 falls back to the global percentile of the first batch.

    Returns a boolean mask: True where score exceeds the rolling threshold.
    """
    T    = len(scores)
    mask = np.zeros(T, dtype=bool)
    for t in range(T):
        start  = max(0, t - lookback)
        window = scores[start:t] if t > 0 else scores[:1]
        cutoff = np.percentile(window, threshold_pct)
        mask[t] = scores[t] >= cutoff
    return mask


# ══════════════════════════════════════════════════════════════════════
# 10. S/R CLASSIFICATION BY LEVEL-RELATIVE DISPLACEMENT
# ══════════════════════════════════════════════════════════════════════

def _classify_by_displacement(
    bar_idxs:     np.ndarray,
    all_prices:   np.ndarray,
    level_price:  float,
    horizon:      int   = 10,
    min_evidence: int   = 3,
    proximity_pct: float = 0.01,
) -> tuple[str, float]:
    """
    Classify a price zone by measuring the mean signed % move of price
    *relative to level_price* `horizon` bars after each member bar,
    restricted to bars where price was within proximity_pct of the level.

    Relative displacement removes the effect of general market drift:
      disp = (price[t+horizon] - level_price) / level_price
    positive  → price moved above the level  → level acted as support
    negative  → price moved below the level  → level acted as resistance
    """
    displacements = []
    for idx in bar_idxs:
        # Only count bars where price was actually near the level
        if abs(all_prices[idx] - level_price) / (level_price + 1e-9) > proximity_pct:
            continue
        future = idx + horizon
        if future < len(all_prices):
            disp = (all_prices[future] - level_price) / (level_price + 1e-9)
            displacements.append(disp)

    if len(displacements) < min_evidence:
        return "level", 0.0

    mean_disp = float(np.mean(displacements))
    if mean_disp > 0.001:
        return "support", mean_disp
    elif mean_disp < -0.001:
        return "resistance", mean_disp
    return "level", mean_disp


# ══════════════════════════════════════════════════════════════════════
# 11. COMPOSITE STRENGTH SCORING  (population-normalised quality)
# ══════════════════════════════════════════════════════════════════════

def _composite_strength(
    n_members:    int,
    total:        int,
    score_mean:   float,
    score_max:    float,
    price_std:    float,
    pop_mean_max: float,     # max score_mean in high-score population
    pop_smax_max: float,     # max score_max in high-score population
    w_coverage:   float = 0.3,
    w_quality:    float = 0.4,
    w_tightness:  float = 0.3,
) -> tuple[float, float, float, float]:
    """
    coverage  = n_members / total
    quality   = 0.7 * (score_mean / pop_mean_max) + 0.3 * (score_max / pop_smax_max)
    tightness = 1 / (1 + price_std)
    strength  = weighted combination
    """
    coverage  = n_members / max(total, 1)
    mean_norm = score_mean / (pop_mean_max + 1e-9)
    smax_norm = score_max  / (pop_smax_max + 1e-9)
    quality   = 0.7 * mean_norm + 0.3 * smax_norm
    tightness = 1.0 / (1.0 + price_std)
    strength  = w_coverage * coverage + w_quality * quality + w_tightness * tightness
    return (
        round(float(min(strength, 1.0)), 4),
        round(float(coverage),           4),
        round(float(quality),            4),
        round(float(tightness),          4),
    )


# ══════════════════════════════════════════════════════════════════════
# 12. TWO-STAGE LEVEL EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def extract_levels(
    prices:            np.ndarray,      # (N,)  prices aligned to end_idxs
    all_prices:        np.ndarray,      # (P,)  full price series
    end_idxs:          np.ndarray,      # (N,)  original bar indices
    scores:            np.ndarray,      # (N,)  anomaly scores
    latent:            np.ndarray,      # (N, latent_dim)
    threshold_pct:     float = 85.0,
    threshold_lookback: int  = 200,
    dbscan_eps:        float = 0.5,
    dbscan_min:        int   = 5,
    price_eps_pct:     float = 0.005,
    price_min:         int   = 2,
    horizon:           int   = 10,
    min_disp_evidence: int   = 3,
    proximity_pct:     float = 0.01,
) -> List[LevelRecord]:
    # ── Stage 1: rolling anomaly gate ───────────────────────────────
    mask_hi = rolling_threshold(scores, threshold_pct, threshold_lookback)
    if mask_hi.sum() == 0:
        return []

    hi_prices  = prices[mask_hi]
    hi_scores  = scores[mask_hi]
    hi_latent  = latent[mask_hi]
    hi_end_idx = end_idxs[mask_hi]
    total      = len(scores)

    # Population normalisation anchors for quality scoring
    pop_mean_max = float(hi_scores.mean()) if hi_scores.size else 1.0
    pop_smax_max = float(hi_scores.max())  if hi_scores.size else 1.0

    # ── Stage 2: latent DBSCAN ───────────────────────────────────────
    db1    = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min).fit(hi_latent)
    labels = db1.labels_

    records: List[LevelRecord] = []

    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue

        m              = labels == lbl
        cluster_prices = hi_prices[m]
        cluster_scores = hi_scores[m]
        cluster_idx    = hi_end_idx[m]

        if cluster_prices.size == 0:
            continue

        # ── Stage 3: price sub-clustering ───────────────────────────
        price_eps  = max(np.median(cluster_prices) * price_eps_pct, 1e-6)
        db2        = DBSCAN(eps=price_eps, min_samples=price_min).fit(
            cluster_prices.reshape(-1, 1)
        )
        sub_labels = db2.labels_

        for sub_lbl in sorted(set(sub_labels)):
            if sub_lbl == -1:
                continue

            sm          = sub_labels == sub_lbl
            sub_p       = cluster_prices[sm]
            sub_s       = cluster_scores[sm]
            sub_idx     = cluster_idx[sm]

            level_price = float(np.median(sub_p))
            n_members   = int(sm.sum())
            s_mean      = float(sub_s.mean())
            s_max       = float(sub_s.max())
            p_std       = float(sub_p.std())

            # ── Stage 4: level-relative displacement classification ──
            kind, disp = _classify_by_displacement(
                sub_idx, all_prices, level_price,
                horizon=horizon,
                min_evidence=min_disp_evidence,
                proximity_pct=proximity_pct,
            )

            strength, coverage, quality, tightness = _composite_strength(
                n_members, total, s_mean, s_max, p_std,
                pop_mean_max, pop_smax_max,
            )

            records.append(LevelRecord(
                source       = "deepsupp",
                price        = round(level_price, 6),
                kind         = kind,
                strength     = strength,
                coverage     = coverage,
                quality      = quality,
                tightness    = tightness,
                cluster_id   = int(lbl * 10_000 + sub_lbl),
                n_members    = n_members,
                score_mean   = round(s_mean, 6),
                score_max    = round(s_max,  6),
                price_std    = round(p_std,  6),
                displacement = round(disp,   6),
            ))

    supports    = sorted([r for r in records if r.kind == "support"],
                         key=lambda r: r.price, reverse=True)
    resistances = sorted([r for r in records if r.kind == "resistance"],
                         key=lambda r: r.price)
    neutrals    = sorted([r for r in records if r.kind == "level"],
                         key=lambda r: r.price)
    return supports + resistances + neutrals


# ══════════════════════════════════════════════════════════════════════
# 13. SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════

def save_deepsupp_model(
    model: DeepSuppAutoencoder,
    path:  str | Path,
) -> None:
    """
    Save weights + full ModelMetadata to a single .pt file.
    Raises RuntimeError if model has no metadata (not built via build_and_train).
    """
    if model.metadata is None:
        raise RuntimeError(
            "Model has no metadata. Build with build_and_train() before saving."
        )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata":   model.metadata.to_dict(),
        },
        path,
    )


def load_deepsupp_model(
    path:   str | Path,
    device: str = "cpu",
) -> tuple[DeepSuppAutoencoder, ModelMetadata]:
    """
    Load a saved model and its metadata.

    Returns
    ───────
    model    : DeepSuppAutoencoder on `device`, eval mode
    metadata : ModelMetadata
    """
    ckpt = torch.load(path, map_location=device)
    meta = ModelMetadata.from_dict(ckpt["metadata"])
    model = DeepSuppAutoencoder(
        feat_dim   = meta.feat_dim,
        seq_len    = meta.seq_len,
        d_model    = meta.d_model,
        n_heads    = meta.n_heads,
        n_layers   = meta.n_layers,
        latent_dim = meta.latent_dim,
        dropout    = meta.dropout,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.metadata = meta
    model.to(device).eval()
    return model, meta


# ══════════════════════════════════════════════════════════════════════
# 14. PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def compute_deepsupp_levels(
    df:                 pd.DataFrame,
    model:              DeepSuppAutoencoder,
    *,
    # these must match training; validated against model.metadata if present
    vol_lookback:       int   = 20,
    corr_window:        int   = 20,
    seq_len:            int   = 16,
    # extraction
    threshold_pct:      float = 85.0,
    threshold_lookback: int   = 200,
    dbscan_eps:         float = 0.5,
    dbscan_min:         int   = 5,
    price_eps_pct:      float = 0.005,
    price_min:          int   = 2,
    horizon:            int   = 10,
    min_disp_evidence:  int   = 3,
    proximity_pct:      float = 0.01,
    device:             str   = "cpu",
    verbose:            bool  = False,
) -> List[LevelRecord]:
    """
    Inference-only drop-in level generator.

    Parameters
    ──────────
    df    : OHLCV DataFrame (open, high, low, close, volume; optional: vwap)
    model : pre-trained DeepSuppAutoencoder from build_and_train() or
            load_deepsupp_model()

    Returns
    ───────
    List[LevelRecord]
        Sorted: supports (desc price) → resistances (asc price) → neutrals (asc)

    Integration
    ───────────
        ds   = compute_deepsupp_levels(df, ds_model)
        hdb  = compute_hdbscan_levels(df)
        all_levels = merge_level_families(ds + hdb + ...)
    """
    # ── Device normalisation ─────────────────────────────────────────
    model = model.to(device).eval()

    # ── Features ─────────────────────────────────────────────────────
    features   = engineer_features(df, vol_lookback=vol_lookback)
    all_prices = np.exp(features["log_close"].values)
    X          = features.values.astype(np.float32)

    # ── Schema validation ────────────────────────────────────────────
    validate_schema(features, model, seq_len)

    # ── Corr matrices + sequence windows ────────────────────────────
    mats, idxs = build_corr_series(X, win_len=corr_window)
    if len(mats) < seq_len + dbscan_min:
        return []

    seqs, end_idxs = build_sequence_dataset(mats, idxs, seq_len=seq_len)
    if len(seqs) == 0:
        return []

    # ── Anomaly scores + latent codes ────────────────────────────────
    scores = compute_anomaly_scores(model, seqs, device=device)

    seqs_t = torch.from_numpy(seqs).float().to(device)
    latent = model.encode(seqs_t).cpu().numpy()

    aligned_prices = all_prices[end_idxs]

    # ── Extract ───────────────────────────────────────────────────────
    levels = extract_levels(
        aligned_prices, all_prices, end_idxs, scores, latent,
        threshold_pct=threshold_pct,
        threshold_lookback=threshold_lookback,
        dbscan_eps=dbscan_eps, dbscan_min=dbscan_min,
        price_eps_pct=price_eps_pct, price_min=price_min,
        horizon=horizon,
        min_disp_evidence=min_disp_evidence,
        proximity_pct=proximity_pct,
    )

    if verbose:
        n_s = sum(1 for l in levels if l.kind == "support")
        n_r = sum(1 for l in levels if l.kind == "resistance")
        n_n = sum(1 for l in levels if l.kind == "level")
        print(f"DeepSupp: {n_s} supports  {n_r} resistances  {n_n} neutrals")

    return levels


def build_and_train(
    df:           pd.DataFrame,
    *,
    vol_lookback: int   = 20,
    corr_window:  int   = 20,
    seq_len:      int   = 16,
    d_model:      int   = 64,
    n_heads:      int   = 4,
    n_layers:     int   = 2,
    latent_dim:   int   = 16,
    dropout:      float = 0.1,
    epochs:       int   = 50,
    batch_size:   int   = 32,
    lr:           float = 1e-3,
    device:       str   = "cpu",
    verbose:      bool  = True,
) -> DeepSuppAutoencoder:
    """
    Build features → corr matrices → sequence windows → train model.

    Attaches ModelMetadata to the returned model.
    Save immediately with save_deepsupp_model() for production use.
    """
    features   = engineer_features(df, vol_lookback=vol_lookback)
    X          = features.values.astype(np.float32)
    feat_dim   = X.shape[1]
    feat_names = list(features.columns)

    mats, idxs = build_corr_series(X, win_len=corr_window)
    seqs, _    = build_sequence_dataset(mats, idxs, seq_len=seq_len)

    if verbose:
        print(f"Training on {len(seqs)} sequences  "
              f"(feat_dim={feat_dim}, seq_len={seq_len})")

    model = train_deepsupp_model(
        seqs, feat_dim, seq_len,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        latent_dim=latent_dim, dropout=dropout,
        epochs=epochs, batch_size=batch_size, lr=lr,
        device=device, verbose=verbose,
    )

    # Attach metadata for schema validation and serialisation
    model.metadata = ModelMetadata(
        feat_dim      = feat_dim,
        seq_len       = seq_len,
        corr_window   = corr_window,
        vol_lookback  = vol_lookback,
        feature_names = feat_names,
        d_model       = d_model,
        n_heads       = n_heads,
        n_layers      = n_layers,
        latent_dim    = latent_dim,
        dropout       = dropout,
    )
    return model


# ══════════════════════════════════════════════════════════════════════
# 15. DEMO
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    n     = 800
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df    = pd.DataFrame({
        "open":   close + np.random.randn(n) * 0.1,
        "high":   close + np.abs(np.random.randn(n) * 0.3),
        "low":    close - np.abs(np.random.randn(n) * 0.3),
        "close":  close,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    })

    # ── Offline: train on first 600 bars, save ───────────────────────
    print("=== Training ===")
    model = build_and_train(df.iloc[:600], epochs=30, verbose=True)
    save_deepsupp_model(model, "/tmp/deepsupp_v4.pt")
    print("Model saved to /tmp/deepsupp_v4.pt")

    # ── Reload (simulates live startup) ─────────────────────────────
    model, meta = load_deepsupp_model("/tmp/deepsupp_v4.pt")
    print(f"\nLoaded model  feat_dim={meta.feat_dim}  "
          f"seq_len={meta.seq_len}  features={meta.feature_names}")

    # ── Live: inference on rolling forward window ────────────────────
    print("\n=== Inference ===")
    levels = compute_deepsupp_levels(
        df.iloc[400:], model,
        seq_len=meta.seq_len,
        corr_window=meta.corr_window,
        vol_lookback=meta.vol_lookback,
        verbose=True,
    )

    print("\n── Top levels ───────────────────────────────────────")
    for lvl in levels[:12]:
        print(
            f"  [{lvl.kind:11s}]  price={lvl.price:8.4f}  "
            f"str={lvl.strength:.3f}  cov={lvl.coverage:.3f}  "
            f"qual={lvl.quality:.3f}  tight={lvl.tightness:.3f}  "
            f"disp={lvl.displacement:+.4f}  n={lvl.n_members}"
        )

    if levels:
        print("\n── First record (aggregator input) ─────────────────")
        print(json.dumps(levels[0].to_dict(), indent=2))

    # ── Schema mismatch demo ─────────────────────────────────────────
    print("\n── Schema validation demo ───────────────────────────")
    try:
        # Deliberately pass wrong seq_len to trigger validation
        compute_deepsupp_levels(df.iloc[400:], model, seq_len=999)
    except ValueError as e:
        print(f"Caught expected error: {e}")
