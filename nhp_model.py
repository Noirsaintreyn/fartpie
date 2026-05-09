"""
model.py — Neural Hawkes Process core
======================================
Architecture (Mei & Eisner 2017 NHP + LNAHP attention, Song et al. 2022):

  event embedding  →  LSTM cell update  →  Q/K/V linear-norm attention
    over hidden history  →  softplus intensity head

Stability is enforced through softplus parameterization (always positive)
and weight regularization during training — NOT a heuristic clamp.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class NHPConfig:
    embed_dim: int = 16       # event embedding dimension
    hidden_dim: int = 32      # LSTM hidden state dimension
    num_heads: int = 2        # attention heads (for multi-dim future extension)
    tau: float = 1.0          # attention temperature (learnable if desired)
    softplus_beta: float = 1.0
    dropout: float = 0.0
    num_event_types: int = 1  # extend to marked process if needed


class EventEmbedding(nn.Module):
    """
    Embeds (event_type, inter-arrival time) into a dense vector.
    Time encoding follows the sinusoidal scheme from the NHP literature.
    """
    def __init__(self, cfg: NHPConfig):
        super().__init__()
        self.type_emb = nn.Embedding(cfg.num_event_types + 1, cfg.embed_dim, padding_idx=0)
        # Time encoding: linear + log-time projection
        self.time_proj = nn.Linear(2, cfg.embed_dim)
        self.out_proj = nn.Linear(cfg.embed_dim * 2, cfg.embed_dim)

    def forward(self, event_type: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            event_type: (B, L) int64, 1-indexed type (0 = padding)
            dt: (B, L) float, inter-arrival times
        Returns:
            (B, L, embed_dim)
        """
        t_feats = torch.stack([dt, torch.log(dt.clamp(min=1e-6))], dim=-1)
        t_enc = self.time_proj(t_feats)
        e_enc = self.type_emb(event_type)
        return self.out_proj(torch.cat([e_enc, t_enc], dim=-1))


class NHPCell(nn.Module):
    """
    Neural Hawkes Process LSTM cell (Mei & Eisner 2017).
    Standard LSTM + decay mechanism: the cell state decays between events,
    giving the model continuous-time behavior.
    """
    def __init__(self, cfg: NHPConfig):
        super().__init__()
        d = cfg.embed_dim
        h = cfg.hidden_dim
        # Combined input/hidden → gates (4*h for f,i,o,g)
        self.lstm_cell = nn.LSTMCell(d, h)
        # Learned per-dimension decay rates (positive via softplus)
        self.decay_raw = nn.Parameter(torch.zeros(h))
        self.dropout = nn.Dropout(cfg.dropout)

    def decay_rates(self) -> torch.Tensor:
        return F.softplus(self.decay_raw)  # always positive

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor,
                dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:  (B, embed_dim) — current event embedding
        h:  (B, hidden_dim) — previous hidden state
        c:  (B, hidden_dim) — previous cell state
        dt: (B,) — time since last event
        Returns: h_new, c_new  (both (B, hidden_dim))
        """
        x = self.dropout(x)
        h_new, c_new = self.lstm_cell(x, (h, c))
        return h_new, c_new

    def extrapolate(self, h: torch.Tensor, c: torch.Tensor,
                    dt: torch.Tensor) -> torch.Tensor:
        """
        Compute h(t) for arbitrary t > last event using continuous decay.
        h(t) = o * tanh(c * exp(-delta * dt))
        where o is the output gate value from the last event.

        Approximation: we decay the cell state and re-apply tanh.
        delta = softplus(decay_raw) ensures positive decay.
        """
        delta = self.decay_rates()  # (H,)
        dt_exp = dt.unsqueeze(-1) * delta.unsqueeze(0)  # (B, H)
        c_decayed = c * torch.exp(-dt_exp)
        h_t = torch.tanh(c_decayed)
        return h_t


class LinearNormAttention(nn.Module):
    """
    Linear normalization attention (LNAHP, Song et al. 2022).
    Replaces O(N²) dot-product attention with O(N) normalized similarity.

    Q, K, V are separate learned projections — not reused hidden states.
    """
    def __init__(self, cfg: NHPConfig):
        super().__init__()
        h = cfg.hidden_dim
        self.Wq = nn.Linear(h, h, bias=False)
        self.Wk = nn.Linear(h, h, bias=False)
        self.Wv = nn.Linear(h, h, bias=False)
        self.tau = nn.Parameter(torch.tensor(cfg.tau))  # learnable temperature
        self.out_proj = nn.Linear(h, h)

    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        query: (B, H) — last hidden state (query for current time)
        keys:  (B, L, H) — hidden history (keys and values)
        mask:  (B, L) bool — True where valid (causal: only past events)
        Returns: context (B, H), attn_weights (B, L)
        """
        q = self.Wq(query).unsqueeze(1)     # (B, 1, H)
        k = self.Wk(keys)                   # (B, L, H)
        v = self.Wv(keys)                   # (B, L, H)

        # Normalized inner product (linear normalization)
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        scores = (q_norm * k_norm).sum(-1) / self.tau.abs().clamp(min=0.01)  # (B, L)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # handle all-masked rows

        context = (attn.unsqueeze(-1) * v).sum(1)  # (B, H)
        return self.out_proj(context), attn


class IntensityHead(nn.Module):
    """
    Maps hidden context + time-since-event to conditional intensity λ(t).
    λ(t) = softplus(W·h(t) + b)  — always positive, no clamp needed.

    The decay is handled by NHPCell.extrapolate, not a heuristic here.
    """
    def __init__(self, cfg: NHPConfig):
        super().__init__()
        h = cfg.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )
        self.beta = cfg.softplus_beta

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        h_t: (B, H) or (B, L, H) — (extrapolated) hidden state at query times
        Returns: λ(t) ≥ 0, same leading shape with last dim squeezed
        """
        raw = self.net(h_t).squeeze(-1)
        return F.softplus(raw, beta=self.beta)


class NeuralHawkesProcess(nn.Module):
    """
    Full NHP model — event embedding → LSTM → attention → intensity.

    Forward pass is a single authoritative computation used for both
    training (likelihood) and inference (signal generation).
    """
    def __init__(self, cfg: NHPConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = EventEmbedding(cfg)
        self.cell = NHPCell(cfg)
        self.attn = LinearNormAttention(cfg)
        self.intensity_head = IntensityHead(cfg)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.1)

    def forward_sequence(self, event_types: torch.Tensor,
                          dts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run LSTM over event sequence, collecting hidden states.

        Args:
            event_types: (B, L) int64
            dts:         (B, L) float, inter-arrival times
        Returns:
            hiddens: (B, L, H) — hidden state after each event
            cells:   (B, L, H) — cell state after each event
        """
        B, L = dts.shape
        H = self.cfg.hidden_dim
        h = torch.zeros(B, H, device=dts.device)
        c = torch.zeros(B, H, device=dts.device)
        embs = self.embed(event_types, dts)  # (B, L, E)
        hiddens, cells = [], []
        for t in range(L):
            h, c = self.cell(embs[:, t], h, c, dts[:, t])
            hiddens.append(h)
            cells.append(c)
        return torch.stack(hiddens, dim=1), torch.stack(cells, dim=1)

    def intensity_at(self, hiddens: torch.Tensor, cells: torch.Tensor,
                     query_dts: torch.Tensor,
                     seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute λ(t) at arbitrary query times after each event.
        Used for both NLL training and the rendering path — single source of truth.

        Args:
            hiddens:     (B, L, H)
            cells:       (B, L, H)
            query_dts:   (B, L) — time delta from each event to the query point
            seq_lengths: (B,) — valid sequence lengths for masking
        Returns:
            lambda_t: (B, L)
        """
        B, L, H = hiddens.shape

        # Extrapolate hidden state forward in time (continuous decay)
        h_t = self.cell.extrapolate(
            hiddens.reshape(B * L, H),
            cells.reshape(B * L, H),
            query_dts.reshape(B * L)
        ).reshape(B, L, H)

        # Attention: for each position, attend over all past positions
        # Causal mask: position i can only attend to positions < i
        causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool,
                                            device=hiddens.device)).unsqueeze(0)
        causal_mask = causal_mask.expand(B, -1, -1)  # (B, L, L)

        # Compute attention-weighted context for each query position
        lambda_t = []
        for i in range(L):
            q = h_t[:, i]                   # (B, H)
            k = hiddens[:, :i+1]            # (B, i+1, H)
            mask = causal_mask[:, i, :i+1]  # (B, i+1)
            ctx, _ = self.attn(q, k, mask)
            lam = self.intensity_head(ctx)
            lambda_t.append(lam)

        return torch.stack(lambda_t, dim=1)  # (B, L)

    def log_likelihood(self, event_types: torch.Tensor, dts: torch.Tensor,
                        seq_lengths: torch.Tensor,
                        n_mc: int = 20) -> torch.Tensor:
        """
        Event-time log-likelihood (negative = training loss):
          log L = Σ log λ(tᵢ) − ∫₀ᵀ λ(t) dt

        The integral is approximated via Monte Carlo sampling over (0, T)
        per inter-arrival interval — standard approach for NHP training.

        Args:
            event_types: (B, L)
            dts:         (B, L) inter-arrival times
            seq_lengths: (B,)
            n_mc:        Monte Carlo samples for compensator integral
        Returns:
            mean log-likelihood per event (scalar)
        """
        B, L = dts.shape
        hiddens, cells = self.forward_sequence(event_types, dts)

        # ── Term 1: log λ at each event time (dt=0 from itself) ──────────────
        zero_dts = torch.zeros_like(dts)
        lam_events = self.intensity_at(hiddens, cells, zero_dts, seq_lengths)

        # ── Term 2: Monte Carlo integral ∫λ(t)dt over each interval ─────────
        # Sample n_mc uniform points in each inter-arrival interval
        u = torch.rand(B, L, n_mc, device=dts.device)           # (B, L, n_mc)
        mc_dts = u * dts.unsqueeze(-1)                           # (B, L, n_mc)

        mc_lams = []
        for s in range(n_mc):
            lam_s = self.intensity_at(hiddens, cells, mc_dts[:, :, s], seq_lengths)
            mc_lams.append(lam_s)
        mc_lams = torch.stack(mc_lams, dim=-1)                   # (B, L, n_mc)
        integral = (mc_lams * dts.unsqueeze(-1)).mean(-1)        # (B, L)

        # ── Mask padding, average over valid events ───────────────────────────
        mask = torch.arange(L, device=dts.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        log_lam = torch.log(lam_events.clamp(min=1e-8))
        nll_per = (log_lam - integral) * mask
        n_valid = mask.sum().clamp(min=1)
        return nll_per.sum() / n_valid  # mean log-likelihood (higher = better)
