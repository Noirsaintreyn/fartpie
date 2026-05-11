"""
model.py — Neural Hawkes Process (v3, bugs fixed)
==================================================
Chain of bugs found and fixed:

BUG 1 — extrapolate() used wrong formula.
  Was:    h(t) = tanh(c * exp(-δ·Δt))
  Should: h(t) = o ⊙ tanh(c_bar + (c - c_bar) * exp(-δ·Δt))
  where c_bar is a learned target cell state (Mei & Eisner 2017, eq.10).
  Without c_bar, decay pulls c toward 0, so h(t)→0 as Δt grows.
  All queries in intensity_at used Δt=0, but the formula still compressed
  everything through tanh on near-zero values → near-constant output.

BUG 2 — IntensityHead had Tanh between two Linear layers.
  softplus(Linear(Tanh(Linear(h)))) — the inner Tanh clipped all variation
  to [-1,1] before the final projection. Replaced with LayerNorm+SiLU.

BUG 3 — LinearNormAttention F.normalize() stripped magnitude.
  Clustering strength is encoded in ||h||. Normalizing before attention
  made sparse and cluster periods produce identical context vectors.
  Fix: scale context by ||query|| after attention.

BUG 4 — xavier_uniform gain=0.5 kept all activations near zero.
  Fix: gain=1.0 on hidden layers, 0.1 only on final output projections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class NHPConfig:
    embed_dim: int = 16
    hidden_dim: int = 32
    num_heads: int = 2
    tau: float = 1.0
    softplus_beta: float = 1.0
    dropout: float = 0.0
    num_event_types: int = 1


class EventEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.type_emb  = nn.Embedding(cfg.num_event_types + 1, cfg.embed_dim, padding_idx=0)
        self.time_proj = nn.Linear(2, cfg.embed_dim)
        self.out_proj  = nn.Linear(cfg.embed_dim * 2, cfg.embed_dim)

    def forward(self, event_type, dt):
        t_feats = torch.stack([dt, torch.log(dt.clamp(min=1e-6))], dim=-1)
        return self.out_proj(torch.cat([self.type_emb(event_type), self.time_proj(t_feats)], dim=-1))


class NHPCell(nn.Module):
    """
    Neural Hawkes LSTM cell with Mei & Eisner (2017) continuous-time decay.
    Each event produces (c, c_bar, o) where:
      c      = current cell state (jumps at event)
      c_bar  = target cell state (cell decays toward this between events)
      o      = output gate
    Then: h(t) = o ⊙ tanh(c_bar + (c - c_bar) * exp(-δ·Δt))
    """
    def __init__(self, cfg):
        super().__init__()
        d = cfg.embed_dim
        h = cfg.hidden_dim
        # Standard LSTM gates
        self.lstm_cell  = nn.LSTMCell(d, h)
        # Target cell state projection (c_bar in Mei & Eisner)
        self.W_cbar     = nn.Linear(h + d, h)
        # Output gate (used in extrapolation)
        self.W_o        = nn.Linear(h + d, h)
        # Per-dimension decay rates
        self.decay_raw  = nn.Parameter(torch.ones(h) * 0.5)
        self.dropout    = nn.Dropout(cfg.dropout)

        # Store last output gate and target for extrapolation
        self._last_o    = None
        self._last_cbar = None

    def decay_rates(self):
        return F.softplus(self.decay_raw) + 1e-4  # always positive, bounded below

    def forward(self, x, h, c, dt):
        """Update LSTM state at an event. Returns h_new, c_new."""
        x = self.dropout(x)
        h_new, c_new = self.lstm_cell(x, (h, c))
        # Compute target cell state and output gate
        hx = torch.cat([h_new, x], dim=-1)
        c_bar = torch.tanh(self.W_cbar(hx))
        o     = torch.sigmoid(self.W_o(hx))
        # Cache for extrapolation
        self._last_cbar = c_bar
        self._last_o    = o
        return h_new, c_new

    def extrapolate(self, h, c, dt, c_bar=None, o=None):
        """
        Compute h(t) for arbitrary t > last event.
        h(t) = o ⊙ tanh(c_bar + (c - c_bar) * exp(-δ·Δt))
        
        If c_bar/o not provided, falls back to simpler decay (backwards compat).
        """
        delta = self.decay_rates()  # (H,)
        decay = torch.exp(-dt.unsqueeze(-1) * delta.unsqueeze(0))  # (B, H)

        if c_bar is not None and o is not None:
            c_t = c_bar + (c - c_bar) * decay
            return o * torch.tanh(c_t)
        else:
            # Fallback: simple decay toward zero
            return torch.tanh(c * decay)


class LinearNormAttention(nn.Module):
    """
    Fixed: context scaled by ||q|| to preserve amplitude information.
    """
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim
        self.Wq       = nn.Linear(h, h, bias=False)
        self.Wk       = nn.Linear(h, h, bias=False)
        self.Wv       = nn.Linear(h, h, bias=False)
        self.tau      = nn.Parameter(torch.tensor(cfg.tau))
        self.out_proj = nn.Linear(h, h)

    def forward(self, query, keys, mask=None):
        q     = self.Wq(query).unsqueeze(1)  # (B,1,H)
        k     = self.Wk(keys)                # (B,L,H)
        v     = self.Wv(keys)                # (B,L,H)

        # Save magnitude before normalization
        q_mag  = query.norm(dim=-1, keepdim=True)  # (B,1)

        scores = (F.normalize(q, dim=-1) * F.normalize(k, dim=-1)).sum(-1) / \
                 self.tau.abs().clamp(min=0.01)     # (B,L)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attn    = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        context = (attn.unsqueeze(-1) * v).sum(1)  # (B,H)

        # Restore amplitude: clustering strength lives in ||q||
        context = context * q_mag

        return self.out_proj(context), attn


class IntensityHead(nn.Module):
    """
    Fixed: LayerNorm+SiLU instead of Tanh between linear layers.
    """
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )
        self.beta = cfg.softplus_beta

    def forward(self, h_t):
        return F.softplus(self.net(h_t).squeeze(-1), beta=self.beta)


class NeuralHawkesProcess(nn.Module):
    def __init__(self, cfg: NHPConfig):
        super().__init__()
        self.cfg            = cfg
        self.embed          = EventEmbedding(cfg)
        self.cell           = NHPCell(cfg)
        self.attn           = LinearNormAttention(cfg)
        self.intensity_head = IntensityHead(cfg)
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                gain = 0.1 if any(x in name for x in ['out_proj', 'net.3']) else 1.0
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.1)

    def forward_sequence(self, event_types, dts, states=None):
        B, L = dts.shape
        H    = self.cfg.hidden_dim
        h    = torch.zeros(B, H, device=dts.device)
        c    = torch.zeros(B, H, device=dts.device)
        embs = self.embed(event_types, dts)
        hiddens, cells, cbars, outputs = [], [], [], []
        for t in range(L):
            h, c = self.cell(embs[:, t], h, c, dts[:, t])
            hiddens.append(h)
            cells.append(c)
            cbars.append(self.cell._last_cbar.clone() if self.cell._last_cbar is not None else torch.zeros_like(c))
            outputs.append(self.cell._last_o.clone()  if self.cell._last_o  is not None else torch.ones_like(h))
        return (torch.stack(hiddens, dim=1), torch.stack(cells, dim=1),
                torch.stack(cbars,   dim=1), torch.stack(outputs, dim=1))

    def intensity_at(self, hiddens, cells, query_dts, seq_lengths, cbars=None, outputs=None):
        B, L, H = hiddens.shape
        causal  = torch.tril(torch.ones(L, L, dtype=torch.bool, device=hiddens.device)).unsqueeze(0).expand(B,-1,-1)

        # Extrapolate with proper Mei & Eisner formula if c_bar and o available
        if cbars is not None and outputs is not None:
            h_t = self.cell.extrapolate(
                hiddens.reshape(B*L, H), cells.reshape(B*L, H),
                query_dts.reshape(B*L),
                cbars.reshape(B*L, H), outputs.reshape(B*L, H),
            ).reshape(B, L, H)
        else:
            h_t = self.cell.extrapolate(
                hiddens.reshape(B*L, H), cells.reshape(B*L, H),
                query_dts.reshape(B*L),
            ).reshape(B, L, H)

        lams = []
        for i in range(L):
            ctx, _ = self.attn(h_t[:,i], hiddens[:,:i+1], causal[:,i,:i+1])
            lams.append(self.intensity_head(ctx))
        return torch.stack(lams, dim=1)

    def log_likelihood(self, event_types, dts, seq_lengths, n_mc=20, states=None):
        B, L = dts.shape
        hiddens, cells, cbars, outs = self.forward_sequence(event_types, dts, states)
        lam_ev   = self.intensity_at(hiddens, cells, torch.zeros_like(dts), seq_lengths, cbars, outs)
        u        = torch.rand(B, L, n_mc, device=dts.device)
        mc_lams  = torch.stack([
            self.intensity_at(hiddens, cells, u[:,:,s]*dts, seq_lengths, cbars, outs)
            for s in range(n_mc)
        ], -1)
        integral = (mc_lams * dts.unsqueeze(-1)).mean(-1)
        mask     = torch.arange(L, device=dts.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        return ((torch.log(lam_ev.clamp(min=1e-8)) - integral) * mask).sum() / mask.sum().clamp(min=1)
