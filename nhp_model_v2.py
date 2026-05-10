"""
model_v2.py — State-Dependent Neural Hawkes Process
=====================================================
Upgrade from nhp_model.py: the event embedding now accepts a continuous
state vector (realized vol, vol-of-vol, skew, regime probs) alongside
the event type and inter-arrival time.

Architecture follows Shi & Cartlidge (2022) State-Dependent Parallel NHP:
  [event_type, dt, state_vector] → embedding → LSTM → Q/K/V attention → softplus λ(t)

The state vector is concatenated at the embedding stage, not injected
into the intensity head, so the LSTM can learn regime-conditioned dynamics.
Everything else (log-likelihood, thinning, single forward pass) is unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

from nhp_model import NHPCell, LinearNormAttention, IntensityHead, NHPConfig


@dataclass
class NHPv2Config(NHPConfig):
    state_dim: int = 6      # [realized_vol, vol_of_vol, |skew|, P(LOW), P(NORMAL), P(HIGH)]
    state_hidden: int = 8   # projection of state vector before concat


class StateAwareEmbedding(nn.Module):
    """
    Embeds (event_type, dt, state_vector) → dense vector.

    State vector is projected separately then concatenated with the
    type+time embedding, following Shi & Cartlidge's design.
    """

    def __init__(self, cfg: NHPv2Config):
        super().__init__()
        self.type_emb   = nn.Embedding(cfg.num_event_types + 1, cfg.embed_dim, padding_idx=0)
        self.time_proj  = nn.Linear(2, cfg.embed_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.state_hidden),
            nn.Tanh(),
        )
        total = cfg.embed_dim * 2 + cfg.state_hidden
        self.out_proj = nn.Linear(total, cfg.embed_dim)

    def forward(
        self,
        event_type: torch.Tensor,   # (B, L)
        dt: torch.Tensor,           # (B, L)
        state: torch.Tensor,        # (B, L, state_dim)
    ) -> torch.Tensor:              # (B, L, embed_dim)
        t_feats = torch.stack([dt, torch.log(dt.clamp(min=1e-6))], dim=-1)
        t_enc   = self.time_proj(t_feats)               # (B, L, E)
        e_enc   = self.type_emb(event_type)             # (B, L, E)
        s_enc   = self.state_proj(state)                # (B, L, state_hidden)
        combined = torch.cat([e_enc, t_enc, s_enc], dim=-1)
        return self.out_proj(combined)


class StateDependentNHP(nn.Module):
    """
    Full state-dependent NHP.
    Drop-in replacement for NeuralHawkesProcess with an extra `states` input.
    """

    def __init__(self, cfg: NHPv2Config):
        super().__init__()
        self.cfg    = cfg
        self.embed  = StateAwareEmbedding(cfg)
        self.cell   = NHPCell(cfg)
        self.attn   = LinearNormAttention(cfg)
        self.head   = IntensityHead(cfg)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.1)

    def forward_sequence(
        self,
        event_types: torch.Tensor,  # (B, L)
        dts: torch.Tensor,          # (B, L)
        states: torch.Tensor,       # (B, L, state_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L = dts.shape
        H = self.cfg.hidden_dim
        h = torch.zeros(B, H, device=dts.device)
        c = torch.zeros(B, H, device=dts.device)
        embs = self.embed(event_types, dts, states)  # (B, L, E)
        hiddens, cells = [], []
        for t in range(L):
            h, c = self.cell(embs[:, t], h, c, dts[:, t])
            hiddens.append(h)
            cells.append(c)
        return torch.stack(hiddens, dim=1), torch.stack(cells, dim=1)

    def intensity_at(
        self,
        hiddens: torch.Tensor,
        cells: torch.Tensor,
        query_dts: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Identical to base NHP — attention over history, softplus output."""
        B, L, H = hiddens.shape
        causal_mask = torch.tril(
            torch.ones(L, L, dtype=torch.bool, device=hiddens.device)
        ).unsqueeze(0).expand(B, -1, -1)

        h_t = self.cell.extrapolate(
            hiddens.reshape(B * L, H),
            cells.reshape(B * L, H),
            query_dts.reshape(B * L),
        ).reshape(B, L, H)

        lambda_t = []
        for i in range(L):
            q = h_t[:, i]
            k = hiddens[:, :i + 1]
            mask = causal_mask[:, i, :i + 1]
            ctx, _ = self.attn(q, k, mask)
            lam = self.head(ctx)
            lambda_t.append(lam)
        return torch.stack(lambda_t, dim=1)

    def log_likelihood(
        self,
        event_types: torch.Tensor,
        dts: torch.Tensor,
        states: torch.Tensor,
        seq_lengths: torch.Tensor,
        n_mc: int = 20,
    ) -> torch.Tensor:
        B, L = dts.shape
        hiddens, cells = self.forward_sequence(event_types, dts, states)

        zero_dts = torch.zeros_like(dts)
        lam_events = self.intensity_at(hiddens, cells, zero_dts, seq_lengths)

        u = torch.rand(B, L, n_mc, device=dts.device)
        mc_dts = u * dts.unsqueeze(-1)
        mc_lams = []
        for s in range(n_mc):
            lam_s = self.intensity_at(hiddens, cells, mc_dts[:, :, s], seq_lengths)
            mc_lams.append(lam_s)
        mc_lams = torch.stack(mc_lams, dim=-1)
        integral = (mc_lams * dts.unsqueeze(-1)).mean(-1)

        mask = torch.arange(L, device=dts.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        nll_per = (torch.log(lam_events.clamp(min=1e-8)) - integral) * mask
        return nll_per.sum() / mask.sum().clamp(min=1)
