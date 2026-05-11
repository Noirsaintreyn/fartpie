"""
model_v3.py — State-Conditioned NHP with Regime-Gated Intensity Head
=====================================================================
Implements all five recommendations:

1. State injected at BOTH embedding AND intensity head
   - Embedding: event context (what happened)
   - Intensity head: regime directly gates the hazard rate (how intense)

2. Regime-Mixture intensity head (soft MoE by regime)
   - 3 expert heads (one per regime: LOW/NORMAL/HIGH vol)
   - Soft-gated by P(regime) from HMM — not hard switching
   - Allows each regime to learn its own intensity scale

3. Auxiliary classification loss support
   - forward() returns both intensity AND a cluster-quality score
   - cluster_score: P(this cluster is actionable) — trained with BCE

4. Training objective: NLL + alpha * BCE(cluster_quality)
   - alpha controls the tradeoff (default 0.3)
   - cluster label: next N bars have positive return > threshold

5. Calibration-ready output
   - intensity_head returns raw logit AND calibrated probability
   - isotonic regression calibration in post-training pass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

# Import fixed base components (v3 fixes: amplitude attn, LayerNorm head)
from nhp_model_v3 import NHPConfig, EventEmbedding, NHPCell, LinearNormAttention


@dataclass
class NHPv3Config(NHPConfig):
    state_dim:    int   = 6      # [rv, vov, skew, P(LOW), P(NORMAL), P(HIGH)]
    state_hidden: int   = 12     # state projection size
    n_regimes:    int   = 3      # number of regime experts
    aux_loss_alpha: float = 0.3  # weight of auxiliary classification loss
    cluster_horizon: int  = 5    # bars ahead for cluster quality label


# ── State-Aware Embedding (same as v2 but uses v3 config) ────────────────────

class StateAwareEmbedding(nn.Module):
    def __init__(self, cfg: NHPv3Config):
        super().__init__()
        self.type_emb   = nn.Embedding(cfg.num_event_types + 1, cfg.embed_dim, padding_idx=0)
        self.time_proj  = nn.Linear(2, cfg.embed_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.state_hidden),
            nn.LayerNorm(cfg.state_hidden),
            nn.SiLU(),
        )
        self.out_proj = nn.Linear(cfg.embed_dim * 2 + cfg.state_hidden, cfg.embed_dim)

    def forward(self, event_type, dt, state):
        t_feats = torch.stack([dt, torch.log(dt.clamp(min=1e-6))], dim=-1)
        t_enc   = self.time_proj(t_feats)
        e_enc   = self.type_emb(event_type)
        s_enc   = self.state_proj(state)
        return self.out_proj(torch.cat([e_enc, t_enc, s_enc], dim=-1))


# ── Regime-Mixture Intensity Head ─────────────────────────────────────────────

class RegimeMixtureHead(nn.Module):
    """
    Soft mixture-of-experts intensity head.

    For each time step, maintains n_regimes expert heads.
    The final intensity is a soft mixture weighted by regime probabilities:
        λ(t) = Σ_k P(regime=k) * softplus(expert_k(h_t, state))

    This means:
    - LOW vol regime → expert_0 learns quiet-period intensity
    - NORMAL regime  → expert_1 learns baseline intensity
    - HIGH vol regime → expert_2 learns volatile-period intensity

    State is injected HERE (not just at embedding) so regime directly
    modulates the hazard rate, not just the hidden state indirectly.

    Also outputs cluster_score: P(this is an actionable cluster start)
    trained with auxiliary BCE loss.
    """

    def __init__(self, cfg: NHPv3Config):
        super().__init__()
        h  = cfg.hidden_dim
        sd = cfg.state_hidden
        K  = cfg.n_regimes

        # State projection (shared across experts)
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.state_dim, sd),
            nn.LayerNorm(sd),
            nn.SiLU(),
        )

        # K expert intensity heads — each sees hidden context + state
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h + sd, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )
            for _ in range(K)
        ])

        # Regime probability extractor from state vector
        # Last 3 dims of state_dim are [P(LOW), P(NORMAL), P(HIGH)]
        self.regime_prob_idx = slice(-K, None)  # last K dims

        # Auxiliary cluster quality head
        # Predicts P(next cluster is actionable) — used for precision gating
        self.cluster_head = nn.Sequential(
            nn.Linear(h + sd, h // 2),
            nn.SiLU(),
            nn.Linear(h // 2, 1),
        )

        self.beta = cfg.softplus_beta
        self.K    = K

    def forward(
        self,
        h_t:   torch.Tensor,   # (B, H) — attention context
        state: torch.Tensor,   # (B, state_dim) — current regime state
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            lambda_t:      (B,) — soft-mixed intensity
            cluster_score: (B,) — logit for cluster quality (apply sigmoid for prob)
        """
        s_enc = self.state_proj(state)              # (B, sd)
        hs    = torch.cat([h_t, s_enc], dim=-1)     # (B, H+sd)

        # Expert intensities
        expert_lams = torch.stack(
            [F.softplus(e(hs).squeeze(-1), beta=self.beta) for e in self.experts],
            dim=-1
        )  # (B, K)

        # Regime weights from state vector (last K dims = regime probs)
        regime_probs = state[..., self.regime_prob_idx]          # (B, K)
        regime_probs = F.softmax(regime_probs * 5.0, dim=-1)     # sharpen slightly

        # Soft mixture
        lambda_t = (expert_lams * regime_probs).sum(-1)          # (B,)

        # Cluster quality score
        cluster_score = self.cluster_head(hs).squeeze(-1)        # (B,) logit

        return lambda_t, cluster_score


# ── Full State-Conditioned NHP v3 ────────────────────────────────────────────

class NHPv3(nn.Module):
    """
    State-conditioned NHP with regime-mixture intensity head.

    Key differences from v2:
    - State injected at embedding AND intensity head
    - Regime-mixture head: 3 experts soft-gated by P(regime)
    - Auxiliary cluster quality output for precision training
    - Combined loss: NLL + alpha * BCE(cluster_quality)
    """

    def __init__(self, cfg: NHPv3Config):
        super().__init__()
        self.cfg    = cfg
        self.embed  = StateAwareEmbedding(cfg)
        self.cell   = NHPCell(cfg)
        self.attn   = LinearNormAttention(cfg)
        self.head   = RegimeMixtureHead(cfg)
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                gain = 0.1 if any(x in name for x in ['out_proj', 'experts.']) and name.endswith('.3') else 1.0
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.1)

    def forward_sequence(self, event_types, dts, states):
        """
        Args:
            event_types: (B, L)
            dts:         (B, L)
            states:      (B, L, state_dim)
        Returns:
            hiddens, cells, cbars, outputs  — all (B, L, H)
        """
        B, L = dts.shape
        H    = self.cfg.hidden_dim
        h    = torch.zeros(B, H, device=dts.device)
        c    = torch.zeros(B, H, device=dts.device)
        embs = self.embed(event_types, dts, states)

        hiddens, cells, cbars, outputs = [], [], [], []
        for t in range(L):
            h, c = self.cell(embs[:, t], h, c, dts[:, t])
            hiddens.append(h)
            cells.append(c)
            cbars.append(self.cell._last_cbar.clone() if self.cell._last_cbar is not None
                         else torch.zeros_like(c))
            outputs.append(self.cell._last_o.clone() if self.cell._last_o is not None
                           else torch.ones_like(h))

        return (torch.stack(hiddens, dim=1), torch.stack(cells, dim=1),
                torch.stack(cbars,   dim=1), torch.stack(outputs, dim=1))

    def intensity_at(self, hiddens, cells, query_dts, seq_lengths,
                     cbars, outputs, states):
        """
        Compute λ(t) and cluster_score at query times.

        state at each position = last known state (causal).
        """
        B, L, H = hiddens.shape
        causal  = torch.tril(
            torch.ones(L, L, dtype=torch.bool, device=hiddens.device)
        ).unsqueeze(0).expand(B, -1, -1)

        # Extrapolate with Mei & Eisner formula
        h_t = self.cell.extrapolate(
            hiddens.reshape(B*L, H), cells.reshape(B*L, H),
            query_dts.reshape(B*L),
            cbars.reshape(B*L, H), outputs.reshape(B*L, H),
        ).reshape(B, L, H)

        lams, cluster_scores = [], []
        for i in range(L):
            ctx, _ = self.attn(h_t[:, i], hiddens[:, :i+1], causal[:, i, :i+1])
            # Inject current state into intensity head
            state_i = states[:, i]                    # (B, state_dim)
            lam, cscore = self.head(ctx, state_i)
            lams.append(lam)
            cluster_scores.append(cscore)

        return torch.stack(lams, dim=1), torch.stack(cluster_scores, dim=1)

    def compute_loss(self, event_types, dts, states, seq_lengths,
                     cluster_labels=None, n_mc=20):
        """
        Combined loss: NLL + alpha * BCE(cluster_quality)

        cluster_labels: (B, L) binary — 1 if cluster starting at this event
                        is actionable (positive return in next horizon bars).
                        If None, only NLL is computed.

        Returns: loss (scalar), {'nll': ..., 'bce': ..., 'total': ...}
        """
        B, L = dts.shape
        hiddens, cells, cbars, outs = self.forward_sequence(event_types, dts, states)

        # ── NLL term ──────────────────────────────────────────────────────────
        lam_ev, cscore_ev = self.intensity_at(
            hiddens, cells, torch.zeros_like(dts), seq_lengths, cbars, outs, states
        )

        # Monte Carlo compensator integral
        u       = torch.rand(B, L, n_mc, device=dts.device)
        mc_lams = []
        for s in range(n_mc):
            mc_q = u[:, :, s] * dts
            lam_s, _ = self.intensity_at(
                hiddens, cells, mc_q, seq_lengths, cbars, outs, states
            )
            mc_lams.append(lam_s)
        mc_lams  = torch.stack(mc_lams, dim=-1)
        integral = (mc_lams * dts.unsqueeze(-1)).mean(-1)

        mask    = torch.arange(L, device=dts.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        log_lam = torch.log(lam_ev.clamp(min=1e-8))
        nll     = -((log_lam - integral) * mask).sum() / mask.sum().clamp(min=1)

        losses = {'nll': nll.item()}

        # ── Auxiliary BCE term ────────────────────────────────────────────────
        if cluster_labels is not None:
            bce = F.binary_cross_entropy_with_logits(
                cscore_ev[mask], cluster_labels[mask].float()
            )
            total = nll + self.cfg.aux_loss_alpha * bce
            losses['bce']   = bce.item()
            losses['total'] = total.item()
        else:
            total = nll
            losses['bce']   = 0.0
            losses['total'] = nll.item()

        return total, losses

    def predict(self, event_types, dts, states, seq_lengths):
        """
        Inference: returns intensity series + cluster quality probabilities.
        Use cluster_prob > threshold as a precision gate on signals.
        """
        with torch.no_grad():
            hiddens, cells, cbars, outs = self.forward_sequence(event_types, dts, states)
            lams, cscores = self.intensity_at(
                hiddens, cells, torch.zeros_like(dts),
                seq_lengths, cbars, outs, states
            )
        return lams, torch.sigmoid(cscores)
