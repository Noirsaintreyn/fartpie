"""
DeepSupp (arXiv:2507.01971): attention-driven correlation-pattern support/
resistance detection.

Pipeline, per the paper:
  1. Feature engineering: Close, VWAP, Volume, PriceChangeVolume, VolumeRatio
     (volume-weighted price action + market microstructure indicators).
  2. Rolling Spearman-rank correlation over sliding windows of length 32,
     producing a sequence of 32x32 correlation "snapshots".
     (The paper is explicit the matrix is 32x32, but with only 5 features
     per bar, a feature-feature correlation matrix can't be 32x32. The only
     construction consistent with their stated dimensionality is a
     TIME-POINT x TIME-POINT correlation: each of the 32 bars in the window
     is treated as an observation described by its 5-feature vector, and the
     32x32 matrix captures how similar each pair of bars is across those 5
     features. That's what's implemented below.)
  3. Multi-head attention autoencoder (4 heads, embed dim 32) compresses each
     32x32 snapshot to a 16-dim embedding, trained by reconstruction loss.
  4. DBSCAN on the embedding space (eps=0.1, min_samples=10% of n) extracts
     clusters; the median price of the bars in each cluster is a level.

Unlike HDBSCAN/OPTICS/GMM/etc., step 3 needs a persistently trained model,
so this is used differently from the other detectors in this project: train
once on an early, chronological slice of history, freeze the weights, then
run inference-only for level extraction on later data. See
backtest_deepsupp.py for the train/holdout split that keeps this PIT-safe.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

WINDOW = 32
EMBED_DIM = 32
LATENT_DIM = 16
N_HEADS = 4


def compute_features(closes, volumes):
    """[Close, VWAP, Volume, PriceChangeVolume, VolumeRatio] per bar,
    MinMax-scaled to [0,1] over the given series (matches the paper's
    preprocessing step). Uses a 20-bar rolling VWAP/volume-ratio window since
    continuous futures have no clean intraday session boundary to anchor a
    cumulative VWAP to."""
    closes = np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    n = len(closes)

    vwap = np.zeros(n)
    vol_ratio = np.zeros(n)
    price_change_vol = np.zeros(n)
    roll_w = 20
    for i in range(n):
        lo = max(0, i - roll_w + 1)
        w_c = closes[lo:i + 1]
        w_v = volumes[lo:i + 1]
        vwap[i] = np.average(w_c, weights=w_v) if w_v.sum() > 0 else w_c.mean()
        vol_ratio[i] = volumes[i] / w_v.mean() if w_v.mean() > 0 else 1.0
        if i > 0 and closes[i - 1] > 0:
            price_change_vol[i] = (closes[i] - closes[i - 1]) / closes[i - 1] * volumes[i]

    feats = np.stack([closes, vwap, volumes, price_change_vol, vol_ratio], axis=1)
    mins = feats.min(axis=0)
    maxs = feats.max(axis=0)
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
    return (feats - mins) / ranges


def spearman_corr_matrix(feature_window):
    """feature_window: [WINDOW, 5] -> [WINDOW, WINDOW] Spearman correlation
    between time points (each time point described by its 5 features)."""
    ranks = np.apply_along_axis(lambda col: pd_rank(col), 0, feature_window)
    # correlate rows (time points) using their ranked 5-feature vectors
    ranks_centered = ranks - ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(ranks_centered, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = ranks_centered / norms
    corr = normed @ normed.T
    return np.clip(corr, -1.0, 1.0)


def pd_rank(col):
    order = col.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(col))
    return ranks


def build_correlation_snapshots(features, stride=4):
    """features: [N, 5] -> list of (center_idx, 32x32 matrix) for each
    sliding window of length WINDOW, stepped by `stride`."""
    n = len(features)
    snapshots = []
    centers = []
    for start in range(0, n - WINDOW + 1, stride):
        window = features[start:start + WINDOW]
        mat = spearman_corr_matrix(window)
        snapshots.append(mat)
        centers.append(start + WINDOW // 2)
    return np.array(snapshots, dtype=np.float32), np.array(centers)


class AttentionAutoencoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_heads=N_HEADS, latent_dim=LATENT_DIM):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 24), nn.ReLU(),
            nn.Linear(24, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24), nn.ReLU(),
            nn.Linear(24, embed_dim),
        )

    def encode(self, x):
        # x: [batch, 32, 32] - the correlation matrix itself is the token
        # sequence (32 tokens, each of dim 32), matching the paper's use of
        # permutation-invariant attention over the correlation structure.
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        z = self.encoder(x)          # [batch, 32, latent_dim]
        return z.mean(dim=1)         # pool over the 32 tokens -> [batch, latent_dim]

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        h = self.norm(x + attn_out)
        z_tokens = self.encoder(h)
        recon_tokens = self.decoder(z_tokens)
        return recon_tokens, z_tokens.mean(dim=1)


def train_autoencoder(snapshots, epochs=15, batch_size=64, lr=1e-3, verbose=False):
    model = AttentionAutoencoder()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(snapshots, dtype=torch.float32)
    n = len(X)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = X[idx]
            recon, _ = model(batch)
            loss = ((recon - batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if verbose:
            print(f"  epoch {epoch+1}/{epochs} recon_mse={total_loss/n:.5f}")
    model.eval()
    return model


def deepsupp_levels(model, highs, lows, closes, volumes, stride=4):
    """Frozen-model inference: build correlation snapshots for this window,
    encode to embeddings, DBSCAN-cluster, median price per cluster."""
    n = len(closes)
    if n < WINDOW + 5:
        return []
    features = compute_features(closes, volumes)
    snapshots, centers = build_correlation_snapshots(features, stride=stride)
    if len(snapshots) < 5:
        return []

    with torch.no_grad():
        X = torch.tensor(snapshots, dtype=torch.float32)
        embeddings = model.encode(X).numpy()

    min_samples = max(2, int(0.10 * len(embeddings)))
    # The paper's eps=0.1 is specific to their own encoder's embedding scale
    # and doesn't transfer to a reimplementation (a from-scratch encoder with
    # different init/architecture produces a different-scale embedding
    # space entirely). Derive eps from this window's own k-distance graph
    # instead of copying their constant - the standard DBSCAN elbow
    # heuristic, using the 40th percentile of each point's distance to its
    # min_samples-th nearest neighbor.
    if len(embeddings) > min_samples + 1:
        nn_finder = NearestNeighbors(n_neighbors=min_samples).fit(embeddings)
        kdist, _ = nn_finder.kneighbors(embeddings)
        eps = float(np.percentile(kdist[:, -1], 40))
        eps = max(eps, 1e-3)
    else:
        eps = 0.2
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(embeddings)

    levels = []
    for label in set(labels):
        if label == -1:
            continue
        idx = centers[labels == label]
        idx = idx[idx < n]
        if len(idx) == 0:
            continue
        count = len(idx)
        price = float(np.median(closes[idx]))
        if price <= 0:
            continue
        confidence = float(np.clip(0.5 + 0.4 * (count / len(embeddings)), 0, 0.95))
        levels.append({
            'price': price, 'type': 'DeepSupp Cluster', 'touches': int(count),
            'strength': confidence, 'breakoutProb': float(1 - confidence),
            'reversionProb': confidence, 'category': 'DeepSupp',
        })
    return levels
