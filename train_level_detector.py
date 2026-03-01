"""
Standalone script to train the neural network level detector
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from sklearn.cluster import OPTICS
    import hdbscan
    
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

class LevelDetectionNet(nn.Module):
    """CNN + (Bi)LSTM/GRU for per-bar level prediction (mirrors backend.LevelDetectionNet)."""
    def __init__(
        self,
        lookback: int = 100,
        rnn_type: str = 'bilstm',   # 'bilstm' or 'bigru'
        hidden_size: int = 64,
        num_layers: int = 1,
        use_volume_profile: bool = True,
        volume_feature_dim: int = 5,
    ):
        super().__init__()
        self.use_volume_profile = use_volume_profile
        self.rnn_type = rnn_type.lower()

        # CNN over OHLC (4 channels)
        self.conv1 = nn.Conv1d(4, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=5, padding=2)

        base_feat_dim = 64

        if self.use_volume_profile:
            self.volume_proj = nn.Linear(volume_feature_dim, 32)
            base_feat_dim += 32

        if self.rnn_type == 'bigru':
            self.rnn = nn.GRU(
                input_size=base_feat_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=base_feat_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )

        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, ohlc, volume_profile_features=None):
        # ohlc: [batch, seq_len, 4]
        x = ohlc.transpose(1, 2)          # [B, 4, L]
        x = torch.relu(self.conv1(x))     # [B, 64, L]
        x = torch.relu(self.conv2(x))     # [B, 128, L]
        x = torch.relu(self.conv3(x))     # [B, 64, L]
        x = x.transpose(1, 2)             # [B, L, 64]

        if self.use_volume_profile and volume_profile_features is not None:
            v = self.volume_proj(volume_profile_features)   # [B, L, 32]
            x = torch.cat([x, v], dim=-1)                  # [B, L, base_feat_dim]

        rnn_out, _ = self.rnn(x)          # [B, L, 2*hidden_size]
        logits = self.fc(rnn_out)         # [B, L, 1]
        return logits.squeeze(-1)         # [B, L]

def calculate_hdbscan_levels(highs, lows, closes, timeframe='1d'):
    """
    HDBSCAN: State-of-the-art density clustering
    Automatically finds optimal structure without parameters
    Clusters on RAW PRICES (canonical backend.py implementation).
    """
    if len(closes) < 20:
        return []

    all_prices = np.concatenate([highs, lows, closes])
    prices_array = all_prices.reshape(-1, 1)

    n_samples = len(prices_array)
    if 'm' in timeframe.lower() or 'min' in timeframe.lower():
        min_cluster_size = max(3, min(8, n_samples // 20))
        min_samples = max(2, min_cluster_size // 2)
    elif 'h' in timeframe.lower() or 'hour' in timeframe.lower():
        min_cluster_size = max(5, min(10, n_samples // 15))
        min_samples = max(3, min_cluster_size // 2)
    else:
        min_cluster_size = max(8, min(15, n_samples // 10))
        min_samples = max(5, min_cluster_size // 2)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=0.0,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    clusterer.fit(prices_array)
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_

    levels = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue

        cluster_mask = labels == label
        if np.sum(cluster_mask) == 0:
            continue

        cluster_prices = all_prices[cluster_mask]
        cluster_probs = probabilities[cluster_mask]

        center = np.average(cluster_prices, weights=cluster_probs)
        strength = np.mean(cluster_probs) if len(cluster_probs) > 0 else 0.5

        price_range = all_prices.max() - all_prices.min()
        price_tolerance = price_range * 0.01
        touches = np.sum(np.abs(all_prices - center) < price_tolerance)

        if not isinstance(center, (int, float)) or np.isnan(center) or np.isinf(center):
            continue

        clamped_strength = float(min(max(strength, 0.1), 0.93))

        levels.append({
            'price': float(center),
            'type': 'HDBSCAN Cluster',
            'touches': int(touches),
            'strength': clamped_strength,
            'breakoutProb': float(1 - clamped_strength),
            'reversionProb': clamped_strength,
            'category': 'Density (HDBSCAN)',
            'source': 'HDBSCAN',
            'avg_membership': float(strength),
            'cluster_size': int(np.sum(cluster_mask))
        })

    return sorted(levels, key=lambda x: x.get('avg_membership', 0), reverse=True)[:8]

def enhanced_optics_levels(highs, lows, closes, timeframe='1d'):
    """
    OPTICS with reachability-based strength scoring (canonical backend.py version).
    Reachability distance = "how dense is this cluster?"
    """
    if len(closes) < 20:
        return []

    all_prices = np.concatenate([highs, lows, closes]).reshape(-1, 1)

    optics = OPTICS(
        min_samples=5,
        xi=0.05,
        min_cluster_size=10,
        metric='euclidean'
    )

    labels = optics.fit_predict(all_prices)
    reachability = optics.reachability_[optics.ordering_]

    levels = []
    for label in set(labels):
        if label == -1:
            continue

        cluster_mask = labels == label
        cluster_prices = all_prices[cluster_mask].flatten()
        center = np.median(cluster_prices)

        cluster_indices = np.where(cluster_mask)[0]
        ordering_map = {optics.ordering_[i]: i for i in range(len(optics.ordering_))}
        cluster_reachability = [
            reachability[ordering_map.get(idx, 0)]
            for idx in cluster_indices if idx in ordering_map
        ]

        if len(cluster_reachability) == 0:
            continue

        avg_reachability = np.mean(cluster_reachability)

        price_scale = np.ptp(all_prices)
        normalized_reach = avg_reachability / (price_scale + 1e-9)
        strength = 1.0 / (1.0 + normalized_reach * 10)

        ordering_positions = [ordering_map.get(idx, 0) for idx in cluster_indices if idx in ordering_map]
        if len(ordering_positions) > 0:
            cluster_reach_vals = reachability[ordering_positions]
            local_min_reach = np.min(cluster_reach_vals)
            start_idx = max(0, min(ordering_positions) - 5)
            end_idx = min(len(reachability), max(ordering_positions) + 5)
            surrounding_reach = np.mean(reachability[start_idx:end_idx])
            valley_depth = (surrounding_reach - local_min_reach) / (surrounding_reach + 1e-9)

            strength *= (1.0 + 0.5 * valley_depth)
            strength = min(strength, 0.95)
        else:
            valley_depth = 0.0

        levels.append({
            'price': float(center),
            'type': 'OPTICS Density Valley',
            'strength': float(strength),
            'touches': len(cluster_prices),
            'avg_reachability': float(avg_reachability),
            'valley_depth': float(valley_depth),
            'category': 'OPTICS',
            'breakoutProb': float(1 - strength),
            'reversionProb': float(strength)
        })

    return sorted(levels, key=lambda x: x['strength'], reverse=True)[:8]

def train_level_detector(ticker='SPY', timeframe='1d', lookback=100, epochs=30, batch_size=32):
    """
    Backwards-compatible wrapper around backend.train_level_detection_network.
    This keeps the CLI interface while delegating to the canonical trainer.
    """
    try:
        from backend import train_level_detection_network
    except ImportError as e:
        print(f"Error importing backend.train_level_detection_network: {e}")
        return False

    result = train_level_detection_network(
        ticker=ticker,
        timeframe=timeframe,
        lookback=lookback,
        epochs=epochs,
        batch_size=batch_size
    )

    if isinstance(result, dict):
        return bool(result.get('success', False))
    return bool(result)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train neural network level detector')
    parser.add_argument('--ticker', default='SPY', help='Stock ticker')
    parser.add_argument('--timeframe', default='1d', help='Timeframe')
    parser.add_argument('--lookback', type=int, default=100, help='Lookback window')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    success = train_level_detector(
        ticker=args.ticker,
        timeframe=args.timeframe,
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    sys.exit(0 if success else 1)
