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

# Define model (same as in backend.py)
class LevelDetectionNet(nn.Module):
    """Neural Network for Level Prediction - CNN + Attention"""
    def __init__(self, lookback=100):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.attention = nn.MultiheadAttention(64, num_heads=4)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, ohlc):
        x = ohlc.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.transpose(1, 2).transpose(0, 1)
        x, _ = self.attention(x, x, x)
        x = x.transpose(0, 1)
        level_logits = self.fc(x)
        return level_logits.squeeze(-1)

def calculate_hdbscan_levels(highs, lows, closes, timeframe='1d'):
    """HDBSCAN level detection"""
    if len(closes) < 20:
        return []
    
    all_prices = np.concatenate([highs, lows, closes])
    prices_array = all_prices.reshape(-1, 1)
    
    n_samples = len(prices_array)
    if 'm' in timeframe.lower():
        min_cluster_size = max(3, min(8, n_samples // 20))
        min_samples = max(2, min_cluster_size // 2)
    elif 'h' in timeframe.lower():
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
    for label in set(labels):
        if label == -1:
            continue
        cluster_mask = labels == label
        cluster_prices = all_prices[cluster_mask]
        cluster_probs = probabilities[cluster_mask]
        center = np.average(cluster_prices, weights=cluster_probs)
        strength = np.mean(cluster_probs) if len(cluster_probs) > 0 else 0.5
        levels.append({'price': float(center), 'strength': float(strength)})
    
    return levels

def enhanced_optics_levels(highs, lows, closes, timeframe='1d'):
    """OPTICS level detection"""
    if len(closes) < 20:
        return []
    
    all_prices = np.concatenate([highs, lows, closes]).reshape(-1, 1)
    optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=10)
    labels = optics.fit_predict(all_prices)
    
    levels = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_prices = all_prices[labels == label].flatten()
        center = np.median(cluster_prices)
        strength = 0.7
        levels.append({'price': float(center), 'strength': float(strength)})
    
    return levels

def train_level_detector(ticker='SPY', timeframe='1d', lookback=100, epochs=30, batch_size=32):
    """Train the level detection network"""
    print(f"Training level detection network for {ticker} at {timeframe}...")
    
    # Fetch data
    stock = yf.Ticker(ticker)
    interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '1h', '1d': '1d'}
    interval = interval_map.get(timeframe, '1d')
    period_map = {'1m': '1mo', '5m': '3mo', '15m': '6mo', '1h': '1y', '4h': '1y', '1d': '2y'}
    period = period_map.get(timeframe, '1y')
    
    print(f"Fetching {period} of {interval} data...")
    hist = stock.history(period=period, interval=interval)

    # If user requested 4h, yfinance doesn't support 4h directly.
    # We download 1h data above and resample it into 4h candles.
    if timeframe == '4h':
        if not isinstance(hist.index, pd.DatetimeIndex):
            hist.index = pd.to_datetime(hist.index)
        hist = hist.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    
    if len(hist) < lookback * 2:
        print(f"Error: Need at least {lookback * 2} bars, got {len(hist)}")
        return False
    
    closes = hist['Close'].values
    highs = hist['High'].values
    lows = hist['Low'].values
    
    print("Generating training samples...")
    X_train = []
    y_train = []
    
    # Generate training data
    for i in range(lookback, len(hist) - 10):
        window_hist = hist.iloc[i-lookback:i]
        window_highs = highs[i-lookback:i]
        window_lows = lows[i-lookback:i]
        window_closes = closes[i-lookback:i]
        
        # Get ground truth levels
        try:
            hdbscan_levels = calculate_hdbscan_levels(window_highs, window_lows, window_closes, timeframe=timeframe)
            optics_levels = enhanced_optics_levels(window_highs, window_lows, window_closes, timeframe=timeframe)
            all_truth_levels = hdbscan_levels + optics_levels
            truth_prices = [l.get('price', 0) for l in all_truth_levels if 'price' in l]
        except:
            continue
        
        # Prepare OHLC
        ohlc_data = window_hist[['Open', 'High', 'Low', 'Close']].values
        ohlc_mean = ohlc_data.mean(axis=0)
        ohlc_std = ohlc_data.std(axis=0) + 1e-9
        ohlc_normalized = (ohlc_data - ohlc_mean) / ohlc_std
        
        # Create labels
        labels = np.zeros(lookback)
        price_tolerance = np.std(window_closes) * 0.01
        
        for j, close_price in enumerate(window_closes):
            for truth_price in truth_prices:
                if abs(close_price - truth_price) < price_tolerance:
                    labels[j] = 1.0
                    break
        
        if np.sum(labels) > 0:
            X_train.append(ohlc_normalized)
            y_train.append(labels)
    
    if len(X_train) == 0:
        print("Error: No training samples generated")
        return False
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    
    print(f"Generated {len(X_train)} training samples")
    print(f"Positive label rate: {np.mean(y_train):.2%}")
    
    # Split train/val
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train[:split_idx]
    y_train_split = y_train[:split_idx]
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_split)
    y_train_tensor = torch.FloatTensor(y_train_split)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Initialize model
    model = LevelDetectionNet(lookback=lookback)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Training for {epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_start in range(0, len(X_train_tensor), batch_size):
            batch_end = min(batch_start + batch_size, len(X_train_tensor))
            X_batch = X_train_tensor[batch_start:batch_end]
            y_batch = y_train_tensor[batch_start:batch_end]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / (len(X_train_tensor) / batch_size)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()
            val_probs = torch.sigmoid(val_logits)
            val_preds = (val_probs > 0.5).float()
            val_acc = (val_preds == y_val_tensor).float().mean().item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'level_detector.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2%}")
    
    print(f"[OK] Training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"[OK] Model saved to level_detector.pth")
    
    # Final metrics
    model.eval()
    with torch.no_grad():
        final_logits = model(X_val_tensor)
        final_probs = torch.sigmoid(final_logits)
        final_preds = (final_probs > 0.5).float()
        final_acc = (final_preds == y_val_tensor).float().mean().item()
        
        tp = ((final_preds == 1) & (y_val_tensor == 1)).float().sum().item()
        pp = (final_preds == 1).float().sum().item()
        ap = (y_val_tensor == 1).float().sum().item()
        
        precision = tp / (pp + 1e-9)
        recall = tp / (ap + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    print(f"\nFinal Metrics:")
    print(f"  Accuracy: {final_acc:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1 Score: {f1:.2%}")
    
    return True

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
