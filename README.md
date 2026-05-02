FartPie – Level Engines Overview

This repo powers the multi-engine support / resistance detection used by FartPie (degencap.uk).

The key “modern” engine is:

- A **CNN + BiLSTM + Attention** bar-level detector (pattern-based)

It feeds into the production level stack alongside HDBSCAN, OPTICS, KDE, local interaction, etc.

---

## CNN + BiLSTM + Attention Levels

The neural network level detector is implemented as `LevelDetectionNet` in:

- `train_level_detector.py` (standalone trainer / CLI)
- `backend.py` (inline definition + `/api/train-level-detector`)

### What it predicts

- **Input**: the last 100 OHLC bars, optionally enriched with per-bar volume-profile features.
- **Output**: for each bar in the window, a **probability that this bar is a meaningful level** (0–1).
- These bar-level probabilities are turned into price levels and merged with the rest of the ML stack.

### Architecture (intuition)

1. **Preprocessing**
   - Normalize OHLC: subtract mean, divide by std per channel.
   - Compute volume-profile features per bar (distance to POC, value area, volume at price, etc.).
   - Concatenate into a feature tensor of shape `[batch, seq_len, feat_dim]`.

2. **Input projection**
   - A linear layer projects features into a hidden channel dimension, e.g. 64.

3. **CNN over time**
   - Two 1D convolutions over the time axis (`Conv1d`):
     - First conv refines local patterns in short rolling windows.
     - Second conv increases channel depth and captures more complex shapes.
   - This stage is good at spotting **local formations**: spikes, wicks, compression, micro swings.

4. **BiLSTM over the sequence**
   - A bidirectional LSTM reads the CNN features from both past and “future” within the window.
   - This encodes **context**: whether a bar is important given what happened before and after.

5. **Multi-head self-attention**
   - A Transformer-style attention layer runs on top of the BiLSTM outputs.
   - It learns which bars in the window matter most when deciding if a given bar is a level.

6. **Per-bar classifier**
   - A final linear head produces a single logit per bar.
   - Sigmoid converts logits → probabilities in `[0, 1]`.

### How levels are extracted

1. Run the model on the latest 100-bar window.
2. Take bars where probability > threshold (default **0.7**).
3. Map those bar indices back to prices (close or high/low depending on the use case).
4. De-duplicate nearby prices and keep the strongest ~10 candidates by probability.
5. Hand these level objects to the same **agglomerative merge / confluence** pipeline used for HDBSCAN and friends.

### Training and fallback

- Training is done via:
  - `python train_level_detector.py` (standalone), or
  - `POST /api/train-level-detector` (backend endpoint).
- If the trained weights file (`level_detector.pth`) is missing or invalid:
  - The system **falls back to local extrema detection** (scipy `argrelextrema`) with a default strength,
  - So the app never fully loses level output if the NN isn’t available.

---

## 📊 Backtesting

### Web Interface

A beautiful web interface is now available for backtesting level detection methods:

1. **Start the server:**
   ```bash
   ./start_backtest.sh
   # or
   python3 backend.py
   ```

2. **Open the backtest interface:**
   - Navigate to: http://localhost:5001/backtest
   - Login with your credentials (user1 / pw or create an account)

3. **Run backtests:**
   - Select ticker symbol (SPY, AAPL, TSLA, etc.)
   - Choose timeframe (1m, 5m, 15m, 1h, 4h, 1d)
   - Pick detection method:
     - **HDBSCAN**: Density clustering (no dependencies)
     - **Neural Network**: CNN+BiLSTM (requires PyTorch + trained model)
   - Adjust lookback/test windows
   - Click "🚀 Run Backtest"

4. **Features:**
   - **Single Method Testing**: Test one method at a time
   - **Compare All Methods**: Side-by-side comparison of all three methods
   - **Export Results**: Download results as JSON
   - **Real-time Metrics**: Success rate, breakout rate, level details
   - **Keyboard Shortcuts**:
     - `Ctrl+Enter`: Run backtest
     - `Ctrl+C`: Compare all methods
     - `Ctrl+S`: Export results

### Command Line Backtesting

For automated testing, use the command line tools:

```bash
# Prepare your data
python3 prepare_data.py --input_dir /path/to/data --output_dir prepared_data

# Run simple backtest (HDBSCAN only)
python3 simple_backtest.py

# Run comprehensive backtest
python3 backtest_levels.py --data_dir prepared_data --output results
```

### API Endpoint

You can also call the backtest API directly:

```bash
curl "http://localhost:5001/api/backtest?ticker=SPY&timeframe=1d&method=hdbscan&lookback=200&test_window=20"
```

### Understanding Results

- **Success Rate**: % of levels that get touched/retested
- **Breakout Rate**: % of levels that break through
- **Total Levels**: How many levels were detected
- **Touched Levels**: Levels that were actually tested by price

## 📝 Notes

- The system focuses on two core level detection methods: HDBSCAN and Neural Network (CNN+BiLSTM)
- All other legacy methods have been removed for better performance and maintainability
- The backend includes authentication system with user management
- Web interface provides easy backtesting with beautiful visualizations

How Neural Network Levels Are Solved

Architecture: CNN + Attention Model

The neural network uses a LevelDetectionNet model with:

Input: OHLC (Open, High, Low, Close) data for last 100 bars CNN Layers: Conv1D layers extract patterns from price sequences 3 convolutional layers: 4→64→128→64 channels Recognizes patterns like support/resistance formations Attention Mechanism: Multi-head attention (4 heads) Identifies which bars are most important for level detection Focuses on significant price action Output: Probability for each bar being a level (0-1) Threshold: 0.7 (only levels with >70% probability are kept) Process Flow

Normalize OHLC data (mean/std normalization)
Convert to tensor [1, 100, 4]
Pass through CNN → Extract patterns
Apply attention → Find important bars
Predict level probability for each bar
Extract bars with probability > 0.7
Return top 10 levels by strength Fallback Behavior
If model file (level_detector.pth) is missing:

Falls back to local extrema detection (scipy argrelextrema) Finds local highs/lows with order=5 Assigns default strength of 0.65 Accuracy Comparison

Based on get_model_accuracy_by_category() function:

Level Type Accuracy Notes HDBSCAN (Density) 62% Highest - Structural levels from density clustering ML-Confluence 60% High - Multiple algorithms agree Neural Network ~50% Default (not explicitly listed, uses default) Interaction (Local Density) 55% Moderate - Short-memory, near current price Peak-Valley 50% Neutral - Simple pattern detection Isolation Forest 48% Lower - Event pivots, fast decay Why Neural Network Levels Have Lower Accuracy

Pattern-Based vs Structure-Based:

Neural networks detect patterns in price sequences HDBSCAN finds actual structural density (where price clusters) Patterns can be coincidental; structure is more reliable Training Data Dependency:

Requires trained model (level_detector.pth) If model not trained or outdated, accuracy drops Falls back to simple extrema detection (50% accuracy) Temporal vs Structural:

Neural networks look at time sequences (which bar is a level?) Density methods look at price space (where does price cluster?) Price clustering is more predictive than temporal patterns No Validation:

Neural network levels are not validated by other methods HDBSCAN levels get validated by MeanShift (boosts confidence) Isolation Forest levels are validated by RL validator How Each Method Works

HDBSCAN (Density) - 62% Accuracy
Method: Clusters raw prices in price space Strength: Finds actual structural support/resistance Why Best: Price clustering is the most reliable signal Validation: MeanShift validates and boosts confidence 2. OPTICS (Multi-Density) - Similar to HDBSCAN

Method: Multi-scale density clustering Strength: Finds levels at different scales Accuracy: Similar to HDBSCAN (~60%) 3. Neural Network - ~50% Accuracy

Method: CNN + Attention on OHLC sequences Strength: Can learn complex patterns Weakness: Requires training data Pattern-based (less structural) No validation from other methods Falls back to simple extrema if model missing 4. Isolation Forest - 48% Accuracy

Method: Detects anomalous price movements (pivots) Strength: Finds event pivots Weakness: Fast decay, lower accuracy Validation: RL validator filters weak levels 5. Local Interaction - 55% Accuracy

Method: Local density histogram near current price Strength: Short-memory, reactive Use Case: "Where will price react today?" 6. Peak-Valley - 50% Accuracy

Method: Simple scipy argrelextrema Strength: Fallback when other methods fail Weakness: No structural validation Recommendations

Primary: Use HDBSCAN + OPTICS (highest accuracy, structural) Secondary: Use ML-Confluence (multiple algorithms agree) Tertiary: Neural Network (if model is well-trained) Fallback: Peak-Valley (when all else fails) Improving Neural Network Accuracy

To improve neural network level accuracy:

Train on validated levels: Use HDBSCAN-validated levels as training targets Add validation: Use LevelValidator to filter weak predictions Combine with density: Use neural network as a filter for density levels Feature engineering: Add volume, volatility, microstructure features Ensemble: Combine neural network predictions with density methods Current Status

Neural network levels are included in level detection They're part of all_ml_levels and get merged with other methods Accuracy is moderate (~50%) compared to density methods (62%) They provide pattern-based complement to structure-based methods Best used as additional signal alongside HDBSCAN/OPTICS
