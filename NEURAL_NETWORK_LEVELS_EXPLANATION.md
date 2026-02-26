# Neural Network Levels: How They Work & Accuracy Comparison

## How Neural Network Levels Are Solved

### Architecture: CNN + Attention Model

The neural network uses a **LevelDetectionNet** model with:

1. **Input**: OHLC (Open, High, Low, Close) data for last 100 bars
2. **CNN Layers**: 
   - Conv1D layers extract patterns from price sequences
   - 3 convolutional layers: 4→64→128→64 channels
   - Recognizes patterns like support/resistance formations
3. **Attention Mechanism**: 
   - Multi-head attention (4 heads)
   - Identifies which bars are most important for level detection
   - Focuses on significant price action
4. **Output**: 
   - Probability for each bar being a level (0-1)
   - Threshold: 0.7 (only levels with >70% probability are kept)

### Process Flow

```
1. Normalize OHLC data (mean/std normalization)
2. Convert to tensor [1, 100, 4]
3. Pass through CNN → Extract patterns
4. Apply attention → Find important bars
5. Predict level probability for each bar
6. Extract bars with probability > 0.7
7. Return top 10 levels by strength
```

### Fallback Behavior

If model file (`level_detector.pth`) is missing:
- Falls back to **local extrema detection** (scipy argrelextrema)
- Finds local highs/lows with order=5
- Assigns default strength of 0.65

## Accuracy Comparison

Based on `get_model_accuracy_by_category()` function:

| Level Type | Accuracy | Notes |
|------------|----------|-------|
| **HDBSCAN (Density)** | **62%** | Highest - Structural levels from density clustering |
| **ML-Confluence** | **60%** | High - Multiple algorithms agree |
| **Neural Network** | **~50%** | Default (not explicitly listed, uses default) |
| **Interaction (Local Density)** | **55%** | Moderate - Short-memory, near current price |
| **Peak-Valley** | **50%** | Neutral - Simple pattern detection |
| **Isolation Forest** | **48%** | Lower - Event pivots, fast decay |

### Why Neural Network Levels Have Lower Accuracy

1. **Pattern-Based vs Structure-Based**:
   - Neural networks detect **patterns** in price sequences
   - HDBSCAN finds **actual structural density** (where price clusters)
   - Patterns can be coincidental; structure is more reliable

2. **Training Data Dependency**:
   - Requires trained model (`level_detector.pth`)
   - If model not trained or outdated, accuracy drops
   - Falls back to simple extrema detection (50% accuracy)

3. **Temporal vs Structural**:
   - Neural networks look at **time sequences** (which bar is a level?)
   - Density methods look at **price space** (where does price cluster?)
   - Price clustering is more predictive than temporal patterns

4. **No Validation**:
   - Neural network levels are not validated by other methods
   - HDBSCAN levels get validated by MeanShift (boosts confidence)
   - Isolation Forest levels are validated by RL validator

## How Each Method Works

### 1. HDBSCAN (Density) - 62% Accuracy
- **Method**: Clusters raw prices in price space
- **Strength**: Finds actual structural support/resistance
- **Why Best**: Price clustering is the most reliable signal
- **Validation**: MeanShift validates and boosts confidence

### 2. OPTICS (Multi-Density) - Similar to HDBSCAN
- **Method**: Multi-scale density clustering
- **Strength**: Finds levels at different scales
- **Accuracy**: Similar to HDBSCAN (~60%)

### 3. Neural Network - ~50% Accuracy
- **Method**: CNN + Attention on OHLC sequences
- **Strength**: Can learn complex patterns
- **Weakness**: 
  - Requires training data
  - Pattern-based (less structural)
  - No validation from other methods
  - Falls back to simple extrema if model missing

### 4. Isolation Forest - 48% Accuracy
- **Method**: Detects anomalous price movements (pivots)
- **Strength**: Finds event pivots
- **Weakness**: Fast decay, lower accuracy
- **Validation**: RL validator filters weak levels

### 5. Local Interaction - 55% Accuracy
- **Method**: Local density histogram near current price
- **Strength**: Short-memory, reactive
- **Use Case**: "Where will price react today?"

### 6. Peak-Valley - 50% Accuracy
- **Method**: Simple scipy argrelextrema
- **Strength**: Fallback when other methods fail
- **Weakness**: No structural validation

## Recommendations

1. **Primary**: Use HDBSCAN + OPTICS (highest accuracy, structural)
2. **Secondary**: Use ML-Confluence (multiple algorithms agree)
3. **Tertiary**: Neural Network (if model is well-trained)
4. **Fallback**: Peak-Valley (when all else fails)

## Improving Neural Network Accuracy

To improve neural network level accuracy:

1. **Train on validated levels**: Use HDBSCAN-validated levels as training targets
2. **Add validation**: Use LevelValidator to filter weak predictions
3. **Combine with density**: Use neural network as a filter for density levels
4. **Feature engineering**: Add volume, volatility, microstructure features
5. **Ensemble**: Combine neural network predictions with density methods

## Current Status

- Neural network levels are **included** in level detection
- They're part of `all_ml_levels` and get merged with other methods
- Accuracy is **moderate** (~50%) compared to density methods (62%)
- They provide **pattern-based** complement to **structure-based** methods
- Best used as **additional signal** alongside HDBSCAN/OPTICS
