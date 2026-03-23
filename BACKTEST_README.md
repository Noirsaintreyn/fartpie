# Level Detection Backtesting Framework

This framework allows you to backtest the three remaining level detection methods:
- **HDBSCAN** - Density clustering on raw prices
- **Neural Network** - CNN+BiLSTM+Attention sequence learning  
- **DeepSupp** - Correlation-series transformer autoencoder

## Quick Start

### 1. Prepare Your Data

First, get your data from Google Drive and prepare it in the required format:

```bash
# Option 1: Generate sample data to test
python prepare_data.py --generate_sample --ticker SPY --days 500

# Option 2: Prepare your own CSV file
python prepare_data.py --input /path/to/your/data.csv --output_dir prepared_data

# Option 3: Prepare all files in a directory
python prepare_data.py --input_dir /path/to/google_drive_data --output_dir prepared_data
```

**Required CSV Format:**
```
Date,Open,High,Low,Close,Volume
2024-01-01,100.0,102.0,99.5,101.5,1000000
2024-01-02,101.5,103.0,100.8,102.5,1200000
...
```

### 2. Run Backtests

```bash
# Single file backtest
python backtest_levels.py --data_path prepared_data/SPY_1d.csv --ticker SPY --timeframe 1d

# Directory backtest (multiple tickers)
python backtest_levels.py --data_dir prepared_data --output results

# Custom parameters
python backtest_levels.py \
    --data_dir prepared_data \
    --lookback 200 \
    --test_window 20 \
    --output results
```

### 3. View Results

Results are saved as JSON files with detailed metrics:

```bash
# View individual results
cat results/SPY_1d_backtest.json | python -m json.tool

# View combined results (multiple tickers)
cat results/combined_backtest_results.json | python -m json.tool
```

## Understanding the Metrics

### Key Performance Indicators

| Metric | What It Means | Good Range |
|--------|---------------|------------|
| **Success Rate** | % of levels that get touched/retested | 60-80% |
| **Profit Factor** | Average profit / average loss | > 1.5 |
| **Breakout Rate** | % of levels that break out | 20-40% |
| **False Breakout Rate** | Breakouts that quickly fail | < 15% |
| **Avg Hold Time** | How long levels remain relevant | Varies by timeframe |

### Level Testing Logic

For each detected level, the backtester:

1. **Identifies Level**: Price level with strength score
2. **Future Test**: Checks next N bars (default: 20)
3. **Touch Detection**: Price comes within tolerance (default: 0.5%)
4. **Breakout Detection**: Price moves beyond tolerance
5. **Profit/Loss**: Simulates trading the level

**Level is considered valid if:**
- Gets touched minimum times (default: 2 touches), OR
- Breaks out (up or down)

## Advanced Usage

### Custom Parameters

```python
# In backtest_levels.py, adjust these parameters:

lookback_window = 200    # Bars used for level detection
test_window = 20         # Bars forward to test effectiveness  
min_touches = 2          # Minimum touches to validate level
tolerance_pct = 0.5      # Price tolerance for touches (%)
```

### Timeframe Considerations

| Timeframe | Lookback | Test Window | Tolerance |
|-----------|----------|-------------|-----------|
| 1m        | 500      | 60          | 0.1%      |
| 5m        | 300      | 48          | 0.2%      |
| 15m       | 200      | 32          | 0.3%      |
| 1h        | 200      | 24          | 0.4%      |
| 4h        | 150      | 18          | 0.5%      |
| 1d        | 100      | 15          | 0.5%      |

### Method-Specific Notes

#### HDBSCAN
- **Strengths**: No parameters required, finds natural price clusters
- **Best for**: All timeframes, especially where price clusters clearly
- **Expected Success**: 65-75% on daily data

#### Neural Network (CNN+BiLSTM)
- **Requirements**: PyTorch installed, trained model available
- **Strengths**: Learns complex patterns, considers volume profile
- **Best for**: Timeframes with sufficient training data
- **Expected Success**: 70-80% if well-trained

#### DeepSupp
- **Requirements**: PyTorch + deepsupp_v4.pt model file
- **Strengths**: Understands price-volume correlations
- **Best for**: Timeframes with clear volume patterns
- **Expected Success**: 60-70% on liquid instruments

## Troubleshooting

### Common Issues

**1. "PyTorch not available"**
```bash
pip install torch
```

**2. "Missing deepsupp_v4.pt model"**
- The model should be in your fartpie directory
- If missing, the backtest will skip DeepSupp testing

**3. "Insufficient data"**
- Need at least 400 bars for DeepSupp training/testing
- Need at least lookback_window + test_window bars for backtesting

**4. "Column not found" errors**
- Use prepare_data.py to standardize column names
- Ensure you have: Date,Open,High,Low,Close,Volume

### Performance Tips

1. **Start with sample data** to verify setup:
   ```bash
   python prepare_data.py --generate_sample --ticker TEST
   python backtest_levels.py --data_path prepared_data/TEST_1d.csv --ticker TEST
   ```

2. **Adjust parameters for your timeframe**:
   - Shorter timeframes need larger lookback windows
   - Higher volatility needs larger tolerance

3. **Monitor memory usage**:
   - Large datasets may require reducing lookback/test windows
   - Neural Network method is most memory-intensive

## Example Results

```
================================================================================
BACKTEST SUMMARY - SPY 1d
================================================================================
Data Points: 1000
Lookback Window: 200 bars
Test Window: 20 bars

Method          Levels   Valid   Success    Profit F   Breakout  
----------------------------------------------------------------------
hdbscan         156      142     73.24%     1.85       28.17%   
neural_network  98       87      78.16%     2.12       24.14%   
deepsupp        134      119     68.07%     1.67       31.09%   

================================================================================
BEST PERFORMERS:
================================================================================
Best Success Rate: neural_network (78.16%)
Best Profit Factor: neural_network (2.12)
Most Levels Found: hdbscan (156 levels)
Most Consistent: hdbscan (142 valid levels)
```

## Next Steps

1. **Download your Google Drive data** to a local folder
2. **Run the data preparation script** to format it correctly
3. **Start with sample parameters** to validate the setup
4. **Adjust parameters** based on your specific timeframe and instrument
5. **Compare results** across different timeframes and tickers
6. **Optimize** the best-performing method for your use case

The backtesting framework will help you understand which level detection method works best for your specific data and trading strategy.
