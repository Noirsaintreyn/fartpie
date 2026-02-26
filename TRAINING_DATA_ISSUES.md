# Training Data Generation Issues Found

## Issues Identified:

### 1. **MTF Levels Fetched Once (Line 8526)**
- **Problem**: Multi-timeframe levels are fetched once for the entire dataset, using the full historical period
- **Impact**: All training samples use the same static levels, not dynamically detected levels per sample window
- **Current**: `mtf_levels = get_multi_timeframe_levels(ticker, base_timeframe)` - fetches levels from entire period
- **Issue**: Each sample should ideally use levels detected from its specific window context

### 2. **Time Bars Calculation (Line 8629)**
- **Status**: ✅ **CORRECT** - `bar_index` is integer index (0, 1, 2...) from enumerate, correctly used
- **Line 8476**: Uses `idx` from `enumerate(future_data.iterrows())` which is correct integer index
- **Line 8629**: `time_bars = max(1, bar_index)` - correctly clamps to minimum 1 bar

### 3. **Historical Touches Accumulation (Line 8644-8659)**
- **Current**: Historical touches accumulate across ALL samples in training loop
- **Impact**: Later samples have context from earlier samples (could be data leakage, but intentional for session context)
- **Fix Needed**: For training, should reset historical_touches per sample OR use windowed history

### 4. **Feature Engineering (Line 8584-8614)**
- **Status**: ✅ **CORRECT** - Features engineered for each bar in lookback window
- **Good**: Handles errors gracefully with fallback to previous features or zeros

### 5. **Sample Skipping (Line 8579-8580)**
- **Issue**: If no levels are touched in 30-bar future window, sample is skipped
- **Impact**: Could result in imbalanced dataset (only samples where levels were touched)
- **Consideration**: This is actually correct - we only want to train on scenarios where levels were actually touched

## Recommendations:

### High Priority:
1. **Make MTF levels dynamic per sample window** - Recalculate levels for each sample's specific window (or at least use rolling window levels)
2. **Reset historical_touches per sample** - Or use only touches from current sample's window to avoid data leakage

### Medium Priority:
3. **Add validation logging** - Log how many samples were skipped, average levels per sample, feature statistics
4. **Normalize price_offset** - Currently raw percentage, should check if values are reasonable (not >50%)

### Low Priority:
5. **Add data quality checks** - Ensure no NaN/Inf values in features or targets
6. **Balance dataset** - Track up/down direction balance, ensure not too skewed
