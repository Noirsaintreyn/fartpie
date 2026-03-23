#!/usr/bin/env python3
"""
Comprehensive Backtesting Framework for Level Detection Methods
Tests HDBSCAN, Neural Network (CNN+BiLSTM), and DeepSupp levels

Usage:
    python backtest_levels.py --data_path /path/to/your/data.csv --ticker SPY --timeframe 1d
    python backtest_levels.py --data_dir /path/to/data_folder --output results/
"""

import argparse
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add backend path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend import (
    calculate_hdbscan_levels,
    detect_levels_with_neural_network,
    detect_levels_with_deepsupp,
    calculate_volume_profile,
    ewma_volatility
)

# Check torch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available - Neural Network and DeepSupp backtesting disabled")

class LevelBacktester:
    """
    Backtesting engine for level detection methods
    """
    
    def __init__(self, data: pd.DataFrame, ticker: str = "UNKNOWN", timeframe: str = "1d"):
        """
        Initialize backtester with OHLCV data
        
        Args:
            data: DataFrame with columns: Date/Open/High/Low/Close/Volume
            ticker: Symbol name for reporting
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        """
        self.data = data.copy()
        self.ticker = ticker
        self.timeframe = timeframe
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Sort by date
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} bars for {ticker} {timeframe}")
        print(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        
    def backtest_method(self, method_name: str, lookback_window: int = 200, 
                       test_window: int = 20, min_touches: int = 2,
                       tolerance_pct: float = 0.5) -> Dict:
        """
        Backtest a specific level detection method
        
        Args:
            method_name: 'hdbscan', 'neural_network', or 'deepsupp'
            lookback_window: Bars to use for level detection
            test_window: Bars forward to test level effectiveness
            min_touches: Minimum touches to consider a level valid
            tolerance_pct: Price tolerance for level touches (percentage)
            
        Returns:
            Dictionary with backtest results
        """
        print(f"\n=== Backtesting {method_name.upper()} ===")
        
        results = {
            'method': method_name,
            'total_levels': 0,
            'valid_levels': 0,
            'touched_levels': 0,
            'breakout_levels': 0,
            'false_levels': 0,
            'avg_strength': 0.0,
            'avg_hold_time': 0.0,
            'profit_factors': [],
            'max_profit': 0.0,
            'max_loss': 0.0,
            'level_details': []
        }
        
        # Slide window through data
        for i in range(lookback_window, len(self.data) - test_window):
            
            # Get historical data for level detection
            hist_window = self.data.iloc[i-lookback_window:i].copy()
            
            # Get future data for testing
            future_window = self.data.iloc[i:i+test_window].copy()
            
            # Extract arrays
            hist_highs = hist_window['High'].values
            hist_lows = hist_window['Low'].values
            hist_closes = hist_window['Close'].values
            
            # Detect levels based on method
            levels = []
            
            try:
                if method_name == 'hdbscan':
                    levels = calculate_hdbscan_levels(
                        hist_highs, hist_lows, hist_closes, self.timeframe
                    )
                    
                elif method_name == 'neural_network':
                    if TORCH_AVAILABLE:
                        levels = detect_levels_with_neural_network(
                            hist_window, lookback=100, threshold=0.7
                        )
                    
                elif method_name == 'deepsupp':
                    if TORCH_AVAILABLE:
                        levels = detect_levels_with_deepsupp(
                            hist_window, model_path='deepsupp_v4.pt', device='cpu'
                        )
                        
            except Exception as e:
                print(f"Error in {method_name} at window {i}: {e}")
                continue
            
            # Test each level against future data
            current_price = self.data.iloc[i]['Close']
            
            for level in levels:
                level_price = level.get('price', 0)
                strength = level.get('strength', 0.5)
                
                if not isinstance(level_price, (int, float)) or np.isnan(level_price):
                    continue
                
                results['total_levels'] += 1
                
                # Test level in future window
                level_result = self._test_level(
                    level_price, strength, future_window, 
                    tolerance_pct, min_touches
                )
                
                # Update aggregate results
                if level_result['valid']:
                    results['valid_levels'] += 1
                    
                    if level_result['touched']:
                        results['touched_levels'] += 1
                    if level_result['breakout']:
                        results['breakout_levels'] += 1
                    if level_result['false_breakout']:
                        results['false_levels'] += 1
                    
                    results['profit_factors'].append(level_result['profit_factor'])
                    results['max_profit'] = max(results['max_profit'], level_result['max_profit'])
                    results['max_loss'] = max(results['max_loss'], abs(level_result['max_loss']))
                    results['avg_hold_time'] += level_result['hold_time']
                    
                    # Store detailed result for sample levels
                    if len(results['level_details']) < 50:  # Limit detail storage
                        results['level_details'].append({
                            'date': self.data.iloc[i]['Date'].strftime('%Y-%m-%d'),
                            'price': float(level_price),
                            'strength': float(strength),
                            'result': level_result
                        })
        
        # Calculate final metrics
        if results['valid_levels'] > 0:
            results['avg_hold_time'] /= results['valid_levels']
            results['avg_strength'] = np.mean([d['strength'] for d in results['level_details']])
            results['success_rate'] = results['touched_levels'] / results['valid_levels']
            results['breakout_rate'] = results['breakout_levels'] / results['valid_levels']
            results['false_breakout_rate'] = results['false_levels'] / results['valid_levels']
            results['avg_profit_factor'] = np.mean(results['profit_factors']) if results['profit_factors'] else 0
        else:
            results['success_rate'] = 0
            results['breakout_rate'] = 0
            results['false_breakout_rate'] = 0
            results['avg_profit_factor'] = 0
        
        return results
    
    def _test_level(self, level_price: float, strength: float, 
                   future_data: pd.DataFrame, tolerance_pct: float, 
                   min_touches: int) -> Dict:
        """
        Test a single level against future price action
        
        Returns:
            Dictionary with test results
        """
        result = {
            'valid': False,
            'touched': False,
            'breakout': False,
            'false_breakout': False,
            'hold_time': 0,
            'profit_factor': 0,
            'max_profit': 0,
            'max_loss': 0,
            'touches': 0
        }
        
        if len(future_data) == 0:
            return result
        
        # Calculate tolerance
        tolerance = level_price * (tolerance_pct / 100)
        
        # Track price action
        touched = False
        breakout_up = False
        breakout_down = False
        max_profit = 0
        max_loss = 0
        touches = 0
        
        for idx, row in future_data.iterrows():
            high = row['High']
            low = row['Low']
            close = row['Close']
            
            # Check for touch
            if low <= level_price <= high:
                touched = True
                touches += 1
                
                # Calculate profit/loss if we traded the level
                if level_price > close:  # Resistance level (short)
                    profit = level_price - close
                    max_profit = max(max_profit, profit)
                    max_loss = max(max_loss, close - level_price)
                else:  # Support level (long)
                    profit = close - level_price
                    max_profit = max(max_profit, profit)
                    max_loss = max(max_loss, level_price - close)
            
            # Check for breakout
            if high > level_price + tolerance:
                breakout_up = True
            if low < level_price - tolerance:
                breakout_down = True
        
        # Determine results
        result['touches'] = touches
        result['touched'] = touched and touches >= min_touches
        result['breakout'] = breakout_up or breakout_down
        
        # Check for false breakout (breaks then comes back)
        if result['breakout'] and touched:
            result['false_breakout'] = True
        
        result['valid'] = result['touched'] or result['breakout']
        result['hold_time'] = len(future_data)
        result['max_profit'] = max_profit
        result['max_loss'] = max_loss
        
        # Calculate profit factor
        if max_loss > 0:
            result['profit_factor'] = max_profit / max_loss if max_profit > 0 else -1
        else:
            result['profit_factor'] = max_profit if max_profit > 0 else 0
        
        return result
    
    def compare_methods(self, lookback_window: int = 200, test_window: int = 20) -> Dict:
        """
        Compare all three methods side by side
        
        Returns:
            Dictionary with comparison results
        """
        print(f"\n{'='*60}")
        print(f"COMPARING ALL METHODS - {self.ticker} {self.timeframe}")
        print(f"{'='*60}")
        
        methods = ['hdbscan']
        if TORCH_AVAILABLE:
            methods.extend(['neural_network', 'deepsupp'])
        
        results = {}
        
        for method in methods:
            try:
                results[method] = self.backtest_method(
                    method, lookback_window, test_window
                )
            except Exception as e:
                print(f"Failed to backtest {method}: {e}")
                results[method] = {'error': str(e)}
        
        # Create comparison summary
        comparison = {
            'ticker': self.ticker,
            'timeframe': self.timeframe,
            'data_points': len(self.data),
            'lookback_window': lookback_window,
            'test_window': test_window,
            'methods': results,
            'summary': self._create_summary(results)
        }
        
        return comparison
    
    def _create_summary(self, results: Dict) -> Dict:
        """
        Create summary comparison table
        """
        summary = {
            'best_success_rate': {'method': None, 'value': 0},
            'best_profit_factor': {'method': None, 'value': 0},
            'most_levels': {'method': None, 'value': 0},
            'most_consistent': {'method': None, 'value': 0}
        }
        
        for method, result in results.items():
            if 'error' in result:
                continue
            
            success_rate = result.get('success_rate', 0)
            profit_factor = result.get('avg_profit_factor', 0)
            total_levels = result.get('total_levels', 0)
            valid_levels = result.get('valid_levels', 0)
            
            # Update best metrics
            if success_rate > summary['best_success_rate']['value']:
                summary['best_success_rate'] = {'method': method, 'value': success_rate}
            
            if profit_factor > summary['best_profit_factor']['value']:
                summary['best_profit_factor'] = {'method': method, 'value': profit_factor}
            
            if total_levels > summary['most_levels']['value']:
                summary['most_levels'] = {'method': method, 'value': total_levels}
            
            if valid_levels > summary['most_consistent']['value']:
                summary['most_consistent'] = {'method': method, 'value': valid_levels}
        
        return summary
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save backtest results to JSON file
        """
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert
        def deep_convert(d):
            if isinstance(d, dict):
                return {k: deep_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [deep_convert(item) for item in d]
            else:
                return convert_numpy(d)
        
        results_json = deep_convert(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def print_summary(self, results: Dict):
        """
        Print formatted summary of results
        """
        print(f"\n{'='*80}")
        print(f"BACKTEST SUMMARY - {results['ticker']} {results['timeframe']}")
        print(f"{'='*80}")
        
        print(f"Data Points: {results['data_points']}")
        print(f"Lookback Window: {results['lookback_window']} bars")
        print(f"Test Window: {results['test_window']} bars")
        
        print(f"\n{'Method':<15} {'Levels':<8} {'Valid':<8} {'Success':<10} {'Profit F':<10} {'Breakout':<10}")
        print(f"{'-'*70}")
        
        for method, result in results['methods'].items():
            if 'error' in result:
                print(f"{method:<15} {'ERROR':<8} {'-':<8} {'-':<10} {'-':<10} {'-':<10}")
                continue
            
            total = result.get('total_levels', 0)
            valid = result.get('valid_levels', 0)
            success = result.get('success_rate', 0)
            profit_f = result.get('avg_profit_factor', 0)
            breakout = result.get('breakout_rate', 0)
            
            print(f"{method:<15} {total:<8} {valid:<8} {success:<10.2%} {profit_f:<10.2f} {breakout:<10.2%}")
        
        print(f"\n{'='*80}")
        print("BEST PERFORMERS:")
        print(f"{'='*80}")
        
        summary = results['summary']
        print(f"Best Success Rate: {summary['best_success_rate']['method']} ({summary['best_success_rate']['value']:.2%})")
        print(f"Best Profit Factor: {summary['best_profit_factor']['method']} ({summary['best_profit_factor']['value']:.2f})")
        print(f"Most Levels Found: {summary['most_levels']['method']} ({summary['most_levels']['value']} levels)")
        print(f"Most Consistent: {summary['most_consistent']['method']} ({summary['most_consistent']['value']} valid levels)")


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file
    Expected columns: Date,Open,High,Low,Close,Volume
    """
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.title()
        
        # Ensure Date column exists and is datetime
        if 'Date' not in df.columns and 'date' in df.columns:
            df = df.rename(columns={'date': 'Date'})
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        return df
        
    except Exception as e:
        print(f"Error loading CSV {file_path}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Backtest level detection methods')
    parser.add_argument('--data_path', type=str, help='Path to CSV file with OHLCV data')
    parser.add_argument('--data_dir', type=str, help='Directory containing CSV files')
    parser.add_argument('--ticker', type=str, default='UNKNOWN', help='Ticker symbol')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe (1m,5m,15m,1h,4h,1d)')
    parser.add_argument('--output', type=str, default='backtest_results', help='Output directory')
    parser.add_argument('--lookback', type=int, default=200, help='Lookback window for level detection')
    parser.add_argument('--test_window', type=int, default=20, help='Test window for validation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    if args.data_path:
        data_files = [args.data_path]
        tickers = [args.ticker]
    elif args.data_dir:
        data_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) 
                     if f.endswith('.csv')]
        tickers = [os.path.splitext(os.path.basename(f))[0] for f in data_files]
    else:
        print("Error: Must provide either --data_path or --data_dir")
        return
    
    # Run backtests
    all_results = {}
    
    for file_path, ticker in zip(data_files, tickers):
        try:
            print(f"\nProcessing {ticker} from {file_path}")
            
            # Load data
            data = load_data_from_csv(file_path)
            
            # Initialize backtester
            backtester = LevelBacktester(data, ticker, args.timeframe)
            
            # Run comparison
            results = backtester.compare_methods(args.lookback, args.test_window)
            
            # Save results
            output_file = os.path.join(args.output, f"{ticker}_{args.timeframe}_backtest.json")
            backtester.save_results(results, output_file)
            
            # Print summary
            backtester.print_summary(results)
            
            all_results[ticker] = results
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    # Save combined results
    if len(all_results) > 1:
        combined_file = os.path.join(args.output, "combined_backtest_results.json")
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nCombined results saved to: {combined_file}")


if __name__ == "__main__":
    main()
