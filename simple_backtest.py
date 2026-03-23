#!/usr/bin/env python3
"""
Simple backtesting for HDBSCAN levels only
Tests the core level detection without neural network dependencies
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

# Add backend path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend import calculate_hdbscan_levels

def simple_backtest(data_path, ticker="TEST", lookback=100, test_window=15):
    """
    Simple backtest for HDBSCAN levels
    """
    print(f"Loading data from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"Loaded {len(df)} bars for {ticker}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    results = {
        'total_levels': 0,
        'touched_levels': 0,
        'breakout_levels': 0,
        'level_details': []
    }
    
    # Slide window through data
    for i in range(lookback, len(df) - test_window):
        
        # Get historical data
        hist_window = df.iloc[i-lookback:i]
        hist_highs = hist_window['High'].values
        hist_lows = hist_window['Low'].values
        hist_closes = hist_window['Close'].values
        
        # Get future data
        future_window = df.iloc[i:i+test_window]
        current_price = df.iloc[i]['Close']
        
        # Detect HDBSCAN levels
        try:
            levels = calculate_hdbscan_levels(hist_highs, hist_lows, hist_closes, '1d')
        except Exception as e:
            print(f"Error at window {i}: {e}")
            continue
        
        # Test each level
        for level in levels:
            level_price = level.get('price', 0)
            strength = level.get('strength', 0.5)
            
            if not isinstance(level_price, (int, float)) or np.isnan(level_price):
                continue
            
            results['total_levels'] += 1
            
            # Test against future data
            touched = False
            breakout = False
            
            for _, row in future_window.iterrows():
                high, low = row['High'], row['Low']
                
                # Check for touch (within 0.5%)
                if low <= level_price <= high:
                    touched = True
                
                # Check for breakout (beyond 1%)
                if high > level_price * 1.01 or low < level_price * 0.99:
                    breakout = True
            
            if touched:
                results['touched_levels'] += 1
            if breakout:
                results['breakout_levels'] += 1
            
            # Store sample details
            if len(results['level_details']) < 20:
                results['level_details'].append({
                    'date': df.iloc[i]['Date'].strftime('%Y-%m-%d'),
                    'price': float(level_price),
                    'strength': float(strength),
                    'current_price': float(current_price),
                    'touched': touched,
                    'breakout': breakout
                })
    
    # Calculate metrics
    if results['total_levels'] > 0:
        results['success_rate'] = results['touched_levels'] / results['total_levels']
        results['breakout_rate'] = results['breakout_levels'] / results['total_levels']
    else:
        results['success_rate'] = 0
        results['breakout_rate'] = 0
    
    return results

def main():
    # Test with sample data
    data_path = "prepared_data/SPY_1d.csv"
    
    if not os.path.exists(data_path):
        print("Sample data not found. Generating...")
        os.system("python3 prepare_data.py --generate_sample --ticker SPY --days 300")
    
    print("\n" + "="*60)
    print("SIMPLE HDBSCAN BACKTEST")
    print("="*60)
    
    results = simple_backtest(data_path, "SPY")
    
    print(f"\nResults:")
    print(f"Total Levels: {results['total_levels']}")
    print(f"Touched Levels: {results['touched_levels']}")
    print(f"Breakout Levels: {results['breakout_levels']}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Breakout Rate: {results['breakout_rate']:.2%}")
    
    print(f"\nSample Levels:")
    for detail in results['level_details'][:5]:
        print(f"  {detail['date']}: Level ${detail['price']:.2f} (strength: {detail['strength']:.2f}) - Touched: {detail['touched']}")
    
    # Save results
    with open("simple_backtest_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: simple_backtest_results.json")

if __name__ == "__main__":
    main()
