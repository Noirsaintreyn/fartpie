#!/usr/bin/env python3
"""
Data preparation script for backtesting level detection methods
Converts various data formats to the required OHLCV format

Required format:
Date,Open,High,Low,Close,Volume
2024-01-01,100.0,102.0,99.5,101.5,1000000
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import argparse

def standardize_columns(df):
    """
    Standardize column names to match required format
    """
    # Common column name variations
    column_mapping = {
        'date': 'Date',
        'datetime': 'Date',
        'time': 'Date',
        'timestamp': 'Date',
        'open': 'Open',
        'o': 'Open',
        'high': 'High', 
        'h': 'High',
        'low': 'Low',
        'l': 'Low',
        'close': 'Close',
        'c': 'Close',
        'volume': 'Volume',
        'vol': 'Volume',
        'v': 'Volume'
    }
    
    # Apply mapping (case insensitive)
    df.columns = [column_mapping.get(col.lower(), col) for col in df.columns]
    
    return df

def validate_ohlcv(df):
    """
    Validate that DataFrame has required OHLCV columns
    """
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    return True

def clean_data(df):
    """
    Clean and validate OHLCV data
    """
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    # Ensure High >= Low
    invalid_hl = df['High'] < df['Low']
    if invalid_hl.any():
        print(f"Warning: {invalid_hl.sum()} rows have High < Low, fixing...")
        df.loc[invalid_hl, 'High'] = df.loc[invalid_hl, 'Low']
    
    # Ensure High >= Open,Close and Low <= Open,Close
    for price_col in ['Open', 'Close']:
        invalid_high = df['High'] < df[price_col]
        invalid_low = df['Low'] > df[price_col]
        
        if invalid_high.any():
            df.loc[invalid_high, 'High'] = df.loc[invalid_high, price_col]
        if invalid_low.any():
            df.loc[invalid_low, 'Low'] = df.loc[invalid_low, price_col]
    
    # Fill missing Volume with 0
    df['Volume'] = df['Volume'].fillna(0)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['Date'])
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

def generate_sample_data(ticker="SAMPLE", days=252, timeframe='1d'):
    """
    Generate sample OHLCV data for testing
    """
    print(f"Generating sample data for {ticker} - {days} days")
    
    # Generate realistic price series
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Simulate price movement with trend and volatility
    returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
    prices = [100.0]  # Starting price
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate OHLC from prices
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Add some intraday noise
        noise = np.random.normal(0, 0.005, 4)
        
        open_price = close * (1 + noise[0])
        close_price = close
        high_price = max(open_price, close_price) * (1 + abs(noise[1]))
        low_price = min(open_price, close_price) * (1 - abs(noise[2]))
        
        # Generate volume (correlated with price movement)
        volume_base = 1000000
        volume_variation = abs(returns[i]) * 10 if i < len(returns) else 1
        volume = volume_base * (1 + volume_variation)
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': int(volume)
        })
    
    df = pd.DataFrame(data)
    return df

def process_file(input_path, output_path, ticker=None):
    """
    Process a single data file
    """
    print(f"Processing {input_path}")
    
    try:
        # Determine file format by extension
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.xlsx') or input_path.endswith('.xls'):
            df = pd.read_excel(input_path)
        elif input_path.endswith('.json'):
            df = pd.read_json(input_path)
        else:
            print(f"Unsupported file format: {input_path}")
            return False
        
        # Standardize columns
        df = standardize_columns(df)
        
        # Validate required columns
        if not validate_ohlcv(df):
            return False
        
        # Clean data
        df = clean_data(df)
        
        # Save processed data
        df.to_csv(output_path, index=False)
        
        print(f"✓ Processed {len(df)} rows")
        print(f"✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"✓ Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
        print(f"✓ Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Prepare data for backtesting')
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--input_dir', type=str, help='Input directory path')
    parser.add_argument('--output_dir', type=str, default='prepared_data', help='Output directory')
    parser.add_argument('--generate_sample', action='store_true', help='Generate sample data')
    parser.add_argument('--ticker', type=str, default='SAMPLE', help='Ticker for sample data')
    parser.add_argument('--days', type=int, default=252, help='Days for sample data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate sample data if requested
    if args.generate_sample:
        sample_df = generate_sample_data(args.ticker, args.days)
        output_path = os.path.join(args.output_dir, f"{args.ticker}_1d.csv")
        sample_df.to_csv(output_path, index=False)
        print(f"✓ Generated sample data: {output_path}")
        return
    
    # Process files
    if args.input:
        # Single file
        output_name = os.path.splitext(os.path.basename(args.input))[0] + "_prepared.csv"
        output_path = os.path.join(args.output_dir, output_name)
        process_file(args.input, output_path)
        
    elif args.input_dir:
        # Directory of files
        for filename in os.listdir(args.input_dir):
            if filename.endswith(('.csv', '.xlsx', '.xls', '.json')):
                input_path = os.path.join(args.input_dir, filename)
                output_name = os.path.splitext(filename)[0] + "_prepared.csv"
                output_path = os.path.join(args.output_dir, output_name)
                process_file(input_path, output_path)
    
    else:
        print("Please provide --input, --input_dir, or --generate_sample")
        print("\nExamples:")
        print("  python prepare_data.py --generate_sample --ticker SPY")
        print("  python prepare_data.py --input data.csv --output_dir prepared")
        print("  python prepare_data.py --input_dir raw_data --output_dir prepared")

if __name__ == "__main__":
    main()
