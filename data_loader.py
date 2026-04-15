"""
Google Drive Data Loader for NQ, ES, VIX historical OHLCV data
Integrates with existing level detection and training systems
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import gdown
from pathlib import Path

# Google Drive folder URL
GOOGLE_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1vyuGhyGaSaMpw5rrypvzpj33CaGOPmVn"

# Local data directory
DATA_DIR = Path("historical_data")
DATA_DIR.mkdir(exist_ok=True)

# Supported symbols and their mappings
SYMBOL_MAPPING = {
    'NQ': ['NQ', 'NQ=F', '^IXIC'],  # Nasdaq 100 Futures / Nasdaq Composite
    'ES': ['ES', 'ES=F', '^GSPC'],  # S&P 500 Futures / S&P 500 Index  
    'VIX': ['VIX', '^VIX']          # VIX Index
}

def download_google_drive_data():
    """
    Download historical data from Google Drive
    Returns: dict with symbol -> file path mapping
    """
    print("📥 Downloading historical data from Google Drive...")
    
    # Download the entire folder
    try:
        url = "https://drive.google.com/drive/folders/1vyuGhyGaSaMpw5rrypvzpj33CaGOPmVn"
        gdown.download_folder(url, output=str(DATA_DIR), quiet=False)
        print(f"✅ Data downloaded to {DATA_DIR}")
        
        # Find downloaded CSV files
        data_files = {}
        for file_path in DATA_DIR.glob("*.csv"):
            # Extract symbol from filename
            symbol = file_path.stem.upper().split('_')[0]
            if symbol in SYMBOL_MAPPING:
                data_files[symbol] = file_path
                print(f"📊 Found {symbol} data: {file_path}")
        
        return data_files
        
    except Exception as e:
        print(f"❌ Failed to download from Google Drive: {e}")
        print("🔄 Falling back to yfinance data...")
        return {}

def load_historical_data(symbol, timeframe='1d', start_date=None, end_date=None, combine_with_realtime=False):
    """
    Load historical data for a symbol, combining Google Drive historical data with real-time yfinance
    
    Args:
        symbol: 'NQ', 'ES', or 'VIX'
        timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
        start_date: datetime or None
        end_date: datetime or None
        combine_with_realtime: If True, combine Google Drive historical data with yfinance real-time data
    
    Returns:
        pandas DataFrame with OHLCV data
    """
    symbol = symbol.upper()
    
    if combine_with_realtime and symbol in ['NQ', 'ES', 'VIX']:
        # Combine Google Drive historical data with real-time yfinance data
        return load_combined_data(symbol, timeframe, start_date, end_date)
    else:
        # Try Google Drive data first
        google_data = load_google_drive_data(symbol, timeframe, start_date, end_date)
        if google_data is not None:
            print(f"✅ Using Google Drive data for {symbol} {timeframe}")
            return google_data
        
        # Fallback to yfinance
        print(f"🔄 Falling back to yfinance for {symbol} {timeframe}")
        return load_yfinance_data(symbol, timeframe, start_date, end_date)

def load_combined_data(symbol, timeframe='1d', start_date=None, end_date=None):
    """
    Combine Google Drive historical data with real-time yfinance data
    Google Drive provides data up to its latest date, yfinance fills the gap to present
    """
    print(f"🔄 Combining Google Drive + real-time data for {symbol} {timeframe}")
    
    # Load Google Drive historical data
    google_data = load_google_drive_data(symbol, timeframe, start_date, None)
    if google_data is None:
        print(f"⚠ Google Drive data not available, using yfinance only")
        return load_yfinance_data(symbol, timeframe, start_date, end_date)
    
    # Get the latest date from Google Drive data
    google_latest = google_data.index.max()
    print(f"📅 Google Drive data available until: {google_latest}")
    
    # Load real-time yfinance data from the day after Google Drive data ends
    realtime_start = google_latest + pd.Timedelta(days=1)
    realtime_data = load_yfinance_data(symbol, timeframe, realtime_start, end_date)
    
    if realtime_data is None:
        print(f"⚠ Real-time yfinance data not available, using Google Drive data only")
        return google_data
    
    print(f"📈 Real-time yfinance data from {realtime_data.index.min()} to {realtime_data.index.max()}")
    
    # Combine the datasets
    combined_data = pd.concat([google_data, realtime_data], ignore_index=False)
    
    # Remove any duplicates (keep the last occurrence for real-time data)
    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
    
    # Sort by date
    combined_data = combined_data.sort_index()
    
    # Filter by requested date range
    if start_date:
        combined_data = combined_data[combined_data.index >= start_date]
    if end_date:
        combined_data = combined_data[combined_data.index <= end_date]
    
    print(f"✅ Combined dataset: {len(google_data)} bars from Google Drive + {len(realtime_data)} bars from yfinance = {len(combined_data)} total bars")
    
    return combined_data

def load_google_drive_data(symbol, timeframe='1d', start_date=None, end_date=None):
    """
    Load data from downloaded Google Drive files
    """
    # Find the data file for this symbol
    data_files = list(DATA_DIR.glob(f"{symbol}*.csv"))
    if not data_files:
        return None
    
    # Use the most recent file
    data_file = max(data_files, key=os.path.getctime)
    
    try:
        # Load the CSV
        df = pd.read_csv(data_file)
        
        # Standardize column names
        df.columns = [col.strip().upper() for col in df.columns]
        
        # Find required columns
        col_mapping = {}
        for col in df.columns:
            if 'OPEN' in col or col == 'O':
                col_mapping['Open'] = col
            elif 'HIGH' in col or col == 'H':
                col_mapping['High'] = col
            elif 'LOW' in col or col == 'L':
                col_mapping['Low'] = col
            elif 'CLOSE' in col or col == 'C' or col in ['PRICE', 'LAST']:
                col_mapping['Close'] = col
            elif 'VOLUME' in col or col == 'V':
                col_mapping['Volume'] = col
            elif 'DATE' in col or col == 'D' or col == 'TIME':
                col_mapping['Datetime'] = col
        
        # Check if we have required columns
        if not all(col in col_mapping for col in ['Open', 'High', 'Low', 'Close']):
            print(f"❌ Missing required OHLC columns in {data_file}")
            return None
        
        # Select and rename columns
        df = df[[col_mapping[col] for col in ['Open', 'High', 'Low', 'Close'] + 
                ([col_mapping['Volume']] if 'Volume' in col_mapping else [])]]
        df.columns = ['Open', 'High', 'Low', 'Close'] + \
                     (['Volume'] if 'Volume' in col_mapping else [])
        
        # Parse datetime
        if 'Datetime' in col_mapping:
            df['Datetime'] = pd.to_datetime(df[col_mapping['Datetime']])
        else:
            # Try to use index as datetime
            df.index = pd.to_datetime(df.index)
            df = df.reset_index()
            df = df.rename(columns={'index': 'Datetime'})
        
        # Set datetime as index
        df = df.set_index('Datetime')
        
        # Sort by date
        df = df.sort_index()
        
        # Filter by date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Handle missing volume
        if 'Volume' not in df.columns:
            df['Volume'] = np.ones(len(df)) * 1000  # Default volume
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        print(f"📊 Loaded {len(df)} bars of {symbol} {timeframe} data from Google Drive")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading {data_file}: {e}")
        return None

def load_yfinance_data(symbol, timeframe='1d', start_date=None, end_date=None):
    """
    Fallback to yfinance data
    """
    try:
        # Map symbol to yfinance ticker
        yf_symbols = SYMBOL_MAPPING.get(symbol, [symbol])
        yf_symbol = yf_symbols[0]  # Use first available
        
        # Map timeframe to yfinance interval
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', 
            '1h': '1h', '4h': '1h', '1d': '1d'
        }
        interval = interval_map.get(timeframe, '1d')
        
        # Determine period
        if start_date and end_date:
            period = None
            start = start_date.strftime('%Y-%m-%d')
            end = end_date.strftime('%Y-%m-%d')
        else:
            period_map = {
                '1m': '7d', '5m': '60d', '15m': '60d',
                '1h': '2y', '4h': '2y', '1d': '5y'
            }
            period = period_map.get(timeframe, '5y')
            start = None
            end = None
        
        # Download data
        ticker = yf.Ticker(yf_symbol)
        if period:
            df = ticker.history(period=period, interval=interval)
        else:
            df = ticker.history(start=start, end=end, interval=interval)
        
        if len(df) == 0:
            print(f"❌ No data found for {symbol}")
            return None
        
        # Resample 4h from 1h if needed
        if timeframe == '4h' and interval == '1h':
            df = df.resample('4H').agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        
        print(f"📊 Loaded {len(df)} bars of {symbol} {timeframe} data from yfinance")
        return df
        
    except Exception as e:
        print(f"❌ Error loading yfinance data for {symbol}: {e}")
        return None

def get_available_symbols():
    """
    Get list of available symbols with data
    """
    available = {}
    
    # Check Google Drive data
    for file_path in DATA_DIR.glob("*.csv"):
        symbol = file_path.stem.upper().split('_')[0]
        if symbol in SYMBOL_MAPPING:
            available[symbol] = {'source': 'google_drive', 'file': str(file_path)}
    
    # Add yfinance symbols as fallback
    for symbol in SYMBOL_MAPPING:
        if symbol not in available:
            available[symbol] = {'source': 'yfinance', 'ticker': SYMBOL_MAPPING[symbol][0]}
    
    return available

def initialize_data():
    """
    Initialize data system - download Google Drive data if needed
    """
    if not DATA_DIR.exists() or len(list(DATA_DIR.glob("*.csv"))) == 0:
        print("🔄 Initializing historical data system...")
        download_google_drive_data()
    
    available = get_available_symbols()
    print(f"📊 Available symbols: {list(available.keys())}")
    return available

if __name__ == "__main__":
    # Test the data loader
    print("🧪 Testing Google Drive data loader...")
    
    # Initialize
    available = initialize_data()
    
    # Test loading data for each symbol
    for symbol in ['NQ', 'ES', 'VIX']:
        if symbol in available:
            print(f"\n📈 Testing {symbol}...")
            df = load_historical_data(symbol, timeframe='1d')
            if df is not None:
                print(f"✅ {symbol}: {len(df)} bars loaded")
                print(f"📅 Range: {df.index.min()} to {df.index.max()}")
                print(f"💰 Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
            else:
                print(f"❌ {symbol}: Failed to load")
