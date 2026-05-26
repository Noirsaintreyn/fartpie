#!/usr/bin/env python3
"""Test CSV persistence functionality"""
import sqlite3
import pandas as pd
import io
from datetime import datetime

DB_PATH = 'users.db'

def test_db_table_exists():
    """Test if csv_data table exists"""
    print("Testing database table...")
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='csv_data'")
        result = c.fetchone()
        conn.close()
        
        if result:
            print("✓ csv_data table exists")
            return True
        else:
            print("✗ csv_data table does not exist")
            return False
    except Exception as e:
        print(f"✗ Error checking table: {e}")
        return False

def test_save_csv():
    """Test saving CSV data"""
    print("\nTesting CSV save...")
    try:
        # Create sample DataFrame
        data = {
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [98, 99, 100],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        }
        dates = pd.date_range('2025-01-01', periods=3, freq='H')
        df = pd.DataFrame(data, index=dates)
        
        # Save to database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        csv_content = df.to_csv()
        ticker = 'TEST'
        timeframe = '1h'
        
        c.execute('''
            INSERT OR REPLACE INTO csv_data 
            (ticker, timeframe, csv_content, bars, start_date, end_date, uploaded_by, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            ticker,
            timeframe,
            csv_content,
            len(df),
            df.index[0].strftime('%Y-%m-%d %H:%M'),
            df.index[-1].strftime('%Y-%m-%d %H:%M'),
            'test_user'
        ))
        
        conn.commit()
        conn.close()
        print("✓ CSV data saved successfully")
        return True
    except Exception as e:
        print(f"✗ Error saving CSV: {e}")
        return False

def test_load_csv():
    """Test loading CSV data"""
    print("\nTesting CSV load...")
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT ticker, timeframe, csv_content, bars FROM csv_data WHERE ticker = ?', ('TEST',))
        row = c.fetchone()
        conn.close()
        
        if row:
            ticker, timeframe, csv_content, bars = row
            df = pd.read_csv(io.StringIO(csv_content), index_col=0, parse_dates=True)
            print(f"✓ Loaded {bars} bars for {ticker} {timeframe}")
            print(f"  Data shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            return True
        else:
            print("✗ No data found")
            return False
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return False

def test_delete_csv():
    """Test deleting CSV data"""
    print("\nTesting CSV delete...")
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM csv_data WHERE ticker = ?', ('TEST',))
        conn.commit()
        deleted = c.rowcount
        conn.close()
        
        if deleted > 0:
            print(f"✓ Deleted {deleted} record(s)")
            return True
        else:
            print("✗ No records deleted")
            return False
    except Exception as e:
        print(f"✗ Error deleting CSV: {e}")
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("CSV Persistence Test Suite")
    print("=" * 50)
    
    results = []
    results.append(("Table exists", test_db_table_exists()))
    results.append(("Save CSV", test_save_csv()))
    results.append(("Load CSV", test_load_csv()))
    results.append(("Delete CSV", test_delete_csv()))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} | {name}")
    
    all_passed = all(r[1] for r in results)
    print("=" * 50)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 50)
