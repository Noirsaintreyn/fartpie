# CSV Data Persistence

## Overview
The application now supports **persistent CSV data storage** that survives server restarts. All uploaded CSV files are stored in the SQLite database and automatically loaded when the server starts.

## Key Features

### 1. **Persistent Storage**
- CSV data is saved to the SQLite database (`users.db`)
- Data persists across server restarts
- Stored in the `csv_data` table

### 2. **Global Sharing**
- All users can see and use uploaded CSV data
- Any logged-in user can upload CSV files
- CSV data is shared across all users (not user-specific)

### 3. **Automatic Loading**
- When the server starts, all CSV data is automatically loaded from the database
- Data is kept in memory for fast access during runtime
- Background sync keeps database and memory in sync

## Database Schema

```sql
CREATE TABLE csv_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    csv_content TEXT NOT NULL,
    bars INTEGER,
    start_date TEXT,
    end_date TEXT,
    uploaded_by TEXT,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, timeframe)
)
```

## How It Works

### Upload Flow
1. User uploads CSV via `/api/motivewave/upload`
2. CSV is parsed and validated
3. Data is stored in memory (`MOTIVEWAVE_DATA` dict)
4. Data is **saved to database** for persistence
5. Success response returned to user

### Server Startup Flow
1. Database is initialized (`init_db()`)
2. All CSV data is loaded from database (`load_all_csv_from_db()`)
3. Data is restored to memory (`MOTIVEWAVE_DATA` dict)
4. Server is ready with all previous uploads

### Delete Flow
1. User deletes dataset via `/api/motivewave/delete`
2. Data is removed from memory
3. Data is **deleted from database**
4. Success response returned to user

## API Endpoints

### Upload CSV
```http
POST /api/motivewave/upload
Content-Type: multipart/form-data

ticker: NQ=F
timeframe: 4h
file: [CSV file]
```

### Get Status
```http
GET /api/motivewave/status
```
Returns list of all uploaded datasets with:
- Ticker
- Timeframe
- Number of bars
- Date range

### Delete Dataset
```http
POST /api/motivewave/delete
Content-Type: application/json

{
  "key": "NQ=F_4h"
}
```

## CSV Format

The application accepts MotiveWave CSV exports with the following columns:
- **Date** or **DateTime**: Timestamp for each bar
- **Open**: Opening price
- **High**: High price
- **Low**: Low price
- **Close**: Closing price
- **Volume** (optional): Trading volume

Example:
```csv
Date,Time,Open,High,Low,Close,Volume
2025-01-01,09:00,4500.00,4525.00,4490.00,4520.00,1000000
2025-01-01,10:00,4520.00,4540.00,4515.00,4535.00,1200000
```

## Code Changes

### Modified Files
1. **backend.py**
   - Added `csv_data` table creation in `init_db()`
   - Added `save_csv_to_db()` function
   - Added `load_all_csv_from_db()` function
   - Modified `/api/motivewave/upload` to save to DB
   - Modified `/api/motivewave/delete` to delete from DB
   - Added CSV loading on startup after `init_db()`

### New Functions
- `save_csv_to_db(ticker, timeframe, df, uploaded_by)`: Saves DataFrame to database
- `load_all_csv_from_db()`: Loads all CSV data from database on startup

## Testing

Run the test suite:
```bash
python test_csv_persistence.py
```

Tests verify:
- ✓ Database table exists
- ✓ CSV data can be saved
- ✓ CSV data can be loaded
- ✓ CSV data can be deleted

## Usage Example

1. **Login** to the application
2. Navigate to **MotiveWave Import** section
3. Select **ticker** (e.g., NQ=F)
4. Select **timeframe** (e.g., 4h)
5. Choose **CSV file** from MotiveWave export
6. Click **Import CSV**
7. Data is now persistent and available to all users!

## Notes

- CSV data replaces yfinance data for matching ticker/timeframe
- Only one CSV per ticker/timeframe combination (uploads overwrite)
- Data is stored as CSV text in the database (efficient and portable)
- Memory cache ensures fast access without database queries for every request

## Troubleshooting

### CSV not persisting after server restart
- Check if database file exists: `ls -la users.db`
- Check database logs during startup
- Verify `csv_data` table exists: `sqlite3 users.db ".schema csv_data"`

### Upload fails
- Ensure CSV has required columns (Date, Open, High, Low, Close)
- Check file format is valid CSV
- Verify user is logged in

### Data not loading on startup
- Check server startup logs for errors
- Verify `load_all_csv_from_db()` is called after `init_db()`
- Check database permissions

## Future Enhancements

Potential improvements:
- User-specific CSV data (private uploads)
- CSV version history
- Data validation and quality checks
- Compression for large CSV files
- Export functionality
- Bulk upload support
