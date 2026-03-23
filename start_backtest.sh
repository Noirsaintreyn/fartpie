#!/bin/bash

echo "🚀 Starting Level Detection Backtest Server"
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import flask, pandas, numpy, sklearn, hdbscan" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Missing required packages"
    echo "Run: pip install -r requirements.txt"
    exit 1
fi

echo "✅ Dependencies OK"

# Start the server
echo "🌐 Starting server on http://localhost:5001"
echo "📊 Backtest interface will be available at: http://localhost:5001/backtest"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 backend.py
