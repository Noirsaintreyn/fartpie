"""Universe switch for the xs pipeline.  XS_U=sp (default, S&P 500 PIT) or XS_U=mid
(optionable non-S&P names from DoltHub, liquidity-filtered)."""
import os

U = os.getenv('XS_U', 'sp')
X = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(X, 'data' if U == 'sp' else f'data_{U}')
REP = os.path.join(X, 'reports' if U == 'sp' else f'reports_{U}')
RAW = os.path.join(X, 'raw')
VH = os.path.join(RAW, 'dolt_volatility_history.csv' if U == 'sp' else 'dolt_volatility_history_all.csv')
os.makedirs(D, exist_ok=True)
