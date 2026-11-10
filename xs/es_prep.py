"""Nov 8 2026: REAL NQ FUTURES 5-MINUTE BARS (Downloads/NQ_5Min.csv, Jan 2014 - Jan 2026, front contract, Central Time bar starts) -> parquet in Eastern Time.
Adds: trading day (the RTH date; the overnight session from 18:00 ET belongs to the next RTH day), RTH flag (09:30 - 16:00 ET), roll flag (contract changed vs the prior RTH day)."""
import pandas as pd, numpy as np
SRC = '/mnt/user-data/uploads/Downloads/ES/5Min/5Min_ES.csv'; OUT = '/tmp/claude-0/ml/es5_real.parquet'
import os; os.makedirs('/tmp/claude-0/ml', exist_ok=True)
D = pd.read_csv(SRC, sep=';', skiprows=1, dtype=str)
num = lambda s: pd.to_numeric(s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
X = pd.DataFrame({'dt': pd.to_datetime(D.Date, format='%m/%d/%Y %I:%M %p') + pd.Timedelta(hours=1), 'sym': D.Symbol, 'open': num(D.Open), 'high': num(D.High), 'low': num(D.Low),
                  'close': num(D.Close), 'volume': num(D.Volume)}).dropna().sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
X['m'] = X.dt.dt.hour * 60 + X.dt.dt.minute; X['rth'] = (X.m >= 570) & (X.m < 960)
X['day'] = (X.dt + pd.to_timedelta(np.where(X.m >= 1080, 1, 0), unit='D')).dt.normalize()
wd = X.day.dt.dayofweek; X.loc[wd == 5, 'day'] += pd.Timedelta(days=2); X = X[X.day.dt.dayofweek < 5].reset_index(drop=True)
rs = X[X.rth].groupby('day').sym.first(); roll = (rs != rs.shift(1)); X['roll'] = X.day.map(roll).fillna(False).astype(bool)
X.to_parquet(OUT); r = X[X.rth]
print(f'{len(X)} bars, {X.day.nunique()} days {X.day.min():%Y-%m-%d} - {X.day.max():%Y-%m-%d}; RTH bars per day {len(r) / r.day.nunique():.1f}; roll days {roll.sum()}', flush=True)
print(r.groupby(r.m).volume.mean().sort_values(ascending=False).head(3))
