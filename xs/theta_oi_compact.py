"""Nov 24 2026: keep the per-strike OPEN INTEREST rows (kind=='oi') that xs/theta_compact.py dropped, so SPY/IWM/DIA/QQQ can get a real OI-based net gamma like xs/ext/gex/QQQ.csv.
Run from the repo root on the machine that has data/theta/<TK>/:  python3 xs/theta_oi_compact.py SPY IWM DIA QQQ
Writes data/theta_oi/<TK>.parquet: d (trade date), e (expiry), K, r (+1 call / -1 put), oi, dte (0-70 days, strikes any)."""
import sys, glob, os, numpy as np, pandas as pd
os.makedirs('data/theta_oi', exist_ok=True)
for tk in sys.argv[1:]:
    parts = []
    for f in sorted(glob.glob(f'data/theta/{tk}/{tk}_*.parquet')):
        x = pd.read_parquet(f); x = x[x.kind == 'oi']
        if not len(x): continue
        col = next((c for c in ('open_interest', 'oi', 'volume', 'size', 'count') if c in x.columns and x[c].notna().any()), None)
        if col is None: print(tk, f, 'no OI column in', list(x.columns)); sys.exit(1)
        d = pd.to_datetime(x.created.astype(str).str[:10]); e = pd.to_datetime(x.expiration); dte = (e - d).dt.days
        y = pd.DataFrame({'d': d.values, 'e': e.values, 'K': x.strike.astype('float32').values, 'r': np.where(x.right.astype(str).str.upper().str[0] == 'C', 1, -1).astype('int8'), 'oi': x[col].astype('float32').values, 'dte': dte.astype('int16').values})
        parts.append(y[(y.dte >= 0) & (y.dte <= 70)])
    if not parts: print(tk, 'no oi rows'); continue
    X = pd.concat(parts, ignore_index=True).drop_duplicates(['d', 'e', 'K', 'r'], keep='last'); X.to_parquet(f'data/theta_oi/{tk}.parquet', index=False); print(tk, len(X), X.d.min().date(), X.d.max().date(), 'oi col', col, flush=True)
