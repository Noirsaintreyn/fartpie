"""Nov 8 2026: daily correlation of the NQ / ES noise-momentum and euro / bond trend-hold streams from xs/fut_assets15.py, and combined Sharpe; data-density check."""
_f = open('/home/claude/fartpie/xs/fut_assets15.py').read(); exec(_f[:_f.rindex("if __name__ == '__main__':")])
P = {}
for s in ('QQQ', 'SPY', 'FXE', 'TLT'):
    H, D = prep(s); days = pd.DatetimeIndex(D.index[D.adr.notna()]); T = noise(H, D) if s in ('QQQ', 'SPY') else vector(H, 'B'); P[s] = T.groupby('day').R.sum().reindex(days).fillna(0.0)
    raw = load(glob.glob(f'{ROOT}/{s}_15m_*.parquet')[0]); raw = raw[(raw.m >= 570) & (raw.m < 960)]
    print(s, 'days kept', len(days), 'of', raw.dt.dt.normalize().nunique(), '| RTH bars per day', round(len(raw) / raw.dt.dt.normalize().nunique(), 1), flush=True)
A = pd.concat(P, axis=1).dropna(); print('daily correlation:\n', A[['QQQ', 'SPY', 'FXE', 'TLT']].corr().round(2))
sh = lambda x: x.mean() / x.std() * np.sqrt(252)
print(f"NQ noise alone Sharpe {sh(A.QQQ):.2f} | NQ + ES noise (equal R) {sh(A.QQQ + A.SPY):.2f} | NQ noise + Euro trend-hold {sh(A.QQQ + A.FXE):.2f} | NQ + ES + Euro + Bonds {sh(A[['QQQ', 'SPY', 'FXE', 'TLT']].sum(axis=1)):.2f}")
