"""Oct 25 2026: sector ETF vector on 1h / 4h (yfinance 730d) / daily / weekly vs screener flips."""
import sys; sys.path.insert(0, 'xs')
exec(open('xs/sector_vector.py').read().split("V = {t: dvec")[0])
import yfinance as yf
def vec_on(x):
    x = x.reset_index(drop=True); reg, *_ = run_state_machine(x[['open', 'high', 'low', 'close']].assign(datetime=x.dt)); return pd.Series(np.asarray(reg), index=x.dt)
def daily(t):
    x = pd.read_parquet(f'etf_daily/{t}.parquet'); x['dt'] = pd.to_datetime(x.datetime).dt.tz_localize(None).dt.normalize(); return x.drop_duplicates('dt').sort_values('dt')
ETFS = sorted(set(EM.values()))
S = {}
for t in ETFS:
    d = daily(t); s_d = vec_on(d)
    w = d.set_index('dt').resample('W-FRI').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()
    s_w = vec_on(w); s_w.index = s_w.index + pd.Timedelta(days=3)                    # known after Friday's close -> from Monday
    S[(t, 'daily')] = s_d.shift(1); S[(t, 'weekly')] = s_w.shift(0)
    try:
        h = yf.download(t, period='730d', interval='1h', progress=False, auto_adjust=True)
        h.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in h.columns]
        h = h.reset_index(); h['dt'] = pd.to_datetime(h.iloc[:, 0]).dt.tz_convert('America/New_York').dt.tz_localize(None)
        s_h = vec_on(h[['dt', 'open', 'high', 'low', 'close']]); s_h.index = s_h.index + pd.Timedelta('1h'); S[(t, '1h')] = s_h
        h4 = h.set_index('dt')[['open', 'high', 'low', 'close']].resample('4h', offset='9h30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()
        s_4 = vec_on(h4); s_4.index = s_4.index + pd.Timedelta('4h'); S[(t, '4h')] = s_4
    except Exception as ex: print(t, 'hourly failed', ex)
T['etf'] = [EX.get(t) or EM.get(SEC.get(t, ''), None) for t in T.tk]
rows = []
for tf in ('1h', '4h', 'daily', 'weekly'):
    st = []
    for e, x in zip(T.etf, T.entry):
        s = S.get((e, tf)); st.append(s.asof(x if tf in ('1h', '4h') else x.normalize()) if s is not None else np.nan)
    T['s_' + tf] = st
H2 = T[T.entry >= '2024-11-01']
print('avg trade % by the sector ETF vector at entry (bull / bear), and the gap:')
for tf in ('1h', '4h', 'daily', 'weekly'):
    for nm, Z in (('2019-21', T[T.entry.dt.year < 2022]), ('2022-26', T[T.entry.dt.year >= 2022]), ('Nov24-26', H2)):
        b, br = Z[Z['s_' + tf] == 1].ret.mean() * 100, Z[Z['s_' + tf] == -1].ret.mean() * 100
        n = Z['s_' + tf].notna().sum()
        if n > 200: print(f'  {tf:<7}{nm:<9} bull {b:+.3f}%  bear {br:+.3f}%  edge {b - br:+.3f}  (n {n}, bull share {(Z["s_" + tf] == 1).mean():.0%})')
print('\nbook, Nov 2024 - Oct 2026 (hourly data window), skip flips when the sector ETF vector is bear:')
P2 = [(np.datetime64('2024-11-01'), np.datetime64('2027-01-01'))]
def sh(nm, Z): print(f'  {nm:<36}' + ''.join(f'{c:>+9.1%} /{dd:>5.0%}' for c, dd in (sim(Z, a, b) for a, b in P2)))
sh('live', T)
for tf in ('1h', '4h', 'daily', 'weekly'): sh(f'skip if {tf} sector vector bear', T[T['s_' + tf] != -1])
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
print('weekly sector vector, full period:')
def show(nm, Z, **kw): print(f'  {nm:<50}' + ''.join(f'{c:>+9.1%} /{dd:>5.0%}' for c, dd in (sim(Z, a, b, **kw) for a, b in P)), flush=True)
show('live', T)
T['kw'] = (T.s_weekly == -1) * 10 + T.ratio / 100; show('tie-break: weekly-sector-BEAR flips first', T, key='kw')
show('only flips whose sector weekly vector is bear', T[T.s_weekly == -1])
