"""Oct 21 2026: is earnings a separate signal or only useful together with the vector?
Every earnings event in the screener's universe (S&P member then, options ratio >= 1.5 then), 2019-2026:
buy at the close before the gap day G, sell at G's open / G's close / 5 trading days later (close). Split by the vector state at that close:
long = a 15m or 1h vector trade (live rules) is open at that moment; flat = no open trade."""
import glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
T = pd.read_parquet('xs/ext/wide_T.parquet'); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
R = pd.read_csv('xs/ext/optvol_ratio_sp.csv', parse_dates=['date']).pivot_table(index='date', columns='tk', values='r').sort_index()
MED = R.rolling(252, min_periods=60).median().shift(1)
M = json.load(open('sp500_pit_membership.json')); mem = {k: {'META' if t == 'FB' else t.replace('.', '') if t == 'BRK.B' else t for t in v} for k, v in M.items()}; ms = sorted(mem)
def member(tk, d): k = f'{d:%Y-%m}'; k = k if k in mem else (ms[-1] if k > ms[-1] else ms[0]); return tk in mem[k]
TD = pd.DatetimeIndex(sorted(pd.to_datetime(pd.read_parquet('etf_daily/SPY.parquet').datetime).dt.tz_localize(None).dt.normalize().unique()))
spy = pd.read_parquet('etf_daily/SPY.parquet'); spy['d'] = pd.to_datetime(spy.datetime).dt.tz_localize(None).dt.normalize(); spy = spy.drop_duplicates('d').set_index('d')
rows = []
for f in glob.glob('xs/ext/earn/*.csv'):
    tk = f.split('/')[-1][:-4]
    if tk.startswith('_'): continue
    try: e = pd.read_csv(f)
    except Exception: continue
    if not len(e) or tk not in MED.columns: continue
    fb = glob.glob(f'lse_clean/{tk}_15m_2018-01-01_*.parquet')
    if not fb: continue
    B = pd.read_parquet(fb[0], columns=['datetime', 'open', 'close']); B['dt'] = pd.to_datetime(B.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = B.dt.dt.hour * 60 + B.dt.dt.minute; B = B[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt'); B['d'] = B.dt.dt.normalize()
    O = B.groupby('d').open.first(); C = B.groupby('d').close.last(); LASTBAR = B.groupby('d').dt.last()
    tt = pd.to_datetime(e.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    Tk = T[(T.tk == tk) & (T['var'] == 'base+sticky')]
    for t0, s in zip(tt, pd.to_numeric(e['Surprise(%)'], errors='coerce')):
        d = t0.normalize(); G = d if t0.hour < 12 else TD[min(TD.searchsorted(d, 'right'), len(TD) - 1)]
        if G < pd.Timestamp('2019-01-02') or G > pd.Timestamp('2026-10-15'): continue
        k = TD.searchsorted(G); pre = TD[k - 1]; post5 = TD[min(k + 5, len(TD) - 1)]
        if pre not in C.index or G not in O.index or post5 not in C.index: continue
        med = MED[tk].asof(pre)
        if not (med >= 1.5 and member(tk, pre)): continue
        tcl = LASTBAR[pre] + pd.Timedelta('15min')                     # moment of the pre-earnings close
        long = ((Tk.entry <= LASTBAR[pre]) & (Tk.exit > LASTBAR[pre])).any()
        sp_ = lambda a, b: spy.adjclose.asof(b) / spy.adjclose.asof(a) - 1
        rows.append(dict(tk=tk, G=G, long=long, surp=s, gap=O[G] / C[pre] - 1, day=C[G] / C[pre] - 1, d5=C[post5] / C[pre] - 1,
                         spy5=sp_(pre, post5), spy1=sp_(pre, G)))
E = pd.DataFrame(rows); E['per'] = np.where(E.G.dt.year >= 2022, '22-26', '19-21')
E['ex1'] = E.day - E.spy1; E['ex5'] = E.d5 - E.spy5
print(f'{len(E)} earnings events, {E.tk.nunique()} names\n')
def tab(g):
    return pd.Series({'n': len(g), 'gap %': g.gap.mean() * 100, 'G-day %': g.day.mean() * 100, '5d %': g.d5.mean() * 100, '5d vs SPY %': g.ex5.mean() * 100,
                      'win 5d': (g.d5 > 0).mean() * 100, 't (5d ex)': g.ex5.mean() / (g.ex5.std() / np.sqrt(len(g)))})
print('ALL events (no vector):'); print(E.groupby('per').apply(tab).round(2).to_string())
print('\nby vector state at the pre-earnings close:'); print(E.groupby(['long', 'per']).apply(tab).round(2).to_string())
