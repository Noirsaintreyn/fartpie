"""Nov 4 2026: four slower / non-price features on the live VECTOR trades (54 names, 17,447 trades). DECLARED BEFORE RESULTS.
All features use only what is known at the entry: prior RTH close for price features; for earnings, the reaction-day OPEN (entries on the reaction day are at/after 09:30).
  E  earnings drift   egap = reaction-day open / prior close - 1 for the last report, if the reaction day was <= 15 sessions before entry; esurp = EPS surprise % (Yahoo) for that report
  R  sector-relative  rel60 = stock 60-session return minus its sector ETF's (cons XLY, tech XLK, semi SMH, indu XLI, comm XLC, fin XLF, stap XLP, energy XLE, hlth XLV); rel20 likewise
  H  52-week high     hi52 = prior close / max close of the last 252 sessions - 1
  O  options ratio    dratio = ratio at entry / ratio ~30 calendar days earlier - 1
PART A: per-trade avg return, win rate, runner rate (ret >= 15%) by feature bucket, 2019-21 / 2022-26.
PART B: book on the 40-run yardstick vs live (6 slots, ratio tie-break). Rank variants drop nothing: K = ratio x (1 + w x good), w = .25 and 1.0. One drop-filter per feature.
PASS = beats live ret/DD in >= 70% of runs in BOTH periods. In-sample; a pass is paper-track only."""
import glob
src = open('xs/stop_1h_ev.py').read(); exec(src[:src.index("B0 = ev(XL)")])
CX = pd.read_parquet('/tmp/claude-0/ml/ctx_rank_X.parquet')[['tk', 'entry', 'h1', 'sec']]
XL = XL.merge(CX.drop_duplicates(['tk', 'entry', 'h1']), on=['tk', 'entry', 'h1'], how='left'); XL['d'] = XL.entry.dt.normalize()
ETF = dict(cons='XLY', tech='XLK', semi='SMH', indu='XLI', comm='XLC', fin='XLF', stap='XLP', energy='XLE', hlth='XLV')
EC = {e: pd.read_parquet(f'/mnt/user-data/uploads/fartpie/etf_daily/{e}_yf.parquet')['Close'].squeeze().rename(e) for e in set(ETF.values())}
feat = []
for tk, g in XL.groupby('tk'):
    H = pd.read_parquet(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')[0]); m = H.datetime.dt.hour * 60 + H.datetime.dt.minute; R = H[(m >= 570) & (m < 960)]
    D = R.groupby(R.datetime.dt.normalize()).agg(o=('open', 'first'), c=('close', 'last')); days = D.index.values
    sec = g.sec.dropna().iloc[0] if g.sec.notna().any() else None; ec = EC.get(ETF.get(sec)) if sec else None
    if ec is not None: ec.index = pd.to_datetime(ec.index).normalize(); ec = ec.reindex(D.index).ffill()
    c = D.c; r60 = c / c.shift(60) - 1; r20 = c / c.shift(20) - 1; hi = c / c.rolling(252, min_periods=200).max() - 1
    e60 = (ec / ec.shift(60) - 1) if ec is not None else r60 * np.nan; e20 = (ec / ec.shift(20) - 1) if ec is not None else r20 * np.nan
    # earnings reaction days
    try: E = pd.read_csv(f'/tmp/claude-0/ml/earn/{tk}.csv')
    except Exception: E = pd.DataFrame(columns=['Earnings Date', 'Surprise(%)'])
    rx = []
    for s_, sp_ in zip(E['Earnings Date'], E['Surprise(%)']):
        ts = pd.Timestamp(str(s_)[:19]); dd = np.datetime64(ts.normalize())
        k = np.searchsorted(days, dd, side='left') if ts.hour < 12 else np.searchsorted(days, dd, side='right')
        if 1 <= k < len(days): rx.append((k, D.o.iloc[k] / c.iloc[k - 1] - 1, float(sp_) if sp_ == sp_ else np.nan))
    rx.sort()
    rk = np.array([a for a, _, _ in rx]); gp = np.array([b for _, b, _ in rx]); su = np.array([s for _, _, s in rx])
    for i, r in g.iterrows():
        j = np.searchsorted(days, np.datetime64(r.d), side='left')                  # index of entry day
        p = j - 1                                                                    # prior completed session
        if p < 1 or j >= len(days): feat.append((i, *[np.nan] * 7)); continue
        q = np.searchsorted(rk, j, side='right') - 1 if len(rk) else -1             # last report whose reaction day <= entry day
        eg = es = np.nan; ds = np.nan
        if q >= 0: ds = j - rk[q]; eg, es = (gp[q], su[q]) if ds <= 15 else (np.nan, np.nan)
        feat.append((i, eg, es, ds, r60.iloc[p] - e60.iloc[p], r20.iloc[p] - e20.iloc[p], hi.iloc[p], np.nan))
F = pd.DataFrame(feat, columns=['i', 'egap', 'esurp', 'dsince', 'rel60', 'rel20', 'hi52', '_']).set_index('i').drop(columns='_')
XL = XL.join(F)
RTs = RT.sort_values('d')
XL = pd.merge_asof(XL.assign(d30=XL.d - pd.Timedelta(days=30)).sort_values('d30'), RTs[['tk', 'd', 'ratio']].rename(columns={'d': 'd30', 'ratio': 'ratio30'}), on='d30', by='tk', direction='backward').sort_values('entry').reset_index(drop=True)
XL['dratio'] = XL.ratio / XL.ratio30 - 1
XL['per'] = np.where(XL.entry.dt.year <= 2021, '19-21', '22-26'); XL['run'] = XL.ret >= .15
print('coverage:', {k: f'{XL[k].notna().mean():.0%}' for k in ('egap', 'esurp', 'rel60', 'hi52', 'dratio')}, ' trades', len(XL), flush=True)
XL.to_parquet('/tmp/claude-0/ml/nonprice_X.parquet')
# ---------------- PART A
def cellA(nm, mask):
    out = f'  {nm:<38}'
    for p in ('19-21', '22-26'):
        z = XL[mask & (XL.per == p)]
        out += f' | {p} n{len(z):>5} avg {z.ret.mean()*100:+.2f}% win {z.ret.gt(0).mean():.0%} runners {z.run.mean()*100:.1f}%'
    print(out, flush=True)
print('\nPART A: per-trade by bucket (all trades: avg, win, runner rate)'); cellA('ALL', XL.ret.notna())
print(' E earnings drift'); cellA('no report in last 15 sessions', XL.egap.isna()); cellA('gap up > +2%', XL.egap > .02); cellA('gap -2..+2%', XL.egap.between(-.02, .02)); cellA('gap down < -2%', XL.egap < -.02)
cellA('surprise > 0', XL.esurp > 0); cellA('surprise <= 0', XL.esurp <= 0)
for col, lab in (('rel60', 'R sector-relative 60d'), ('rel20', 'R sector-relative 20d'), ('hi52', 'H distance to 52w high'), ('dratio', 'O options ratio 30d change')):
    q = XL.groupby(XL.entry.dt.year)[col].transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False) if s.notna().sum() > 30 else s * np.nan)
    print(f' {lab} (terciles within year)')
    for k, nm in ((0, 'low'), (1, 'mid'), (2, 'high')): cellA(f'{nm} tercile', q == k)
cellA('H within 5% of 52w high', XL.hi52 >= -.05); cellA('H more than 25% below', XL.hi52 < -.25)
# ---------------- PART B
B0 = ev(XL); print('\nPART B: book, 40 matched runs, 2019-21 / 2022-26'); line('L live', B0)
def tryk(nm, good, w):
    Y = XL.copy(); Y['ratio'] = XL.ratio * (1 + w * good.fillna(False).astype(float).values); line(nm, ev(Y), B0)
def tryf(nm, keep):
    Y = XL[keep.fillna(True).values].copy(); line(nm + f' (drops {(~keep.fillna(True)).mean():.0%})', ev(Y), B0)
gE = (XL.egap > .02); gS = (XL.esurp > 0); gR = (XL.rel60 > 0); gH = (XL.hi52 >= -.05); gO = (XL.dratio > .10)
for nm, gd in (('E1 post-earnings gap up', gE), ('E2 positive surprise', gS), ('R1 beats sector 60d', gR), ('H1 within 5% of 52w high', gH), ('O1 ratio up 10%+ in 30d', gO)):
    tryk(nm + ' w.25', gd, .25); tryk(nm + ' w1.0', gd, 1.0)
tryf('E3 skip 15d after gap down', ~(XL.egap < -.02)); tryf('R3 skip worst-tercile rel60', ~(XL.groupby(XL.entry.dt.year).rel60.transform(lambda s: s.rank(pct=True)) < 1 / 3))
tryf('H3 skip >25% below 52w high', ~(XL.hi52 < -.25)); tryk('O3 REVERSE ratio falling w.25', XL.dratio < -.10, .25)
n = gE.astype(int) + gR.astype(int) + gH.astype(int) + gO.astype(int); Y = XL.copy(); Y['ratio'] = XL.ratio * (1 + .25 * n.values); line('C1 combo ratio x (1+.25 n good)', ev(Y), B0)
