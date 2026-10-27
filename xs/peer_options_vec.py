"""Oct 26 2026: two new inputs for the vector, measured on every live trade (known at the prior close).
 1 PEER CONTAGION  the daily vector (on daily closes) run on every S&P name; for each trade: its industry peers (Yahoo industry,
   point-in-time S&P members, sector if < 3 peers) -> share of peers in a bull trend, peers that flipped bull in the last 3 days,
   and leader / laggard: did the stock's own daily flip come before or after most of its bull peers' flips
 2 OPTIONS-FLOW VECTOR  the vector state machine run on each name's daily option volume, call share and ATM implied vol
   (38 names with daily option summaries): is option activity in an uptrend (building) or downtrend, is IV trending up
python3 xs/peer_options_vec.py EXT_DIR PIT_JSON SUBIND_JSON TRADES_GLOB OUT_FILE"""
import sys, os, glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); A_ = sys.argv[1:]; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
EXT, PIT, SUB, TG, OUTF = A_
P = pd.read_parquet(f'{EXT}/close_px_sp.parquet').pivot_table(index='date', columns='ticker', values='close').sort_index()
P = P[P.index >= '2016-06-01']
M = json.load(open(PIT)); mem = pd.DataFrame(False, index=P.index, columns=P.columns)
for ym, tks in M.items(): mem.loc[P.index.strftime('%Y-%m') == ym, [t for t in tks if t in P.columns]] = True
mem.loc[P.index.strftime('%Y-%m') < min(M), [t for t in M[min(M)] if t in P.columns]] = True
SI = json.load(open(SUB)); ind = {t: v.get('industry') for t, v in SI.items()}; sec = {t: v.get('sector') for t, v in SI.items()}
def cvec(s):                                    # vector on a close-only series (open = high = low = close)
    s = s.dropna(); 
    if len(s) < 80: return pd.Series(dtype=float)
    d = pd.DataFrame({'open': s.values, 'high': s.values, 'low': s.values, 'close': s.values}); return pd.Series(vec(d), index=s.index)
ST = pd.DataFrame({t: cvec(P[t]) for t in P.columns}).reindex(P.index); print('daily vectors', ST.shape, flush=True)
FL = (ST == 1) & (ST.shift(1) != 1)
lastflip = pd.DataFrame(np.where(FL, np.arange(len(P))[:, None], np.nan), index=P.index, columns=P.columns).ffill()
T = pd.concat(pd.read_parquet(f) for f in glob.glob(TG)); T = T[T['var'] == 'vec'][['tk', 'h1', 'entry', 'ret']]
days = P.index.values; rows = []
groups = {}
for t in P.columns: groups.setdefault(ind.get(t), []).append(t)
sgroups = {}
for t in P.columns: sgroups.setdefault(sec.get(t), []).append(t)
# options
O = {os.path.basename(f)[:-4]: pd.read_csv(f, parse_dates=['date']).set_index('date') for f in glob.glob(f'{EXT}/opt_summary/*.csv')}
OS = {}
for t, d in O.items():
    d = d[~d.index.duplicated()]; vol = (d.call_vol + d.put_vol).replace(0, np.nan)
    OS[t] = pd.DataFrame({'ov': cvec(np.log(vol).rolling(5).mean()), 'cs': cvec((d.call_vol / (d.call_vol + d.put_vol)).rolling(5).mean()), 'iv': cvec(d.atm_iv)})
for t_ in T.itertuples():
    e = pd.Timestamp(t_.entry); k = np.searchsorted(days, np.datetime64(e.normalize())) - 1     # prior session
    if k < 1 or t_.tk not in P.columns: rows.append((t_.tk, t_.h1, t_.entry) + (np.nan,) * 8); continue
    peers = [p for p in groups.get(ind.get(t_.tk), []) if p != t_.tk and mem.iloc[k].get(p, False)]
    if len(peers) < 3: peers = [p for p in sgroups.get(sec.get(t_.tk), []) if p != t_.tk and mem.iloc[k].get(p, False)]
    st = ST.iloc[k][peers].dropna(); share = (st == 1).mean() if len(st) else np.nan
    recent = ((k - lastflip.iloc[k][peers]) <= 2).sum() if len(peers) else np.nan
    own = lastflip.iloc[k].get(t_.tk, np.nan); bp = lastflip.iloc[k][peers][st.reindex(peers).values == 1] if len(st) else pd.Series(dtype=float)
    lead = np.nan if (own != own or not len(bp)) else float((bp > own).mean())        # share of bull peers that flipped AFTER the stock (1 = leader)
    o = OS.get(t_.tk); ov = cs = iv = np.nan
    if o is not None:
        z = o.loc[:days[k]]
        if len(z): ov, cs, iv = z.iloc[-1].values
    rows.append((t_.tk, t_.h1, t_.entry, share, recent, lead, len(peers), np.nan_to_num(ST.iloc[k].get(t_.tk, 0)), ov, cs, iv))
pd.DataFrame(rows, columns=['tk', 'h1', 'entry', 'peer_bull', 'peer_new', 'lead', 'npeers', 'own_d', 'opt_vol_trend', 'call_share_trend', 'iv_trend']).to_parquet(OUTF); print('ok', len(rows))
