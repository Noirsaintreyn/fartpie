"""Nov 14 2026 SEC / FINRA / FRED FILTERS ON SMALL/MID FLIPS (declared before results). Data: SEC Form 4 bulk sets (open-market purchases 'P' / sales 'S' by officers+directors; NOTE bulk files for 2026 Q2+ are essentially empty so insider tests stop at 2026-03-31), SEC submissions (offering/shelf forms 424B5/424B4/S-3/S-1; late-filing notices NT 10-K/10-Q), FINRA bi-monthly short interest (used only after settlement+8 days = publication lag), FRED daily macro (HY OAS, VIX, 10y-2y, USD, NFCI; used lagged 1 day). Hypotheses fixed before results: H1 insider cluster buying (>=1 officer/director purchase in prior 30d; cluster >=2 owners) -> BETTER long flips (strongest in small caps); H2 offering/shelf filing in prior 10 trading days -> WORSE; H3 late-filing notice in prior 30d -> WORSE; H4 high days-to-cover (top third) -> two-sided (squeeze vs overhang), judged on halves; H5 HY OAS falling over 20d (risk-on) -> BETTER small-cap longs; high VIX -> two-sided. Long flips (shorts also shown for the macro/SI features), Apr 2025-Nov 2026, net of 15bp, halves, PASS = same sign both halves and pooled |t|>=2."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = SC + 'ext/'
src = open('/home/claude/fartpie/xs/report_56_pead_flips.py').read(); exec(src.split('# ---- large caps')[0]); exec('def tag_pre' + src.split('def tag_pre')[1].split('def show_pre')[0])
P = pd.read_pickle(SC + 'part_small.pkl'); D = pd.read_pickle('live/midsmall_bars.pkl')['d']; dc = {tk: B.close.groupby(B.index.normalize()).last() for tk, B in D.items()}
K = pd.read_csv('/home/claude/fartpie/xs/ext/8k_items_midsmall.csv'); K['items'] = K['items'].astype(str); ev = events(K, dc); P = tag_pre(tag(P, ev, dc), ev, dc); P['net'] = P.ret - 0.0015; P = P.sort_values('t').reset_index(drop=True); P['td'] = P.t.dt.normalize()
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 5 and len(b) > 5 else np.nan
# ---- insider
T = pd.read_pickle(X + 'insider.pkl'); T = T[(T.TRANS_CODE == 'P') & (T.TRANS_ACQUIRED_DISP_CD == 'A') & T.ISSUERTRADINGSYMBOL.notna()].copy(); T['fd'] = pd.to_datetime(T.FILING_DATE, errors='coerce'); T['sym'] = T.ISSUERTRADINGSYMBOL.str.upper(); T['off'] = T.RPTOWNER_RELATIONSHIP.fillna('').str.contains('Officer|Director', case=False)
T = T[T.off & T.fd.notna()]; P['INS'] = False; P['CLU'] = False
for tk, ix in P.groupby('tk').groups.items():
    x = T[T.sym == tk.upper()].sort_values('fd')
    if x.empty: continue
    f = x.fd.values; o = x.RPTOWNERCIK.values; t = P.loc[ix, 'td'].values
    for j, i in enumerate(ix):
        m = (f <= t[j]) & (f > t[j] - np.timedelta64(30, 'D'))
        if m.any(): P.loc[i, 'INS'] = True; P.loc[i, 'CLU'] = len(set(o[m])) >= 2
# ---- forms
Fm = pd.read_csv(X + 'forms.csv'); Fm['acc'] = pd.to_datetime(Fm.accept, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
def formflag(forms, days):
    per = {tk: x.acc.sort_values().values for tk, x in Fm[Fm.form.isin(forms)].groupby('ticker')}; flag = np.zeros(len(P), bool)
    for tk, ix in P.groupby('tk').groups.items():
        a = per.get(tk)
        if a is None: continue
        t = P.loc[ix, 't'].values; pos = np.searchsorted(a, t, side='right') - 1; ok = pos >= 0; last = a[np.clip(pos, 0, None)]; flag[P.index.get_indexer(ix)] = ok & (last >= t - np.timedelta64(days, 'D'))
    return flag
P['OFFER'] = formflag(['424B5', '424B4', 'S-3', 'S-1', 'S-3ASR', '424B3'], 14); P['LATE'] = formflag(['NT 10-K', 'NT 10-Q'], 30)
# ---- short interest
SI = pd.read_pickle(X + 'shortint.pkl'); SI['sd'] = pd.to_datetime(SI.settlementDate.astype(str)); SI['pub'] = SI.sd + pd.Timedelta(days=8); SI = SI.rename(columns={'symbolCode': 'tk'})[['tk', 'pub', 'daysToCoverQuantity', 'changePercent']].sort_values('pub')
P['td'] = P.td.astype('datetime64[ns]'); SI['pub'] = SI.pub.astype('datetime64[ns]'); P['tk'] = P.tk.astype(str); P = pd.merge_asof(P.sort_values('td'), SI.assign(tk=SI.tk.str.upper()), left_on='td', right_on='pub', by='tk', direction='backward').sort_values('t').reset_index(drop=True); P['DTC'] = pd.to_numeric(P.daysToCoverQuantity, errors='coerce'); P['SIchg'] = pd.to_numeric(P.changePercent, errors='coerce')
# ---- fred
Fr = pd.read_pickle(X + 'fred.pkl').set_index('date').ffill(); Fr = Fr.shift(1)
for c in ('HY_OAS', 'VIX', 'T10Y2Y', 'USD', 'NFCI'): P[c] = Fr[c].reindex(P.td).values; P[c + '_d20'] = (Fr[c] - Fr[c].shift(20)).reindex(P.td).values
def pit(c):
    v = P[c].values; o = np.full(len(v), np.nan)
    for i in range(300, len(v)):
        h = v[:i]; h = h[~np.isnan(h)]
        if not np.isnan(v[i]) and len(h) > 50: q1, q2 = np.quantile(h, [1 / 3, 2 / 3]); o[i] = 0 if v[i] <= q1 else 2 if v[i] >= q2 else 1
    return o
mid = pd.Timestamp('2026-03-01'); END_INS = pd.Timestamp('2026-04-01')
def line(name, S, on, off):
    out = f' {name:30s}'
    for lab, m in (('h1', S.t < mid), ('h2', S.t >= mid), ('all', S.t >= S.t.min())):
        a = S[m & on]; b = S[m & off]; out += f' | {lab}: {a.net.mean()*100:+.2f} (n {len(a)}) vs {b.net.mean()*100:+.2f} t {tt(a.net, b.net):+.1f}'
    print(out)
for d, nm in ((1, 'LONG'), (-1, 'SHORT')):
    for sub, mk in (('ALL FLIPS', lambda S: np.ones(len(S), bool)), ('BV>=1.5', lambda S: (S.BV >= 1.5).values)):
        S = P[P.d == d].copy().reset_index(drop=True); m = pd.Series(mk(S)); print(f'\n[{nm} | {sub}]  feature on vs off: net mean %, n, t')
        if d == 1:
            Si = S[S.t < END_INS]; mi = m[S.t < END_INS].values; Si = Si.reset_index(drop=True); mi = pd.Series(mi)
            line('INSIDER buy 30d (<=Mar26)', Si, mi & Si.INS, mi & ~Si.INS); line('INSIDER cluster>=2 (<=Mar26)', Si, mi & Si.CLU, mi & ~Si.CLU)
        line('OFFERING/shelf filed <=14d', S, m & S.OFFER, m & ~S.OFFER); line('late-filing notice <=30d', S, m & S.LATE, m & ~S.LATE)
        S['T_DTC'] = np.nan; v = S.DTC.values; o = np.full(len(v), np.nan)
        for i in range(300, len(v)):
            h = v[:i]; h = h[~np.isnan(h)]
            if not np.isnan(v[i]) and len(h) > 50: q1, q2 = np.quantile(h, [1 / 3, 2 / 3]); o[i] = 0 if v[i] <= q1 else 2 if v[i] >= q2 else 1
        S['T_DTC'] = o; line('days-to-cover top 3rd vs low', S, m & (S.T_DTC == 2), m & (S.T_DTC == 0))
        for c in ('HY_OAS_d20', 'HY_OAS', 'VIX', 'T10Y2Y_d20', 'USD_d20', 'NFCI'):
            v = S[c].values; o = np.full(len(v), np.nan)
            for i in range(300, len(v)):
                h = v[:i]; h = h[~np.isnan(h)]
                if not np.isnan(v[i]) and len(h) > 50: q1, q2 = np.quantile(h, [1 / 3, 2 / 3]); o[i] = 0 if v[i] <= q1 else 2 if v[i] >= q2 else 1
            S['T_' + c] = o; line(f'{c} high 3rd vs low 3rd', S, m & (S['T_' + c] == 2), m & (S['T_' + c] == 0))
P.to_pickle(SC + 'part_small_alt.pkl')
