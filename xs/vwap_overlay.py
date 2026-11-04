"""Nov 4 2026: VWAP overlays on the vector's trades. DECLARED BEFORE RESULTS. 40 matched runs vs live both periods.
 W1 filter: entry open > prior RTH session VWAP | W2 filter: entry open > session-to-date VWAP (04:00-bar start of that day, incl. premarket)
 W3 exit: 15m close < anchored VWAP (from the entry bar), armed after +1% MFE, exit next RTH open | W4 exit: same on 1h closes (4-bar blocks aligned :30)"""
exec(open('xs/ema_common.py').read())
rows = []
for tk, G in XL.groupby('tk'):
    B = BAR.get(tk)
    if B is None: continue
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet'); H = load(f[0]).reset_index(drop=True); tp = ((H.high + H.low + H.close) / 3).values; v = H.volume.values.astype(float); d = H.dt.dt.normalize().values
    ctpv = np.cumsum(tp * v); cv = np.cumsum(v); dfirst = pd.Series(np.arange(len(H))).groupby(d).transform('min').values
    sw = np.where(dfirst > 0, ctpv[dfirst - 1], 0.); sv = np.where(dfirst > 0, cv[dfirst - 1], 0.)
    sess = (ctpv - sw) / np.maximum(cv - sv, 1)                       # session-to-date VWAP incl. current bar
    rth = (H.m.values >= 570) & (H.m.values < 960); pr = pd.Series(np.where(rth, tp * v, 0.)).groupby(d).sum() / pd.Series(np.where(rth, v, 0.)).groupby(d).sum().replace(0, np.nan)
    prior = pd.Series(d).map(pr.shift(1).to_dict()).values
    for i, r in G.iterrows():
        j = B['pos'].get(r.entry)
        if j is None or np.ndim(j) or j < 1: continue
        j = int(j); je = B['pos'].get(r.exit); jx = int(je) if (je is not None and not np.ndim(je)) else None; px = B['o'][j]; off = 0.0
        if jx is not None:
            g = B['o'][jx] / px - 1
            if abs(r.ret - g) < .005: off = r.ret - g
        rec = dict(i=i, W1=px > prior[j] if prior[j] == prior[j] else np.nan, W2=px > sess[j - 1]); end = jx if jx is not None else len(H) - 1
        base_t, base_v = ctpv[j - 1], cv[j - 1]
        for nm in ('W3', 'W4'):
            hit = None; armed = False
            for b in range(j, end):
                av = (ctpv[b - 1 if b > j else j] - base_t) / max(cv[b - 1 if b > j else j] - base_v, 1) if b > j else np.nan
                if b == j or not (570 <= B['m'][b] < 956): continue
                if B['h'][b - 1] / px - 1 >= .01: armed = True
                if not armed: continue
                avp = (ctpv[b - 1] - base_t) / max(cv[b - 1] - base_v, 1)
                cc = B['c'][b - 1] if nm == 'W3' else B['c1'][b - 1]
                if cc < avp: hit = b; break
            if hit is not None: rec[nm + '_x'] = B['t'][hit]; rec[nm + '_r'] = B['o'][hit] / px - 1 + off
        rows.append(rec)
F = pd.DataFrame(rows).set_index('i'); Z = XL.join(F, how='inner'); Z['per'] = np.where(Z.entry.dt.year <= 2021, '19-21', '22-26'); print('joined', len(Z))
for f in ('W1', 'W2'): print(f'  {f} true share {Z[f].mean():.0%};  avg trade true/false: ' + ' | '.join(f'{p} {g[g[f] == True].ret.mean()*100:+.2f}%/{g[g[f] == False].ret.mean()*100:+.2f}%' for p, g in Z.groupby('per')))
B0 = ev(XL); print('\n40 matched runs, 2019-21 / 2022-26'); line('L live', B0)
for nm, col in (('W1 entry > prior-session VWAP', 'W1'), ('W2 entry > session VWAP', 'W2')):
    Y = XL[XL.index.isin(Z[Z[col] == True].index)].reset_index(drop=True); line(nm, ev(Y), B0); print(f'     kept {len(Y)}/{len(XL)}; runners kept {(Y.ret >= .15).sum()}/{(XL.ret >= .15).sum()}', flush=True)
for nm in ('W3', 'W4'):
    Y = XL.copy(); m = Z[nm + '_r'].notna(); idx = m.index[m]; Y.loc[idx, 'exit'] = Z.loc[idx, nm + '_x']; Y.loc[idx, 'ret'] = Z.loc[idx, nm + '_r']
    print(f'  {nm}: shortened {m.sum()} ({m.mean():.0%}), runners hit {(Z[m].ret >= .15).sum()}/{(XL.ret >= .15).sum()}'); line(f'{nm} exit on anchored-VWAP rejection', ev(Y), B0)
