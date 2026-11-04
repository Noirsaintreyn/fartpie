"""Nov 4 2026: daily and 4h SMA crosses added to the VECTOR's trades (55 names with bars). DECLARED BEFORE RESULTS. Book test: 40 matched runs vs live, 2019-21 / 2022-26.
State at the last COMPLETED bar before the entry (daily known 16:00, 4h known at its close; 4h bars 04-08, 08-12, 12-16 ET):
  D50  daily SMA50 > SMA200 (golden cross state)   D20  daily SMA20 > SMA50   H20  4h SMA20 > SMA50   H50  4h SMA50 > SMA200
Entry filters (drop trades):  D50 | D20 | H20 | H50 | D20&H20 | D50&H50 | any-of (D20|H20)
Rank (no trades dropped, tie-break replaced): K = number of bull states among {D50, D20, H20, H50} * 100 + options ratio
Exits (replace the live exit if first; next RTH bar open):  XD daily SMA20 < SMA50 | XH 4h SMA20 < SMA50 (a cross-down while holding)"""
exec(open('xs/ema_common.py').read())
def add_sma(tk):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')[0]; H = load(f).reset_index(drop=True); B = BAR[tk]; t = H.dt.values
    x = H.set_index('dt')[['open', 'high', 'low', 'close']]; r4 = x.resample('4h', offset='4h').agg(dict(open='first', high='max', low='min', close='last')).dropna()
    for n in (20, 50, 200): r4[f's{n}'] = r4.close.rolling(n).mean()
    r4['H20'] = (r4.s20 > r4.s50).astype(float).where(r4.s50.notna()); r4['H50'] = (r4.s50 > r4.s200).astype(float).where(r4.s200.notna()); r4['known'] = r4.index + pd.Timedelta('4h')
    K = pd.merge_asof(pd.DataFrame({'q': t + np.timedelta64(15, 'm')}), r4[['known', 'H20', 'H50']].sort_values('known'), left_on='q', right_on='known', direction='backward'); B['H20'], B['H50'] = K.H20.values, K.H50.values
    R = H[(H.m >= 570) & (H.m < 960)]; dd = R.groupby(R.dt.dt.normalize()).agg(c=('close', 'last'))
    for n in (20, 50, 200): dd[f's{n}'] = dd.c.rolling(n).mean()
    dd['D20'] = (dd.s20 > dd.s50).astype(float).where(dd.s50.notna()); dd['D50'] = (dd.s50 > dd.s200).astype(float).where(dd.s200.notna()); dd['known'] = dd.index + pd.Timedelta(hours=16)
    D = pd.merge_asof(pd.DataFrame({'q': t + np.timedelta64(15, 'm')}), dd.reset_index(drop=True).sort_values('known'), left_on='q', right_on='known', direction='backward'); B['D20'], B['D50'] = D.D20.values, D.D50.values
for tk in BAR: add_sma(tk)
rows = []
for i, r in XL.iterrows():
    B = BAR.get(r.tk)
    if B is None: continue
    j = B['pos'].get(r.entry)
    if j is None or np.ndim(j) or j < 1: continue
    j = int(j); je = B['pos'].get(r.exit); jx = int(je) if (je is not None and not np.ndim(je)) else None; px = B['o'][j]; off = 0.0
    if jx is not None:
        g = B['o'][jx] / px - 1
        if abs(r.ret - g) < .005: off = r.ret - g
    rec = dict(i=i, **{k: B[k][j - 1] for k in ('D50', 'D20', 'H20', 'H50')}); end = (jx if jx is not None else len(B['o']) - 1)
    for nm, key in (('XD', 'D20'), ('XH', 'H20')):
        for b in range(j + 1, end):
            if B['m'][b] < 570 or B['m'][b] >= 956: continue
            if B[key][b - 1] == 0.0 and B[key][j - 1] == 1.0: rec[nm + '_x'] = B['t'][b]; rec[nm + '_r'] = B['o'][b] / px - 1 + off; break
    rows.append(rec)
F = pd.DataFrame(rows).set_index('i'); Z = XL.join(F, how='inner'); Z['per'] = np.where(Z.entry.dt.year <= 2021, '19-21', '22-26'); print('joined', len(Z), 'of', len(XL))
for f in ('D50', 'D20', 'H20', 'H50'):
    print(f'  {f}: share bull {Z[f].mean():.0%};  avg trade bull/bear: ' + ' | '.join(f'{p} {g[g[f] == 1].ret.mean()*100:+.2f}%/{g[g[f] == 0].ret.mean()*100:+.2f}% (runners {(g[g[f]==1].ret>=.15).mean()*100:.1f}%/{(g[g[f]==0].ret>=.15).mean()*100:.1f}%)' for p, g in Z.groupby('per')))
B0 = ev(XL); print('\n40 matched runs, 2019-21 / 2022-26'); line('L live', B0)
def drop(nm, keep):
    Y = XL[XL.index.isin(Z[keep(Z)].index)].reset_index(drop=True); line(nm, ev(Y), B0); print(f'     kept {len(Y)}/{len(XL)}; runners kept {(Y.ret >= .15).sum()}/{(XL.ret >= .15).sum()}', flush=True)
drop('D50 daily SMA50>200', lambda z: z.D50 == 1); drop('D20 daily SMA20>50', lambda z: z.D20 == 1); drop('H20 4h SMA20>50', lambda z: z.H20 == 1); drop('H50 4h SMA50>200', lambda z: z.H50 == 1)
drop('D20 & H20', lambda z: (z.D20 == 1) & (z.H20 == 1)); drop('D50 & H50', lambda z: (z.D50 == 1) & (z.H50 == 1)); drop('D20 or H20', lambda z: (z.D20 == 1) | (z.H20 == 1))
Y = XL.copy(); n = Z[['D50', 'D20', 'H20', 'H50']].fillna(0).sum(axis=1); Y.loc[Z.index, 'ratio'] = Z.ratio + 100 * n; print('\nRANK by SMA states then ratio (no drops)'); line('rank: states*100 + ratio', ev(Y), B0)
Y = XL.copy(); n = Z[['D20', 'H20']].fillna(0).sum(axis=1); Y.loc[Z.index, 'ratio'] = Z.ratio + 100 * n; line('rank: (D20,H20 count)*100 + ratio', ev(Y), B0)
for nm in ('XD', 'XH'):
    Y = XL.copy(); m = Z[nm + '_r'].notna(); idx = m.index[m]; Y.loc[idx, 'exit'] = Z.loc[idx, nm + '_x']; Y.loc[idx, 'ret'] = Z.loc[idx, nm + '_r']
    print(f'  {nm}: shortened {m.sum()} ({m.mean():.0%}), runners hit {(Z[m].ret >= .15).sum()}/{(XL.ret >= .15).sum()}'); line(f'{nm} exit on SMA cross-down', ev(Y), B0)
