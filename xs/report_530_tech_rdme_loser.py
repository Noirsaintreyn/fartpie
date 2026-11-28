"""Report 530 (Nov 28 2026): TECH lead from 529 (RDME_hi in past losers: +6.8pp t 2.6, 3/3 eras) -- Rey's attachment #9 asks: robustness, sub-industry heterogeneity, triple-sort cell, flip tag.
DECLARED BEFORE RESULTS. Universe TECH (SIC 35/36/73/48/37) sub-$5.
 A  loser definitions: DD252 below train median (529) vs 60d return below train median (reversal horizon check; Yu 2016 says sign may flip with horizon).
 B  sub-industry: RDME_hi lift within losers by SIC2 (35 hardware, 36 electronics/semis, 73 software/services, 48 comms, 37 transport equip) -- Yang 2025: heterogeneous.
 C  triple cell: RDME_hi & loser & small (MCAP below train median) vs rest of TECH, template lift.
 D  flip tag: TECH flips (350): RDME_hi (train tertile) & loser vs rest; reach/mean/median by era, test 2022+, permutation 5000 vs random same-size TECH flips, x4 trials (A, B pooled, C, D).
PASS = stock-day pooled |t|>=3 & same sign >=3/4 eras; flip PASS = permutation p*4<0.05 on reach AND train/test same sign.
"""
import numpy as np, pandas as pd, json, io, contextlib, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_516_sector.py').read().split("TRm = (P.yr")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
P['pb'] = np.select([P.c < 1, P.c < 2.5], [0, 1], 2); P['era4'] = np.select([P.yr == 2018, P.yr <= 2021, P.yr == 2022], ['2018', '2019-21', '2022'], '2023-26')
DDp = pd.read_pickle(X + 'runs515_panel.pkl')[['tk', 'd', 'DD252']]; DDp['d'] = DDp.d.astype('datetime64[ns]'); P['d'] = P.d.astype('datetime64[ns]'); P = P.merge(DDp, on=['tk', 'd'], how='left')
S = pd.read_csv(X + 'sic.csv'); S.columns = [c.lower() for c in S.columns]; scol = [c for c in S.columns if 'sic' in c][0]; tcol = 'tk' if 'tk' in S.columns else S.columns[0]; S['sic2'] = (pd.to_numeric(S[scol], errors='coerce') // 100).astype('Int64'); sic2 = S.set_index(tcol).sic2.to_dict()
sh = pd.read_pickle(X + 'shares_pit.pkl'); sh['filed'] = pd.to_datetime(sh.filed).astype('datetime64[ns]'); sh = sh.sort_values('filed')
rd = []
for tk in P[P.sector == 'TECH'].tk.unique():
    try: gg = json.load(open(f'data/fund/sec_small/{tk}.json'))['facts']['us-gaap']
    except Exception: continue
    a = ann(gg, ['ResearchAndDevelopmentExpense'], 0)
    if a is None: continue
    e = pd.DataFrame({'RDX': a.val, 'frd': a.filed}); e['tk'] = tk; rd.append(e.dropna(subset=['frd']))
RDX = pd.concat(rd).sort_values('frd'); RDX['frd'] = RDX.frd.astype('datetime64[ns]')
def build(G, dcol='d', pxcol='c'):
    G = G.copy(); G[dcol] = G[dcol].astype('datetime64[ns]')
    G = pd.merge_asof(G.sort_values(dcol), sh[['tk', 'filed', 'val']].rename(columns={'filed': 'fs', 'val': 'SH'}), left_on=dcol, right_on='fs', by='tk', direction='backward', tolerance=pd.Timedelta(days=730))
    G = pd.merge_asof(G.sort_values(dcol), RDX.rename(columns={'frd': 'fr'}), left_on=dcol, right_on='fr', by='tk', direction='backward', tolerance=pd.Timedelta(days=540)); G['MCAP'] = G.SH * G[pxcol]; G['RDME'] = G.RDX / G.MCAP; return G.reset_index(drop=True)
def lift2(Gx, col):
    d = Gx[Gx[col].isin([0, 1])]; r = []
    for (m, b, p), gp in d.groupby(['mon', 'ab10', 'pb']):
        a, c = gp[gp[col] == 1].y, gp[gp[col] == 0].y
        if len(a) >= 3 and len(c) >= 3: r.append((m, a.mean() - c.mean(), len(a)))
    if len(r) < 4: return np.nan, np.nan
    o = pd.DataFrame(r, columns=['m', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); s = mm.std(); return mm.mean(), (mm.mean() / (s / np.sqrt(len(mm))) if s and s > 0 else np.nan)
def rowof(lab, Gx, col):
    r = dict(cell=lab, n=int((Gx[col] == 1).sum()))
    for e in ('2018', '2019-21', '2022', '2023-26'): d, t = lift2(Gx[Gx.era4 == e], col); r[e] = None if d != d else round(d * 100, 1)
    d, t = lift2(Gx, col); r['pooled_pp'] = None if d != d else round(d * 100, 1); r['t'] = None if t != t else round(t, 1); return r
G = P[P.sector == 'TECH'].copy(); G['ab10'] = pd.qcut(G.ATRP.rank(method='first'), 10, labels=False); G = build(G); tm = G.yr.between(2019, 2021)
q = G.loc[tm, 'RDME'].quantile([.33, .67]); G['hi'] = np.where(G.RDME > q.iloc[1], 1, np.where(G.RDME <= q.iloc[0], 0, np.nan))
G['R60'] = G.groupby('tk').c.transform(lambda s: s / s.shift(60) - 1) if 'R60' not in G else G.R60
G['L_dd'] = np.where(G.DD252 < G.loc[tm, 'DD252'].median(), 1, np.where(G.DD252.notna(), 0, np.nan)); G['L_60'] = np.where(G.R60 < G.loc[tm, 'R60'].median(), 1, np.where(G.R60.notna(), 0, np.nan)); G['SM'] = np.where(G.MCAP < G.loc[tm, 'MCAP'].median(), 1, np.where(G.MCAP.notna(), 0, np.nan)); G['s2'] = G.tk.map(sic2)
print('TECH stock-days', len(G), 'RDME cov', round(G.RDME.notna().mean(), 2), 'SIC2 cov', round(G.s2.notna().mean(), 2), G.s2.value_counts().head(6).to_dict()); pd.set_option('display.width', 220); rows = []
rows.append(rowof('RDMEhi | loser(DD252)', G[G.L_dd == 1], 'hi')); rows.append(rowof('RDMEhi | loser(60d ret)', G[G.L_60 == 1], 'hi')); rows.append(rowof('RDMEhi | winner(60d ret)', G[G.L_60 == 0], 'hi'))
for s in (35, 36, 73, 48, 37): rows.append(rowof(f'RDMEhi | loser & SIC{s}', G[(G.L_dd == 1) & (G.s2 == s)], 'hi'))
G['cell'] = np.where((G.hi == 1) & (G.L_dd == 1) & (G.SM == 1), 1, np.where(G.hi.notna() & G.L_dd.notna() & G.SM.notna(), 0, np.nan)); rows.append(rowof('TRIPLE hi&loser&small vs rest', G, 'cell'))
print(pd.DataFrame(rows).to_string(index=False))
# D flip tag
F = pd.read_pickle(X + 'flips517.pkl'); H = F[F.sector == 'TECH'].copy(); H['d'] = pd.to_datetime(H.d).astype('datetime64[ns]')
H = pd.merge(H, DDp.drop_duplicates(['tk', 'd']), on=['tk', 'd'], how='left'); H = build(H, 'd', 'px'); H['yr'] = H.d.dt.year; trf = H.yr.between(2019, 2021)
qh = H.loc[trf, 'RDME'].quantile(.67); mdd = H.loc[trf, 'DD252'].median(); H['TAG'] = ((H.RDME > qh) & (H.DD252 < mdd)).astype(int); te = H.yr >= 2022
print(f'\nTECH flips {len(H)} RDME cov {H.RDME.notna().mean():.2f} DD252 cov {H.DD252.notna().mean():.2f} | cut RDME>{qh:.3f} DD252<{mdd:.2f} | TAG n {H.TAG.sum()}')
rng = np.random.default_rng(7); hit = H[te].hit.astype(float).values
for lab, m in (('ALL TECH', H.TAG >= 0), ('TAG=1', H.TAG == 1), ('TAG=0', H.TAG == 0)):
    s = H[m]; o = f'{lab:9} n={len(s):3d} reach {s.hit.mean()*100:3.0f}% mean {s.r.mean()*100:5.1f} med {s.r.median()*100:5.1f}'
    for nm, mk in (('train', trf), ('2022', H.yr == 2022), ('2023-26', H.yr >= 2023)): t = s[mk.loc[s.index]]; o += f' | {nm} n{len(t)} reach {t.hit.mean()*100 if len(t) else float("nan"):.0f}% mean {t.r.mean()*100 if len(t) else float("nan"):.1f}'
    print(o)
t = H[(H.TAG == 1) & te]
if len(t) >= 5: d = rng.choice(len(hit), (5000, len(t))); p = (hit[d].mean(1) >= t.hit.mean()).mean(); print(f'TAG test n{len(t)} reach {t.hit.mean()*100:.0f}% vs base {hit.mean()*100:.0f}% perm p {p:.4f} x4 {min(1,p*4):.3f}; mean {t.r.mean()*100:.1f} med {t.r.median()*100:.1f}')
