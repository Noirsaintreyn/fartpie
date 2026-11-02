"""Nov 2 2026: (A) 'buy everything, cull what isn't a big move' with a FIXED budget: N slots at 1/N each (N = 6 .. 40),
exits = live or the cull rules in cull_gen.py. (B) random-combination search over the parts that ever showed something:
slots, exit per regime (stress / calm / other), universe width in stress, ranker, breaker. Discovery on 2019-21 only, then the
top picks are judged on 2022-26 (never seen). Matched drops: the same signals (by ticker + entry day) are dropped in every variant."""
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
import json, zlib, time
_W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); _W['d'] = _W.entry.dt.normalize()
RT = _W.groupby(['tk', 'd']).agg(ratio=('ratio', 'first'), sp=('sp', 'first')).reset_index().sort_values('d')
def prep2(fs):
    S_ = pd.concat(pd.read_parquet(f) for f in fs); S_['d'] = S_.entry.dt.normalize()
    S_ = S_[(S_.entry >= '2019-01-01') & ((S_.entry.dt.hour * 60 + S_.entry.dt.minute) < 840)].sort_values('d')
    S_ = pd.merge_asof(S_, RT, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); S_ = S_[(S_.sp == True) & (S_.ratio >= 1.0)]; return S_.reset_index(drop=True)
RX = pd.concat([prep2(glob.glob('/tmp/claude-0/ml/rx/*.parquet')), prep2(glob.glob('/tmp/claude-0/ml/cull/*.parquet')).query("var != 'live'"),
                prep2(glob.glob('/tmp/claude-0/ml/rx2/*.parquet')).query("var == 'eod'")], ignore_index=True)
d = RX.d.values
dn = mk.dn.reindex(d, method='ffill').fillna(False).astype(bool).values
Ast = st['A'].reindex(d, method='ffill').values; Cst = st['C'].reindex(d, method='ffill').values
RX['stress'] = dn | (Ast == 'Rebound') | (Cst == 'JM-0'); RX['calm'] = (Ast == 'Bull') & ~RX.stress
# ranker features from daily bars, known at the prior close
SEC = json.load(open('/home/claude/fartpie/sp500_sectors.json')); CLd, ONs = {}, {}
for tk in RX.tk.unique():
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0], columns=['datetime', 'open', 'close']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m_ = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m_ >= 570) & (m_ < 960)]; g = H.groupby(H.dt.dt.normalize()); D = pd.DataFrame({'o': g.open.first(), 'c': g.close.last()})
    on = np.log(D.o / D.c.shift(1)); idr = np.log(D.c / D.o); ONs[tk] = ((on ** 2).rolling(60).sum() / ((on ** 2) + (idr ** 2)).rolling(60).sum()).shift(1); CLd[tk] = D.c
CL = pd.DataFrame(CLd).sort_index(); R_ = CL.pct_change(); BIG = (R_ > 2 * R_.rolling(20).std().shift(1)).astype(float).fillna(0)
own = BIG.ewm(alpha=1 - np.exp(-1 / 10), adjust=False).mean(); sec = pd.Series({c: SEC.get(c, 'x') for c in BIG.columns}); peer = own * 0
for s_, cols in sec.groupby(sec).groups.items():
    tot = own[list(cols)].sum(axis=1)
    for c in cols: peer[c] = tot - own[c]
HK = (own + .3 * peer).shift(1); ON = pd.DataFrame(ONs)
def look(T, tk, d_): return [T[t].asof(x) if t in T else np.nan for t, x in zip(tk, d_)]
key = RX[['tk', 'd']].drop_duplicates(); key['hk'] = look(HK, key.tk, key.d); key['ons'] = look(ON, key.tk, key.d); RX = RX.merge(key, on=['tk', 'd'], how='left')
RX['yr'] = RX.d.dt.year; pr = lambda s: s.groupby(RX.yr).rank(pct=True).fillna(.5)
RANK = {'ratio': RX.ratio, 'ratio+hawkes': pr(RX.ratio) + pr(RX.hk), 'ratio+news share': pr(RX.ratio) + pr(RX.ons), 'stress first': pr(RX.ratio) + RX.stress * 10,
        '1h track first': pr(RX.ratio) + (RX.h1 == True) * 10}
for k_, v_ in RANK.items(): RX['k_' + k_] = v_.values
HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(RX.tk, RX.d.dt.strftime('%Y%m%d'))], dtype=np.uint64)
DROPS = [np.zeros(len(RX), bool)] + [((HSH * 2654435761 + i * 40503) % 1000) < 50 for i in range(1, 41)]
def build(cfg):
    parts = []
    for reg, m in (('stress', RX.stress), ('calm', RX.calm), ('other', ~RX.stress & ~RX.calm)):
        z = (RX['var'] == cfg[reg]) & m & ((RX.ratio >= 1.5) | ((RX.ratio >= cfg['wide']) & RX.stress)); parts.append(z.values)
    return np.logical_or.reduce(parts)
P1 = (np.datetime64('2019-01-01'), np.datetime64('2022-01-01')); P2_ = (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))
def one(eq, a, b):
    e = eq[(eq.index >= a) & (eq.index < b)]; e = np.exp(e - e.iloc[0]); yr = (e.index[-1] - e.index[0]).days / 365.25
    c = e.iloc[-1] ** (1 / yr) - 1; dd = (e / e.cummax() - 1).min(); return c, dd, c / -dd
def sim(cfg, dl, per=(P1, P2_)):
    mk_ = build(cfg) & ~dl; Z = RX[mk_][['entry', 'exit', 'ret', 'h1', 'tk']].copy(); Z['ratio'] = RX.loc[mk_, 'k_' + cfg['rank']].values
    if per == (P1,): Z = Z[Z.entry < '2022-01-01']
    if per == (P2_,): Z = Z[Z.entry >= '2022-01-01']
    eq = dret(simw(Z, n=cfg['n'], size=1 / cfg['n'], brk=cfg['brk'])).cumsum(); return [v for a, b in per for v in one(eq, a, b)]
LIVE = dict(n=6, stress='live', calm='live', other='live', wide=1.5, rank='ratio', brk=.08)
def ev(cfg, runs=40, per=(P1, P2_)): return np.array([sim(cfg, DROPS[i], per) for i in range(runs + 1)])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<44} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}  (return {(a[1:, 0] > b[1:, 0]).mean():.0%} / {(a[1:, 3] > b[1:, 3]).mean():.0%})'
    print(s, flush=True)
if os.getenv('PART', 'A') == 'A':
    t0 = time.time(); B0 = ev(LIVE); print(f'(A) buy-everything + cull, fixed budget, 40 matched runs, 2019-21 / 2022-26   [{time.time() - t0:.0f}s per variant]'); line('live (6 slots, live exits)', B0)
    for n in (6, 10, 15, 25, 40):
        for ex in ('live', 'c1_1', 'c1_2', 'c2_2', 'cbig1', 'cbig', 'eod'):
            if n == 6 and ex == 'live': continue
            c = dict(LIVE, n=n, stress=ex, calm=ex, other=ex); line(f'{n} slots @1/{n}, exits {ex}', ev(c), B0)
else:
    rng = np.random.default_rng(int(os.getenv('SEED', 7))); EX = ['live', 'loose', 'tight', 'leash', 'eod', 'c1_1', 'c1_2', 'c2_2', 'cbig1', 'cbig']
    SP = dict(n=[4, 6, 8, 10, 15, 25], stress=EX, calm=EX, other=EX, wide=[1.5, 1.2, 1.0], rank=list(RANK), brk=[.08, .12, 0])
    NC = int(os.getenv('NC', 200)); CF = []
    seen = set()
    while len(CF) < NC:
        c = {k: v[rng.integers(len(v))] for k, v in SP.items()}; s = json.dumps(c, sort_keys=True, default=float)
        if s not in seen: seen.add(s); CF.append(c)
    b1 = ev(LIVE, 10, (P1,)); b2 = ev(LIVE, 10, (P2_,)); res = []
    for i, c in enumerate(CF):
        a1 = ev(c, 10, (P1,)); a2 = ev(c, 10, (P2_,))
        res.append(dict(c, d_rdd=np.median(a1[1:, 2]), d_ret=np.median(a1[1:, 0]), d_win=(a1[1:, 2] > b1[1:, 2]).mean(),
                        h_rdd=np.median(a2[1:, 2]), h_ret=np.median(a2[1:, 0]), h_win=(a2[1:, 2] > b2[1:, 2]).mean()))
        if i % 20 == 19: print(f'  {i + 1} combos done', flush=True)
    R = pd.DataFrame(res); R.to_parquet(f'/tmp/claude-0/ml/combo_{os.getenv("SEED", 7)}.parquet')
    print(f'live: 2019-21 ret/DD {np.median(b1[1:, 2]):.2f} ret {np.median(b1[1:, 0]):+.1%}   2022-26 ret/DD {np.median(b2[1:, 2]):.2f} ret {np.median(b2[1:, 0]):+.1%}')
    print(f'rank correlation, discovery (2019-21) score vs holdout (2022-26) score across {len(R)} combos: {R.d_rdd.corr(R.h_rdd, method="spearman"):+.2f}')
    print(f'combos beating live in discovery (>=70% of runs): {(R.d_win >= .7).sum()};  of those, also beating live in holdout (>=70%): {((R.d_win >= .7) & (R.h_win >= .7)).sum()}')
    print(f'combos beating live in holdout regardless: {(R.h_win >= .7).sum()}')
    pd.set_option('display.width', 250); print('\nTop 10 by discovery score, with their unseen 2022-26 result:')
    print(R.sort_values('d_rdd', ascending=False).head(10).round(3).to_string(index=False))
