"""Nov 15 2026 SHORT ONLY WHEN THINGS ALIGN (Rey: short if insider, volume, or pivot-ratchet/EMA align). Declared before results. Same short book (daily wide universe bear flips, next-open entry, cover open after bull flip, 30bp + 15% annualized borrow). Flags at the bear-flip close: VOL = flip-bar volume >= 1.5x prior-20d mean; EMA = vector bear AND 21-EMA < 55-EMA AND close < 55-EMA (ratchet and EMA stack agree); INS = open-market insider SALES (Form 4 code S, disposition D) by officers/directors totaling >= $50k filed in the prior 30 days (insider-sale data exists ONLY 2025-01..2026-03, so INS is one-regime and cannot be split by era). Test: each flag vs not, then confluence counts (VOL+EMA; VOL+EMA+INS where INS exists). PASS = flagged shorts have mean net > 0 in BOTH eras with pooled month-clustered t >= 2 vs unflagged. INS reported for 2025-26 only as a lead. Survivorship biases shorts down; borrow unmodeled."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
from vector_core import ema, FAST_LEN, SLOW_LEN
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl')
I = pd.read_pickle(X + 'insider.pkl'); I = I[(I.TRANS_CODE == 'S') & (I.TRANS_ACQUIRED_DISP_CD == 'D')].copy(); I['fd'] = pd.to_datetime(I.FILING_DATE, errors='coerce'); I = I[I.RPTOWNER_RELATIONSHIP.astype(str).str.contains('Officer|Director', na=False)]
I['val'] = pd.to_numeric(I.TRANS_SHARES, errors='coerce') * pd.to_numeric(I.TRANS_PRICEPERSHARE, errors='coerce'); ins = {k: z.sort_values('fd')[['fd', 'val']] for k, z in I.dropna(subset=['fd', 'val']).groupby('ISSUERTRADINGSYMBOL')}
rows = []
for tk, B in D.items():
    B = B[~B.index.duplicated()]; B = B[B.index >= '2018-06-01']
    if len(B) < 300: continue
    o = B.open.values; c = B.close.values; v = B.volume.values.astype(float); idx = B.index; n = len(B); reg, _, _, _ = run_state_machine(B.assign(datetime=B.index)); ef = ema(c, FAST_LEN); es = ema(c, SLOW_LEN); dv = pd.Series(c * v).rolling(20).median().values; vm = pd.Series(v).rolling(20).mean().shift(1).values; i = 130
    while i < n - 2:
        if reg[i] == -1 and reg[i - 1] != -1:
            j = i + 1
            while j < n and reg[j] != 1: j += 1
            if j + 1 < n and '2019-01-01' <= str(idx[i])[:10] < '2026-03-01' and c[i] >= 1 and dv[i] >= 5e5:
                s = -(o[j + 1] / o[i + 1] - 1) - .003 - .15 * (j - i) / 252; insv = np.nan
                if tk in ins and idx[i] >= pd.Timestamp('2025-02-01'):
                    z = ins[tk]; insv = z[(z.fd <= idx[i]) & (z.fd > idx[i] - pd.Timedelta(days=30))].val.sum()
                elif idx[i] >= pd.Timestamp('2025-02-01'): insv = 0.0
                rows.append((tk, idx[i], s, v[i] / vm[i] >= 1.5, bool(ef[i] < es[i] and c[i] < es[i]), insv, c[i]))
            i = j
        i += 1
F = pd.DataFrame(rows, columns=['tk', 'd', 's', 'VOL', 'EMA', 'insv', 'px']); F['era'] = np.where(F.d < '2022-01-01', '2019-21', '2022-26Q1'); F['mo'] = F.d.dt.to_period('M'); F['INS'] = F.insv >= 5e4; print('bear flips', len(F), 'with insider data', int(F.insv.notna().sum()))
st = lambda x: f'n {len(x):5d} mean {x.mean()*100:+6.1f}% med {x.median()*100:+5.1f}% win {(x>0).mean()*100:3.0f}%'
def ct(z, m):
    a = z[m].groupby('mo').s.mean(); b = z[~m].groupby('mo').s.mean(); d = (a - b).dropna(); return d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 5 else np.nan
for lab, m in (('VOL (>=1.5x flip-bar volume)', F.VOL), ('EMA (ratchet + 21<55 + close<55)', F.EMA), ('VOL & EMA', F.VOL & F.EMA)):
    print(f'\n {lab}')
    for en in ('2019-21', '2022-26Q1', 'POOLED'):
        z = F if en == 'POOLED' else F[F.era == en]; mm = m.loc[z.index]; print(f'  {en:9s} FLAGGED {st(z[mm].s)} || NOT {st(z[~mm].s)} || diff {(z[mm].s.mean()-z[~mm].s.mean())*100:+.1f}pp t {ct(z, mm):+.1f}')
    print('  ->', 'PASS' if all(F[(F.era == e) & m].s.mean() > 0 for e in ('2019-21', '2022-26Q1')) else 'no pass (flagged mean not > 0 in both eras)')
z = F[F.insv.notna()]; print('\n INSIDER SALES (2025-02..2026-03 only, one regime): ')
for lab, m in (('INS', z.INS), ('INS & VOL', z.INS & z.VOL), ('INS & EMA', z.INS & z.EMA), ('INS & VOL & EMA', z.INS & z.VOL & z.EMA)): print(f'  {lab:16s} FLAGGED {st(z[m].s)} || NOT {st(z[~m].s)}')
F.to_pickle(X + 'short_confluence.pkl')
