"""Nov 1 2026: ORCL and CVNA's recent bull flips vs the typical flip: where do they rank on each feature (percentile among 2019-26 book flips)?"""
import sys, os, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie'); os.chdir('/home/claude/fartpie'); import live_screener as LS
src = open('/home/claude/fartpie/xs/flip_quality.py').read(); exec(src[src.index('def feats('):src.index('rows = []')])
REF = pd.read_parquet('/tmp/claude-0/ml/flipq.parquet')
def get(tk):
    d = yf.Ticker(tk).history(period='60d', interval='15m', prepost=True).reset_index()
    d['dt'] = pd.to_datetime(d.iloc[:, 0]).dt.tz_convert('America/New_York').dt.tz_localize(None); d = d.rename(columns=str.lower)
    m = d.dt.dt.hour * 60 + d.dt.dt.minute; return d[(m >= 240) & (m < 960) & (d.volume > 0)][['dt', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
for tk in ('ORCL', 'CVNA'):
    H = get(tk); F, V = feats(H, pd.Timedelta('15min')); reg = V.reg.values; c = H.close.values
    fl = [i for i in range(1, len(H)) if reg[i] == 1 and reg[i - 1] != 1 and H.dt[i] >= pd.Timestamp('2026-10-20')]
    for i in fl:
        e = i + 1 if i + 1 < len(H) else i; ent = H.open[e]; after = H.iloc[e:]; nxt_bear = next((j for j in range(e, len(H)) if reg[j] == -1), len(H) - 1)
        best = H.high[e:nxt_bear + 1].max() / ent - 1; endp = H.close[nxt_bear] / ent - 1
        r = F.iloc[i]
        pct = {k: (REF[k] < r[k]).mean() * 100 for k in ('clv', 'uwick', 'beyond', 'relvol', 'dur', 'width') if r[k] == r[k]}
        print(f"{tk} flip bar {H.dt[i]:%a %m-%d %H:%M}  entry {ent:.2f}  best {best:+.1%}  to bear flip {endp:+.1%}")
        print('   ' + '  '.join(f'{k} {r[k]:.2f} (p{pct[k]:.0f})' for k in pct))
print('\nreference medians:', REF[['clv', 'uwick', 'beyond', 'relvol', 'dur', 'width']].median().round(2).to_dict())
