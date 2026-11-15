"""Fetch 1H (730d) and 15m (60d) RTH bars from Yahoo for the top-150 dollar-volume large caps (px>=20), cache to scratchpad/ext/largecap_{1h,15m}.pkl. Generic client UA, no credentials."""
import pandas as pd, numpy as np, requests, time, pickle
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl')
dvs = {}
for tk, q in D.items():
    q = q.tail(30)
    if len(q) < 25 or q.close.iloc[-1] < 20: continue
    dvs[tk] = (q.close * q.volume).median()
U = [t for t, _ in sorted(dvs.items(), key=lambda kv: -kv[1])[:150]]; print('universe', len(U), U[:15])
S = requests.Session(); S.headers['User-Agent'] = 'Mozilla/5.0 quant-research-script'
def get(tk, rng, iv):
    for _ in range(3):
        r = S.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{tk}', params=dict(range=rng, interval=iv, includePrePost='false'), timeout=30)
        if r.status_code == 200:
            j = r.json()['chart']['result']
            if not j or 'timestamp' not in j[0]: return None
            j = j[0]; q = j['indicators']['quote'][0]; ix = pd.to_datetime(j['timestamp'], unit='s', utc=True).tz_convert('America/New_York')
            return pd.DataFrame(dict(open=q['open'], high=q['high'], low=q['low'], close=q['close'], volume=q['volume']), index=ix).dropna()
        time.sleep(2)
    return None
for rng, iv, nm in (('730d', '60m', '1h'), ('60d', '15m', '15m')):
    out = {}
    for tk in U:
        z = get(tk.replace('.', '-'), rng, iv)
        if z is not None and len(z) > 200: out[tk] = z
        time.sleep(.15)
    pickle.dump(out, open(X + f'largecap_{nm}.pkl', 'wb')); print(nm, len(out), 'names', min(len(v) for v in out.values()), 'min bars', next(iter(out.values())).index[[0, -1]].tolist())
