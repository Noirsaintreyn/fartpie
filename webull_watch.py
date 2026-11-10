"""
WEBULL WATCHLISTS + PHONE PUSH for the live vector screener (Nov 9 2026). NEVER places an order.

Two Webull watchlists, newest on top:
  "Vector slots"     the book's 6 slots: every BUY the screener takes (live/signal_log.csv), off again when it SELLs
                     (once no track of that name is held)
  "Vector mentions"  honorable mentions: flipped bullish but all slots were full, so the book did not buy (live/posted.json,
                     keys like 'TSLA|HM|2026-11-09 10:30'); today's only - cleared every morning, and a mention that the book
                     later buys moves to "Vector slots"
Each new flip can also send a phone push through ntfy (free, no account: install the ntfy app, subscribe to your NTFY_TOPIC).
The screener's Discord posts keep going exactly as now; this file only reads its logs.

SETUP (once) - same API key as webull_exec.py
  .env:  WEBULL_APP_KEY=...  WEBULL_APP_SECRET=...  WEBULL_API_HOST=api.webull.com
         WATCHLIST_SLOTS=Vector slots   WATCHLIST_HM=Vector mentions      (optional names)
         NTFY_TOPIC=rey-vec-<random>    (optional; anyone who knows the topic name can read it, so make it long and random)
  When you register the API application in the Webull Developer Console, enable Market Data as well as Trading: the
  watchlist endpoints are market-data endpoints (POST /market-data/watchlists/instruments/add). Stocks only (no options),
  max 20 watchlists and 1,000 names in total.
RUN (from ~/fartpie, next to the screener's --loop)
  python3 webull_watch.py --check      list your Webull watchlists, create the two lists if missing, send a test push
  python3 webull_watch.py --loop       follow the screener until 16:10 ET (push-only if there is no WEBULL_APP_KEY yet)
  python3 webull_watch.py --selftest   offline test with a fake Webull and fake screener logs
"""
import argparse, json, os, re, sys, time
import pandas as pd, requests
from dotenv import load_dotenv
load_dotenv('.env'); load_dotenv('discord_bot/.env')
ET = 'America/New_York'; LIVE = os.getenv('LIVE_DIR', 'live')
SIG, POSTED, STATE = f'{LIVE}/signal_log.csv', f'{LIVE}/posted.json', f'{LIVE}/webull_watch.json'
SLOTS_L, HM_L = os.getenv('WATCHLIST_SLOTS', 'Vector slots'), os.getenv('WATCHLIST_HM', 'Vector mentions')
TOPIC = os.getenv('NTFY_TOPIC'); NTFY = os.getenv('NTFY_SERVER', 'https://ntfy.sh').rstrip('/')
SYMBOL = {'BRKB': 'BRK.B', 'BFB': 'BF.B'}                     # screener strips the dot; Webull wants it
def now_et(): return pd.Timestamp.now(tz=ET).tz_localize(None)
def log(msg):
    line = f'{now_et():%Y-%m-%d %H:%M:%S}  {msg}'; print(line, flush=True)
    try:
        with open(f'{LIVE}/webull_watch.log', 'a') as f: f.write(line + '\n')
    except OSError: pass
def push(title, msg, prio='high'):
    if not TOPIC: return
    try: requests.post(f'{NTFY}/{TOPIC}', data=msg.encode(), headers={'Title': title.encode('ascii', 'ignore').decode(), 'Priority': prio, 'Tags': 'chart_with_upwards_trend'}, timeout=10)
    except Exception as e: log(f'! push: {str(e)[:80]}')
def base_of(ticker): return str(ticker).split()[0]                 # 'TSLA 1h' -> 'TSLA'
def sym_of(base): return SYMBOL.get(base, base)

class WebullWatch:
    def __init__(self):
        from webull.core.client import ApiClient
        from webull.data.data_client import DataClient
        os.environ.setdefault('WEBULL_OPENAPI_TOKEN_DIR', f'{LIVE}/webull_token')
        api = ApiClient(os.environ['WEBULL_APP_KEY'], os.environ['WEBULL_APP_SECRET'], 'us')
        api.add_endpoint('us', os.getenv('WEBULL_API_HOST', 'api.webull.com'))
        self.w = DataClient(api).watchlist; self.ids = {}
    def _ok(self, res, what):
        if res.status_code != 200: raise RuntimeError(f'{what}: HTTP {res.status_code} {res.text[:200]}')
        try: return res.json()
        except ValueError: return {}
    def lists(self):
        out = []
        def walk(x):
            if isinstance(x, dict):
                if 'watchlist_id' in x and 'name' in x: out.append(x)
                for v in x.values(): walk(v)
            elif isinstance(x, list):
                for v in x: walk(v)
        walk(self._ok(self.w.get_watchlist(), 'list watchlists')); return out
    def ensure(self, name):
        if name in self.ids: return self.ids[name]
        hit = [x for x in self.lists() if x.get('name') == name]
        if hit: self.ids[name] = hit[0]['watchlist_id']; return self.ids[name]
        js = self._ok(self.w.create_watchlist(name), f'create watchlist {name}')
        m = re.search(r'"watchlist_id"\s*:\s*"?([^",}]+)', json.dumps(js))
        wid = m.group(1) if m else ([x for x in self.lists() if x.get('name') == name] or [{}])[0].get('watchlist_id')
        if not wid: raise RuntimeError(f'created {name} but could not read its id: {json.dumps(js)[:200]}')
        log(f'created watchlist "{name}" ({wid})'); self.ids[name] = wid; return wid
    def add(self, name, sym, sort): self._ok(self.w.add_instruments(self.ensure(name), [dict(symbol=sym, category='US_STOCK', sort=int(sort))]), f'add {sym} to {name}')
    def remove(self, name, sym): self._ok(self.w.remove_instruments(self.ensure(name), [dict(symbol=sym, category='US_STOCK')]), f'remove {sym} from {name}')

def st_load():
    try: return json.load(open(STATE))
    except Exception: return None
def st_save(S):
    tmp = STATE + '.tmp'; json.dump(S, open(tmp, 'w'), indent=1, default=str); os.replace(tmp, STATE)

class Watcher:
    def __init__(self, wl):
        self.w = wl; S = st_load()
        if S is None:                                              # first run: start from what is already logged, do not replay history
            S = dict(rows=0, hm_seen=[], on={SLOTS_L: {}, HM_L: {}}, held={}, day=None)
            try: S['rows'] = len(pd.read_csv(SIG))
            except Exception: pass
            S['hm_seen'] = [k for k in self.posted() if '|HM|' in k]
        self.S = S
    def posted(self):
        try: return json.load(open(POSTED)).get('signals', [])
        except Exception: return []
    def on(self, name, sym, why):
        if sym in self.S['on'][name]: return False
        self.S['on'][name][sym] = str(now_et())
        if self.w is not None:
            try: self.w.add(name, sym, int(time.time() // 60)); log(f'+ {sym} on "{name}" ({why})')
            except Exception as e: log(f'!! add {sym} to {name} failed: {e}'); push('Watchlist add failed', f'{sym} -> {name}: {str(e)[:200]}', 'default')
        return True
    def off(self, name, sym, why):
        if self.S['on'][name].pop(sym, None) is None: return
        if self.w is not None:
            try: self.w.remove(name, sym); log(f'- {sym} off "{name}" ({why})')
            except Exception as e: log(f'!! remove {sym} from {name} failed: {e}')
    def cycle(self):
        S = self.S; today = str(now_et().date())
        if S['day'] != today:                                      # new session: yesterday's mentions go; slots stay = what the book holds
            for sym in list(S['on'][HM_L]): self.off(HM_L, sym, 'new day')
            held = {sym_of(b) for b in S['held'].values()}
            for sym in list(S['on'][SLOTS_L]):
                if sym not in held: self.off(SLOTS_L, sym, 'not held')
            S['day'] = today
        if os.path.exists(SIG):
            df = pd.read_csv(SIG, parse_dates=['run_at', 'bar'])
            for r in df.iloc[S['rows']:].itertuples():
                base = base_of(r.ticker); sym = sym_of(base)
                if r.action == 'BUY':
                    S['held'][r.ticker] = base; self.off(HM_L, sym, 'the book bought it'); self.on(SLOTS_L, sym, f'BUY {r.ticker}')
                    push(f'BUY {r.ticker} (slot)', f"{r.ticker} BUY - {pd.Timestamp(r.bar):%H:%M} bar, last {float(r.price):.2f}, stop {float(r.stop):.2f}. On {SLOTS_L}.")
                elif r.action == 'SELL':
                    S['held'].pop(r.ticker, None)
                    if base not in S['held'].values(): self.off(SLOTS_L, sym, f'screener SELL {r.ticker}')
            S['rows'] = len(df)
        for k in self.posted():
            if '|HM|' not in k or k in S['hm_seen']: continue
            S['hm_seen'].append(k); tk, _, when = k.split('|', 2); when = when.replace('pre|', 'premarket ')
            if today not in when: continue
            sym = sym_of(base_of(tk))
            if sym not in S['on'][SLOTS_L]: self.on(HM_L, sym, f'mention {tk}')
            push(f'Mention {tk} (slots full)', f"{tk} flipped bullish ({when}) but all slots are full - the book is NOT buying it. On {HM_L}.")
        S['hm_seen'] = S['hm_seen'][-1000:]; st_save(S)

def selftest():
    import tempfile, shutil
    global SIG, POSTED, STATE, TOPIC
    d = tempfile.mkdtemp(); SIG, POSTED, STATE = f'{d}/s.csv', f'{d}/p.json', f'{d}/w.json'; TOPIC = None
    class Fake:
        def __init__(self): self.L = {SLOTS_L: set(), HM_L: set()}
        def add(self, name, sym, sort): self.L[name].add(sym)
        def remove(self, name, sym): self.L[name].discard(sym)
    F = Fake(); t = now_et(); day = f'{t:%Y-%m-%d}'
    pd.DataFrame([dict(run_at=t - pd.Timedelta(days=3), ticker='AMD', action='BUY', bar=t, price=150.0, stop=145, entry=None, pnl=None)]).to_csv(SIG, index=False)
    json.dump({'signals': ['META|HM|2026-01-02 10:30']}, open(POSTED, 'w'))
    W = Watcher(F); W.cycle(); assert F.L == {SLOTS_L: set(), HM_L: set()}, 'first run must not replay history'
    pd.DataFrame([dict(run_at=t, ticker='NVDA', action='BUY', bar=t, price=200.0, stop=190, entry=None, pnl=None),
                  dict(run_at=t, ticker='BRKB 1h', action='BUY', bar=t, price=480.0, stop=470, entry=None, pnl=None)]).to_csv(SIG, mode='a', header=False, index=False)
    json.dump({'signals': ['META|HM|2026-01-02 10:30', f'TSLA|HM|{day} 10:45', f'AMD|HM|{day} 11:00', f'NVDA 1h|HM|{day} 11:00']}, open(POSTED, 'w')); W.cycle(); W.cycle()
    assert F.L == {SLOTS_L: {'NVDA', 'BRK.B'}, HM_L: {'TSLA', 'AMD'}}, F.L
    pd.DataFrame([dict(run_at=t, ticker='AMD', action='BUY', bar=t, price=151.0, stop=146, entry=None, pnl=None),
                  dict(run_at=t, ticker='NVDA', action='SELL', bar=t, price=210.0, stop=0, entry=200, pnl=.05)]).to_csv(SIG, mode='a', header=False, index=False); W.cycle()
    assert F.L == {SLOTS_L: {'BRK.B', 'AMD'}, HM_L: {'TSLA'}}, f'a mention the book buys moves to slots; a SELL leaves slots: {F.L}'
    W.S['day'] = '2000-01-01'; W.cycle(); assert F.L == {SLOTS_L: {'BRK.B', 'AMD'}, HM_L: set()}, f'new day clears mentions, keeps slots: {F.L}'
    W2 = Watcher(F); assert W2.S['held'] == {'BRKB 1h': 'BRKB', 'AMD': 'AMD'}, 'state survives a restart'
    shutil.rmtree(d); print(f'selftest OK: {SLOTS_L} = {sorted(F.L[SLOTS_L])}, {HM_L} = {sorted(F.L[HM_L])}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); ap.add_argument('--loop', action='store_true'); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest: selftest(); sys.exit()
    os.makedirs(LIVE, exist_ok=True)
    wl = None
    if os.getenv('WEBULL_APP_KEY'):
        try: wl = WebullWatch()                                    # connects on construction (401 here = wrong key / host)
        except Exception as e:
            if a.check: raise
            log(f'!! Webull login failed ({str(e)[:150]}) - push alerts only today'); push('Webull watchlist down', f'Login failed: {str(e)[:200]}', 'default')
    else: log('no WEBULL_APP_KEY in .env - push alerts only, no watchlists')
    if a.check:
        if wl: print(json.dumps(wl.lists(), indent=1)[:3000]); print({n: wl.ensure(n) for n in (SLOTS_L, HM_L)})
        push('Vector flips test', 'Test push from webull_watch.py - alerts for new flips will look like this.'); print('test push sent' if TOPIC else 'no NTFY_TOPIC set - no push'); sys.exit()
    if a.loop:
        log(f'watch up: "{SLOTS_L}" + "{HM_L}"' + (' (Webull)' if wl else ' (no Webull key)') + (f', push to {NTFY}/<topic>' if TOPIC else ', no push'))
        W = Watcher(wl)
        while True:
            t = now_et()
            if t.hour * 60 + t.minute >= 970: log('16:10 ET - watch done for the day'); st_save(W.S); break
            try: W.cycle()
            except Exception as e: log(f'! cycle: {str(e)[:200]}')
            time.sleep(20)
