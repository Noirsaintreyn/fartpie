"""
WEBULL EXECUTOR for the live vector screener (Nov 9 2026).

Mirrors the screener's decisions into a Webull account WITHOUT changing the screener:
  - reads live/signal_log.csv  (one row per BUY / SELL the screener posts to Discord, written at the 15m close)
  - reads live/ledger.json     (the screener's own book: every slot open / closed) to catch exits that never get a
                                signal_log row (late SELLs, earnings-guard sells at the close, profit lock)
The screener keeps doing everything it does now (signals, 6 slots, highest-ratio first, 8% breaker, no buys after
14:00, lock, earnings guard). This file only turns its BUY / SELL into Webull orders.

Sizing: each slot = account equity / SLOTS, fractional shares (Webull OpenAPI takes decimal quantities), market order,
regular session, DAY. A BUY older than BUY_STALE_MIN minutes is NOT chased (alert only). SELLs always go out.

SAFETY
  LIVE_ORDERS=0 (default) -> dry run: logs and Discord-posts the orders it WOULD send, sends nothing.
  touch live/webull_kill  -> stops all new orders until the file is removed.
  MAX_ORDER_USD caps any single buy; SLOTS caps open slots; only Mon-Fri 09:30-16:00 ET.
  client_order_id is a hash of the slot + signal bar, so a restart cannot send the same order twice.

SETUP (once)
  pip install webull-openapi-python-sdk python-dotenv pandas requests
  Webull website -> avatar -> Developer Console -> API Keys -> Apply for API Service (1-2 business days; the
  PaperTrade sandbox is approved in minutes) -> Register an API Application (Trading permission) -> Generate Key.
  ~/fartpie/.env:
      WEBULL_APP_KEY=...            WEBULL_APP_SECRET=...        WEBULL_ACCOUNT_ID=...  (python3 webull_exec.py --check prints it)
      WEBULL_API_HOST=api.webull.com   (production; use the sandbox host from the Developer Console for PaperTrade)
      LIVE_ORDERS=0                 SLOTS=6        MAX_ORDER_USD=500       WEBULL_START=2026-11-16 09:30
  The first authenticated call asks you to approve a 2FA token in the Webull app; it is saved in live/webull_token/.

RUN (from ~/fartpie, next to the screener's --loop)
  python3 webull_exec.py --check      auth, accounts, raw balance and positions (check the equity / buying-power fields)
  python3 webull_exec.py --status     executor slots vs Webull positions
  python3 webull_exec.py --loop       mirror the screener until 16:10 ET, then exit (dry run unless LIVE_ORDERS=1)
  python3 webull_exec.py --selftest   offline test with a fake broker and a fake signal log
"""
import argparse, hashlib, json, math, os, sys, time
import pandas as pd, requests
from dotenv import load_dotenv
load_dotenv('.env'); load_dotenv('discord_bot/.env')
ET = 'America/New_York'; LIVE = os.getenv('LIVE_DIR', 'live')
SIG, LEDGER, STATE, KILL = f'{LIVE}/signal_log.csv', os.getenv('LEDGER', f'{LIVE}/ledger.json'), f'{LIVE}/webull_state.json', f'{LIVE}/webull_kill'
SLOTS = int(os.getenv('SLOTS', '6')); MAX_USD = float(os.getenv('MAX_ORDER_USD', '500')); MIN_USD = float(os.getenv('MIN_ORDER_USD', '5'))
LIVE_ORDERS = os.getenv('LIVE_ORDERS', '0') == '1'; STALE = int(os.getenv('BUY_STALE_MIN', '10'))
START = pd.Timestamp(os.getenv('WEBULL_START', '2026-11-16 09:30')); CAPITAL = float(os.getenv('CAPITAL', '0') or 0)
SYMBOL = {'BRKB': 'BRK.B', 'BFB': 'BF.B'}                     # screener strips the dot; Webull wants it
def now_et(): return pd.Timestamp.now(tz=ET).tz_localize(None)
def log(msg):
    line = f'{now_et():%Y-%m-%d %H:%M:%S}  {msg}'; print(line, flush=True)
    try:
        with open(f'{LIVE}/webull_exec.log', 'a') as f: f.write(line + '\n')
    except OSError: pass
def notify(msg):
    url = os.getenv('WEBULL_DISCORD_WEBHOOK') or os.getenv('DISCORD_WEBHOOK_URL')
    if not url: return
    try: requests.post(url, json={'content': msg[:1900]}, timeout=10)
    except Exception as e: log(f'! discord: {str(e)[:80]}')
def find_num(js, keys):
    """first numeric value under any of `keys` (priority order) anywhere in a nested Webull response"""
    hits = {}
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in keys and k not in hits:
                    try: hits[k] = float(v)
                    except (TypeError, ValueError): pass
                walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(js)
    for k in keys:
        if k in hits: return hits[k]
    return None
EQUITY_KEYS = ['net_liquidation_value', 'net_liquidation', 'total_net_liquidation_value', 'total_asset', 'total_assets', 'account_value']
BP_KEYS = ['cash_buying_power', 'settled_cash', 'buying_power', 'day_buying_power', 'cash_balance', 'total_cash']

class Webull:
    def __init__(self):
        from webull.core.client import ApiClient
        from webull.trade.trade_client import TradeClient
        os.environ.setdefault('WEBULL_OPENAPI_TOKEN_DIR', f'{LIVE}/webull_token')
        api = ApiClient(os.environ['WEBULL_APP_KEY'], os.environ['WEBULL_APP_SECRET'], 'us')
        api.add_endpoint('us', os.getenv('WEBULL_API_HOST', 'api.webull.com'))
        self.tc = TradeClient(api); self.acct = os.getenv('WEBULL_ACCOUNT_ID')
    def _ok(self, res, what):
        if res.status_code != 200: raise RuntimeError(f'{what}: HTTP {res.status_code} {res.text[:200]}')
        return res.json()
    def accounts(self): return self._ok(self.tc.account_v2.get_account_list(), 'accounts')
    def balance(self): return self._ok(self.tc.account_v2.get_account_balance(self.acct), 'balance')
    def positions(self): return self._ok(self.tc.account_v2.get_account_position(self.acct), 'positions')
    def place(self, symbol, side, qty, coid):
        order = dict(combo_type='NORMAL', client_order_id=coid, symbol=symbol, instrument_type='EQUITY', market='US', order_type='MARKET',
                     quantity=f'{qty:.4f}'.rstrip('0').rstrip('.'), support_trading_session='CORE', side=side, time_in_force='DAY', entrust_type='QTY')
        return self._ok(self.tc.order_v3.place_order(self.acct, [order]), f'place {side} {symbol}')
    def order(self, coid): return self._ok(self.tc.order_v3.get_order_detail(self.acct, coid), 'order detail')

def st_load():
    try: return json.load(open(STATE))
    except Exception: return dict(rows=0, slots={}, done=[])
def st_save(S):
    tmp = STATE + '.tmp'; json.dump(S, open(tmp, 'w'), indent=1, default=str); os.replace(tmp, STATE)
def coid(*parts): return hashlib.md5('|'.join(map(str, parts)).encode()).hexdigest()
def rth(t): return t.weekday() < 5 and 570 <= t.hour * 60 + t.minute < 960
def ledger_open():
    """(ticker, track) pairs the screener's book holds open right now"""
    try: L = json.load(open(LEDGER))
    except Exception: return None
    return {(v['tk'], v['track']) for k, v in L.items() if not k.startswith('__') and isinstance(v, dict) and v.get('status') == 'open'}
def slot_of(ticker):                                          # signal_log ticker 'TSLA' / 'TSLA 1h' -> (base, track)
    p = str(ticker).split(); return p[0], (p[1] if len(p) > 1 else '15m')

class Executor:
    def __init__(self, broker): self.b = broker; self.S = st_load()
    def equity(self):
        if self.b is None: return CAPITAL, CAPITAL
        js = self.b.balance(); eq = find_num(js, EQUITY_KEYS); bp = find_num(js, BP_KEYS)
        return (eq if eq else CAPITAL), (bp if bp is not None else eq)
    def send(self, symbol, side, qty, key, why):
        msg = f"{'' if LIVE_ORDERS else '[DRY RUN] '}{side} {qty:.4f} {symbol} ({why})"
        if LIVE_ORDERS and self.b is not None and not os.path.exists(KILL):
            try: res = self.b.place(symbol, side, qty, key); log(f'{msg} -> sent {json.dumps(res)[:200]}')
            except Exception as e: log(f'!! {msg} FAILED: {e}'); notify(f'⚠️ Webull order FAILED: {msg}\n{str(e)[:300]}'); return False
        elif os.path.exists(KILL): log(f'KILL file present - not sending: {msg}'); return False
        else: log(msg)
        notify(f'\U0001f916 Webull: {msg}'); return True
    def buy(self, ticker, price, bar, run_at):
        if ticker in self.S['slots']: return
        if (now_et() - run_at) > pd.Timedelta(minutes=STALE):
            log(f'stale BUY {ticker} from {run_at:%H:%M} - not chasing'); notify(f'⚠️ Webull: BUY {ticker} signal is {STALE}+ min old - not chasing, check by hand'); return
        if len(self.S['slots']) >= SLOTS: log(f'BUY {ticker}: all {SLOTS} executor slots full - skipped'); return
        eq, bp = self.equity(); usd = min(eq / SLOTS, MAX_USD, bp if bp is not None else eq)
        if not price or price <= 0 or usd < MIN_USD: log(f'BUY {ticker}: ${usd:.2f} / price {price} - too small, skipped'); return
        base, track = slot_of(ticker); qty = math.floor(usd / price * 1e4) / 1e4; key = coid(ticker, bar, 'BUY')
        if self.send(SYMBOL.get(base, base), 'BUY', qty, key, f'slot {ticker}, ${usd:,.2f} of ${eq:,.2f} equity, signal bar {bar}'):
            self.S['slots'][ticker] = dict(symbol=SYMBOL.get(base, base), qty=qty, coid=key, t=str(now_et()), px=price, bar=str(bar))
    def sell(self, ticker, why):
        s = self.S['slots'].get(ticker)
        if not s: return
        if self.send(s['symbol'], 'SELL', s['qty'], coid(ticker, s['bar'], 'SELL'), why): self.S['slots'].pop(ticker); self.S['done'].append(dict(s, ticker=ticker, out=str(now_et()), why=why))
    def cycle(self):
        if not os.path.exists(SIG): return
        df = pd.read_csv(SIG, parse_dates=['run_at', 'bar']); new = df.iloc[self.S['rows']:]
        for r in new.itertuples():
            if r.run_at < START: continue
            if r.action == 'BUY': self.buy(r.ticker, float(r.price), r.bar, r.run_at)
            elif r.action == 'SELL': self.sell(r.ticker, f'screener SELL, bar {r.bar}')
        self.S['rows'] = len(df)
        lo = ledger_open()                                       # exits with no signal_log row (late SELL, earnings guard, lock)
        if lo is not None:
            for ticker, s in list(self.S['slots'].items()):
                if slot_of(ticker) not in lo and now_et() - pd.Timestamp(s['t']) > pd.Timedelta(minutes=20): self.sell(ticker, 'screener book closed this slot')
        st_save(self.S)

def selftest():
    import tempfile, shutil
    global SIG, LEDGER, STATE, KILL, START, CAPITAL, LIVE_ORDERS
    d = tempfile.mkdtemp(); SIG, LEDGER, STATE, KILL = f'{d}/s.csv', f'{d}/l.json', f'{d}/w.json', f'{d}/kill'; START = pd.Timestamp('2000-01-01'); CAPITAL = 700.0
    sent = []
    class Fake:
        def balance(self): return {'data': {'net_liquidation_value': '700.00', 'cash_buying_power': '650'}}
        def place(self, sym, side, qty, key): sent.append((sym, side, qty, key)); return {'ok': 1}
    LIVE_ORDERS = True; t = now_et()
    pd.DataFrame([dict(run_at=t, ticker='NVDA', action='BUY', bar=t, price=200.0, stop=190, entry=None, pnl=None),
                  dict(run_at=t, ticker='BRKB 1h', action='BUY', bar=t, price=480.0, stop=470, entry=None, pnl=None)]).to_csv(SIG, index=False)
    json.dump({'NVDA|15m|x': dict(tk='NVDA', track='15m', status='open'), 'BRKB|1h|x': dict(tk='BRKB', track='1h', status='open')}, open(LEDGER, 'w'))
    E = Executor(Fake()); E.cycle(); E.cycle()
    assert [s[:2] for s in sent] == [('NVDA', 'BUY'), ('BRK.B', 'BUY')], sent
    assert abs(sent[0][2] - math.floor(700 / 6 / 200 * 1e4) / 1e4) < 1e-9, sent[0]
    pd.DataFrame([dict(run_at=t, ticker='NVDA', action='SELL', bar=t, price=210.0, stop=0, entry=200, pnl=.05)]).to_csv(SIG, mode='a', header=False, index=False); E.cycle()
    assert sent[-1][:2] == ('NVDA', 'SELL') and sent[-1][2] == sent[0][2], sent
    E.S['slots']['BRKB 1h']['t'] = str(now_et() - pd.Timedelta(hours=1)); json.dump({'BRKB|1h|x': dict(tk='BRKB', track='1h', status='closed')}, open(LEDGER, 'w')); E.cycle()
    assert sent[-1][:2] == ('BRK.B', 'SELL'), sent
    stale = now_et() - pd.Timedelta(minutes=30); n = len(sent)
    pd.DataFrame([dict(run_at=stale, ticker='AMD', action='BUY', bar=stale, price=150.0, stop=0, entry=None, pnl=None)]).to_csv(SIG, mode='a', header=False, index=False); E.cycle()
    assert len(sent) == n, 'stale BUY must not be sent'
    open(KILL, 'w').close(); t2 = now_et()
    pd.DataFrame([dict(run_at=t2, ticker='META', action='BUY', bar=t2, price=600.0, stop=0, entry=None, pnl=None)]).to_csv(SIG, mode='a', header=False, index=False); E.cycle()
    assert len(sent) == n, 'kill file must block orders'
    shutil.rmtree(d); print('selftest OK:', [(s[0], s[1], s[2]) for s in sent])

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); ap.add_argument('--status', action='store_true')
    ap.add_argument('--loop', action='store_true'); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    if a.selftest: selftest(); sys.exit()
    os.makedirs(LIVE, exist_ok=True)
    b = Webull() if os.getenv('WEBULL_APP_KEY') else None
    if b is None: log('no WEBULL_APP_KEY in .env - running without a broker (dry run, sizes from CAPITAL)')
    if a.check:
        print(json.dumps(b.accounts(), indent=1)[:3000]); print(json.dumps(b.balance(), indent=1)[:3000]); print(json.dumps(b.positions(), indent=1)[:3000])
        eq, bp = Executor(b).equity(); print(f'\nequity read as ${eq}, buying power ${bp} -> slot ${(eq or 0) / SLOTS:,.2f}'); sys.exit()
    if a.status:
        S = st_load(); print(json.dumps(S['slots'], indent=1))
        if b: print(json.dumps(b.positions(), indent=1)[:3000])
        sys.exit()
    if a.loop:
        log(f"executor up: {'LIVE ORDERS' if LIVE_ORDERS else 'DRY RUN'}, {SLOTS} slots, max ${MAX_USD:,.0f}/order, signals from {START}")
        E = Executor(b)
        while True:
            t = now_et()
            if t.hour * 60 + t.minute >= 970: log('16:10 ET - executor done for the day'); st_save(E.S); break
            if t.weekday() < 5 and 565 <= t.hour * 60 + t.minute < 965:
                try: E.cycle()
                except Exception as e: log(f'! cycle: {str(e)[:200]}'); notify(f'⚠️ Webull executor error: {str(e)[:300]}')
            time.sleep(20)
