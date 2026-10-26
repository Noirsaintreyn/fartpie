"""
DegenCap level bot (Discord) - talks to the fartpie backend.

  /levels ticker timeframe view   view = Levels (support/resistance, high to low) or Theoretical HOD/LOD
  /chart  ticker timeframe model_timeframe view   candles colored by the vector regime; view = THOD/TLOD or Levels, built from model_timeframe
  /check  [ticker]   the screener's open positions with their charts (all, or one ticker)
  /history           every screener trade (on-time fills), book P&L, book vs SPY   (odds / touch-prob views removed Oct 18 2026)

Env: DISCORD_TOKEN, BACKEND_URL, BACKEND_USER, BACKEND_PASSWORD  (see .env.example)
"""
import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import functools
import io
import os

import discord
from discord import app_commands

import analysis as A
import daily_after as DA
import vchart as VC
try:
    import gexchart as GX                     # GEX-by-strike charts (needs scipy + matplotlib)
except Exception as _e:
    GX = None; print(f'GEX charts disabled: {_e}')
try:
    import qr_zones as QZ                     # 1h / 4h quantile-regression THOD/TLOD -> zones with your lines (needs statsmodels)
except Exception as _e:
    QZ = None; print(f'QR zones disabled: {_e}')
from backend_client import Backend, BackendDown, TIMEFRAMES, norm_tf, norm_ticker

try:                                                    # local runs: read tokens from discord_bot/.env and ./.env (Render sets real env vars)
    from dotenv import load_dotenv
    _here = os.path.dirname(os.path.abspath(__file__)); load_dotenv(os.path.join(_here, '.env')); load_dotenv(os.path.join(_here, '..', '.env'))
except ImportError: pass


bk = Backend()
intents = discord.Intents.none()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
TF_CHOICES = [app_commands.Choice(name=t, value=t) for t in TIMEFRAMES]
COLOR = 0x7e57c2
FOOT = 'Levels are structural candidates (ML-filtered where the backend filter applies), not signals. Not financial advice.'


async def run(fn, *a):
    return await asyncio.get_running_loop().run_in_executor(None, fn, *a)


async def run_retry(fn, *a, tries=4, wait=25):
    """backend calls: if Render is cold-starting / restarting (502), wait and retry before giving up (the interaction is already deferred, so we have minutes)"""
    for i in range(tries):
        try:
            return await run(fn, *a)
        except BackendDown:
            if i == tries - 1:
                raise
            await asyncio.sleep(wait)


async def keep_warm():
    """ping the backend every 10 minutes so the Render service does not spin down while the bot is running"""
    import requests
    from backend_client import BASE
    while True:
        try:
            await run(lambda: requests.get(f'{BASE}/api/health', timeout=60))
        except Exception:
            pass
        await asyncio.sleep(600)


def er_line(er):
    if not er:
        return None
    return (f"`{er['expected_low']:,.2f}` – `{er['expected_high']:,.2f}` ({er['expected_range_pct']:.2f}% wide, "
            f"{er.get('position_description', '')})")


async def load(ticker, tf):
    t, f = norm_ticker(ticker), norm_tf(tf)
    resp = await run_retry(bk.data, t, f)
    er = await run(bk.expected_range, t)
    levels = A.flatten_levels(resp)
    atr = A.atr14(resp['priceData'])
    return t, f, resp, er, levels, atr


VIEW_CHOICES = [app_commands.Choice(name='Levels (support / resistance)', value='levels'),
                app_commands.Choice(name='Theoretical HOD / LOD', value='hodlod')]


@tree.command(name='levels', description='Levels or theoretical HOD/LOD for a ticker + timeframe')
@app_commands.describe(ticker='e.g. NQ, ES, AAPL', timeframe='bar size', view='what to show')
@app_commands.choices(timeframe=TF_CHOICES, view=VIEW_CHOICES)
async def levels_cmd(inter: discord.Interaction, ticker: str, timeframe: app_commands.Choice[str], view: app_commands.Choice[str]):
    await inter.response.defer(thinking=True)
    t, f = norm_ticker(ticker), timeframe.value
    try:
        if view.value == 'hodlod':
            hl = await run_retry(bk.hod_lod, t, f)
            price = hl['currentPrice']
            p_, hod, lod = hl['predicted'], hl['hod'], hl['lod']
            rows = [f"{'':<8}{'HOD':>12}{'LOD':>12}",
                    f"{'1σ':<8}{hod['1std']:>12,.2f}{lod['1std']:>12,.2f}",
                    f"{'2σ':<8}{hod['2std']:>12,.2f}{lod['2std']:>12,.2f}",
                    f"{'3σ':<8}{hod['3std']:>12,.2f}{lod['3std']:>12,.2f}",
                    f"{'Predicted':<8}{p_['hod']:>12,.2f}{p_['lod']:>12,.2f}",
                    f"{'Conf':<8}{p_['hod_confidence']:>12.0%}{p_['lod_confidence']:>12.0%}"]
            emb = discord.Embed(title=f'{t} · {f} theoretical HOD / LOD', color=COLOR,
                                description=f'Price `{price:,.2f}`\n```\n' + '\n'.join(rows) + '\n```')
            er = await run(bk.expected_range, t)
            if er:
                emb.add_field(name='Expected range today', value=er_line(er), inline=False)
            emb.set_footer(text=FOOT)
            return await inter.followup.send(embed=emb)
        resp = await run_retry(bk.data, t, f)
    except Exception as e:
        return await inter.followup.send(f'Could not build {view.name.lower()}: {e}')
    price = resp['currentPrice']
    lv = A.all_levels(resp)
    res = [l for l in lv if l['price'] >= price]
    sup = [l for l in lv if l['price'] < price]
    embs = []
    for title, group in ((f'Resistance ({len(res)})', res), (f'Support ({len(sup)})', sup)):
        for i, d in enumerate(A.chunk_block('', A.level_rows(group, price)) if group else ['none']):
            embs.append(discord.Embed(title=title if i == 0 else None, description=d, color=0xff7043 if title.startswith('Res') else 0x66bb6a))
    embs[0].description = f'**{t} · {f}** · price `{price:,.2f}` · high → low\n' + embs[0].description
    embs[-1].set_footer(text=FOOT)
    await inter.followup.send(embeds=embs[:10])


CHART_VIEWS = [app_commands.Choice(name='THOD / TLOD', value='thod'), app_commands.Choice(name='Levels', value='levels')]


@tree.command(name='chart', description='Chart with vector-colored candles: pick the timeframe, the model timeframe, and what to show')
@app_commands.describe(ticker='e.g. NQ, ES, NVDA', timeframe='candle timeframe of the chart', model_timeframe='timeframe the levels / THOD-TLOD are built from',
                       view='THOD/TLOD or Levels')
@app_commands.choices(timeframe=TF_CHOICES, model_timeframe=TF_CHOICES, view=CHART_VIEWS)
async def chart_cmd(inter: discord.Interaction, ticker: str, timeframe: app_commands.Choice[str], model_timeframe: app_commands.Choice[str], view: app_commands.Choice[str]):
    await inter.response.defer(thinking=True)
    t, f, mtf = norm_ticker(ticker), timeframe.value, model_timeframe.value
    try:
        resp = await run_retry(bk.data, t, f)
        r = resp if mtf == f else await run_retry(bk.data, t, mtf)
        hl = await run_retry(bk.hod_lod, t, mtf) if view.value == 'thod' else None
        buf = await run(VC.render, t, f, view.value, resp, [dict(tf=mtf, resp=r, hl=hl, levels=A.all_levels(r))])
    except Exception as e:
        return await inter.followup.send(f'Could not build the chart: {e}')
    await inter.followup.send(file=discord.File(buf, filename=f'{t}_{f}_{view.value}.png'))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENER_PY = os.getenv('SCREENER_PY', '/usr/bin/python3')       # same interpreter the screener's launchd job uses


async def _screener_card(inter, extra, out):
    """run live_screener.py with the screener's interpreter, send the PNG it writes (live prices during the session)"""
    if ON_RENDER and os.environ.get('RUN_BOT') != '1':                # RUN_BOT=1: bot shares the screener's worker + disk, so it works on Render too
        return await inter.followup.send("This command needs the screener's data, which this bot instance can't see.")
    now = datetime.now(NYTZ); live = now.weekday() < 5 and 4 <= now.hour and (now.hour, now.minute) < (16, 15)
    cmd = [SCREENER_PY, 'live_screener.py'] + extra + ([] if live else ['--no-fetch'])
    try:
        p = await asyncio.wait_for(asyncio.create_subprocess_exec(*cmd, cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT), 10)
        so, _ = await asyncio.wait_for(p.communicate(), 300); so = so.decode(errors='ignore')
    except Exception as e:
        return await inter.followup.send(f'Could not run the screener: {e}')
    try:
        if not os.path.exists(out):
            msg = [l for l in so.splitlines() if l.startswith('FLAT:') or l.startswith('!')]
            return await inter.followup.send(msg[-1] if msg else f"Screener failed:\n```\n{so[-1500:]}\n```")
        await inter.followup.send(file=discord.File(out, filename=os.path.basename(out).split('_', 1)[0] + '.png'))
    finally:
        try: os.remove(out)
        except OSError: pass


@tree.command(name='check', description="Screener's open positions with their charts (live prices)")
@app_commands.describe(ticker='optional: one ticker, e.g. ADBE (shows its 15m and 1h positions)')
async def check_cmd(inter: discord.Interaction, ticker: str = None):
    await inter.response.defer(thinking=True)
    out = os.path.join(ROOT, 'live', f'positions_{inter.id}.png')
    await _screener_card(inter, ['--check-card', out] + (['--check-ticker', ticker.strip().upper()] if ticker else []), out)


@tree.command(name='history', description='Every screener trade since the start: returns, total book P&L, and the book vs SPY')
async def history_cmd(inter: discord.Interaction):
    await inter.response.defer(thinking=True)
    out = os.path.join(ROOT, 'live', f'history_{inter.id}.png')
    await _screener_card(inter, ['--history-card', out], out)


async def _screener_text(inter, extra):
    """run live_screener.py and send the text it prints between <<<OUT and OUT>>> (split to Discord's 2000-char limit)"""
    now = datetime.now(NYTZ); live = now.weekday() < 5 and 4 <= now.hour and (now.hour, now.minute) < (16, 15)
    cmd = [SCREENER_PY, 'live_screener.py'] + extra + ([] if live else ['--no-fetch'])
    try:
        p = await asyncio.wait_for(asyncio.create_subprocess_exec(*cmd, cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT), 10)
        so, _ = await asyncio.wait_for(p.communicate(), 300); so = so.decode(errors='ignore')
    except Exception as e:
        return await inter.followup.send(f'Could not run the screener: {e}')
    if '<<<OUT' not in so: return await inter.followup.send(f"Screener failed:\n```\n{so[-1500:]}\n```")
    txt = so.split('<<<OUT', 1)[1].split('OUT>>>', 1)[0].strip(); chunk = ''
    for line in txt.splitlines():
        if len(chunk) + len(line) > 1900: await inter.followup.send(chunk); chunk = ''
        chunk += line + '\n'
    if chunk: await inter.followup.send(chunk)


@tree.command(name='pro', description="One stock: vector state, volatility regime, and whether today's high or low is in (with the projected other side)")
@app_commands.describe(ticker='e.g. AAPL')
async def pro_cmd(inter: discord.Interaction, ticker: str):
    await inter.response.defer(thinking=True)
    await _screener_text(inter, ['--pro', ticker.strip().upper()])


@tree.command(name='brief', description='Morning report: the book, sector ETFs with their vectors, calendar, news, and a page per position')
async def brief_cmd(inter: discord.Interaction):
    await inter.response.defer(thinking=True)
    out = os.path.join(ROOT, 'live', f'report_{inter.id}')
    now = datetime.now(NYTZ); live = now.weekday() < 5 and 4 <= now.hour and (now.hour, now.minute) < (16, 15)
    cmd = [SCREENER_PY, 'live_screener.py', '--report-dir', out] + ([] if live else ['--no-fetch'])
    try:
        p = await asyncio.wait_for(asyncio.create_subprocess_exec(*cmd, cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT), 10)
        so, _ = await asyncio.wait_for(p.communicate(), 300); so = so.decode(errors='ignore')
    except Exception as e:
        return await inter.followup.send(f'Could not build the report: {e}')
    fs = [l.split('FILE:', 1)[1].strip() for l in so.splitlines() if l.startswith('FILE:')]
    if not fs: return await inter.followup.send(f"Report failed:\n```\n{so[-1500:]}\n```")
    try: await inter.followup.send(files=[discord.File(f, filename=os.path.basename(f)) for f in fs[:10]])
    finally:
        import shutil; shutil.rmtree(out, ignore_errors=True)


@tree.command(name='hm', description='Honorable mentions: flips the screener skipped because every slot was full (tracked, not traded)')
async def hm_cmd(inter: discord.Interaction):
    await inter.response.defer(thinking=True)
    out = os.path.join(ROOT, 'live', f'hm_{inter.id}.png')
    await _screener_card(inter, ['--hm-card', out], out)


@tree.command(name='hh', description='Honorable-mention history: every flip skipped for full slots since the start, as if traded, vs the real book')
async def hh_cmd(inter: discord.Interaction):
    await inter.response.defer(thinking=True)
    out = os.path.join(ROOT, 'live', f'hh_{inter.id}.png')
    await _screener_card(inter, ['--hh-card', out], out)


DATA_DIR = os.environ.get('DATA_DIR') or os.path.dirname(os.path.abspath(__file__))       # /data on Render (persistent disk), this folder locally
os.makedirs(DATA_DIR, exist_ok=True)
DAILY_FILE = os.path.join(DATA_DIR, 'daily_channels.json')
ON_RENDER = bool(os.environ.get('RENDER'))
NYTZ = ZoneInfo('America/New_York')


def load_daily():
    try:
        return json.load(open(DAILY_FILE))
    except Exception:
        return {'channels': {}, 'last': ''}


def save_daily(st):
    json.dump(st, open(DAILY_FILE, 'w'))


def vol_word(rv_bps):          # RV30 terciles from the last-month sample (2.2 / 3.4 / 5.4 bps per minute medians)
    return 'quiet' if rv_bps < 2.8 else ('normal' if rv_bps < 4.4 else 'fast')


ISSUED = {}                              # ticker -> what the 09:30 chart was issued with (saved to disk after the post, used by the 16:05 'after' post)


DAILY_STOCKS = ['V', 'AMD', 'MSFT', 'MCD', 'WMT']     # Oct 18 2026: names where fading THOD/TLOD beat random lines in both years (xs/thod_by_stock_app.py); was AAPL AMD MSFT NVDA


QR_ZONES = os.getenv('QR_ZONES', '0') == '1'             # Oct 18 2026: off - zones/QR lines added nothing over THOD lines (xs/qr_zone_app.py); QR_ZONES=1 to show
THOD_TFS = ('1h', '4h')                                   # Oct 14 2026: daily dropped; 1h + 4h THOD/TLOD + quantile-regression zones


def _gex_scaled(gx, price):
    """index futures use QQQ/SPY options: rescale strikes to the futures price"""
    if gx is None or abs(price / gx['spot'] - 1) < .2: return gx
    f = price / gx['spot']; g = dict(gx); G = gx['G'].copy(); G.index = G.index * f; g['G'] = G; g['spot'] = price
    for k in ('call_wall', 'put_wall', 'zero_gamma'): g[k] = gx[k] * f if gx.get(k) is not None else None
    return g


_QR = {}
async def qr_today():
    """one pooled QR fit per day for all 5 names (refreshes the 15m history first); ~5 s"""
    d = str(datetime.now(NYTZ).date())
    if QZ is None: return {}
    if d not in _QR:
        await run(QZ.update_bars); _QR.clear(); _QR[d] = await run(QZ.predict, d, True)
    return _QR[d]


async def build_thod_chart(t):
    """5m candles (regular), full-width THOD/TLOD 1-3 sigma lines from the daily and the 4h model with prices on the right, GEX zones"""
    r5 = await run_retry(bk.data, t, '5m')
    H = [await run_retry(bk.hod_lod, t, tf) for tf in THOD_TFS]
    ISSUED[t] = DA.issue_meta(r5, H[0], H[1], THOD_TFS, H[2] if len(H) > 2 else None)
    bands, xlines = [], []
    try:
        if not QR_ZONES: raise RuntimeError('off')
        qr = await qr_today(); px = float(r5['currentPrice'])
        for tf, h, col in zip(THOD_TFS, H, (VC.STYLE[0]['thod'], VC.STYLE[1]['thod'])):
            b_, x_ = QZ.zones((qr.get(t) or {}).get(tf), h, px, tf, col); bands += b_; xlines += x_
        ISSUED[t]['qr'] = {tf: (qr.get(t) or {}).get(tf) for tf in THOD_TFS}; ISSUED[t]['qr_px'] = px
    except Exception as ex:
        if QR_ZONES: print(f'{t} QR zones skipped: {ex}')
    gx = None
    if GX is not None:
        try: gx = _gex_scaled(await run(GX.profile, t), float(r5['currentPrice']))
        except Exception as ex: print(f'{t} GEX zones skipped: {ex}')
    return await run(VC.render, t, '5m', 'thod', r5, [dict(tf=tf, resp=r5, hl=h, levels=[]) for tf, h in zip(THOD_TFS, H)], 110, None, gx, bands, xlines)


async def post_daily(st):
    for cid, t in list(st['channels'].items()):
        try:
            ch = client.get_channel(int(cid)) or await client.fetch_channel(int(cid))
            files = [discord.File(await build_thod_chart(t), filename=f"{t.replace('=F', '')}_0930_thod.png")]     # index chart now the same style as the stocks
            for st_ in DAILY_STOCKS:
                if st_ == t: continue
                try: files.append(discord.File(await build_thod_chart(st_), filename=f'{st_}_0930_thod.png'))
                except Exception as ex: print(f'{st_} THOD chart failed: {ex}')
            await ch.send(files=files)
            try: await post_gex(ch, t)
            except Exception as ex: print(f'GEX post failed: {ex}')
        except Exception as e:
            print(f'daily post failed for {cid}: {e}')
    DA.save_issue(str(datetime.now(NYTZ).date()), ISSUED)


async def post_gex(ch, t):
    if GX is None: return
    """GEX-by-strike charts (yesterday's close + this morning's OI) with the same 1h/4h THOD/TLOD lines, for the index and DAILY_STOCKS"""
    files = []
    for tk in ['QQQ'] + [x for x in DAILY_STOCKS if x != 'QQQ']:              # QQQ in its own prices (not scaled to NQ)
        try:
            h1 = await run_retry(bk.hod_lod, tk, '1h'); h4 = await run_retry(bk.hod_lod, tk, '4h')
            ref = None
            files.append(discord.File(io.BytesIO(await run(GX.render, tk, ref, h1, h4)), filename=f"{tk.replace('=F', '')}_0930_gex.png"))
        except Exception as ex: print(f'{tk} GEX chart failed: {ex}')
    if files: await ch.send(files=files)


async def gex_prefetch_loop():
    """09:05 ET weekdays: pull this month's option EOD + OI from Theta Terminal for the GEX charts (needs Theta Terminal running)"""
    await client.wait_until_ready(); done = None
    while True:
        try:
            now = datetime.now(NYTZ)
            if now.weekday() < 5 and (now.hour, now.minute) >= (9, 5) and now.hour < 10 and done != now.date():
                done = now.date()
                for tk in ({'QQQ', *DAILY_STOCKS} if GX else []):
                    await run(GX.refresh, tk)
        except Exception as e:
            print(f'gex prefetch error: {e}')
        await asyncio.sleep(30)


async def post_after(st):
    """16:05 ET: the 09:30 charts again - lines exactly as issued, candles through the close"""
    iss = DA.load_issue()
    if iss.get('date') != str(datetime.now(NYTZ).date()): return
    for cid, t in list(st['channels'].items()):
        try:
            ch = client.get_channel(int(cid)) or await client.fetch_channel(int(cid)); files = []
            for tk in [t] + [x for x in DAILY_STOCKS if x != t]:
                m = iss['issued'].get(tk)
                if not m: continue
                try:
                    resp = await run_retry(bk.data, tk, '5m'); fn = DA.render_after_stock
                    files.append(discord.File(io.BytesIO(await run(fn, tk, resp, m)), filename=f'{tk}_1600_after.png'))
                except Exception as ex: print(f'{tk} after chart failed: {ex}')
            if files: await ch.send(files=files)
        except Exception as e:
            print(f'after post failed for {cid}: {e}')
    try:                                                   # research archive: issued levels + day bars + GEX profile
        import archive_day; print(await run(archive_day.archive, iss['date'], GX))
    except Exception as e:
        print(f'archive failed: {e}')


async def daily_loop():
    await client.wait_until_ready()
    while True:
        try:
            now = datetime.now(NYTZ); st = load_daily()
            if now.weekday() < 5 and (now.hour, now.minute) >= (9, 30) and now.hour < 12 and st.get('last') != str(now.date()) and st['channels']:
                st['last'] = str(now.date()); save_daily(st)          # mark first so a crash can't double-post
                await post_daily(st)
        except Exception as e:
            print(f'daily loop error: {e}')
        await asyncio.sleep(20)


async def after_loop():
    await client.wait_until_ready()
    while True:
        try:
            now = datetime.now(NYTZ); st = load_daily()
            if now.weekday() < 5 and (now.hour, now.minute) >= (16, 5) and now.hour < 20 and st.get('last_after') != str(now.date()) and st['channels']:
                st['last_after'] = str(now.date()); save_daily(st)          # mark first so a crash can't double-post
                await post_after(st)
        except Exception as e:
            print(f'after loop error: {e}')
        await asyncio.sleep(20)


DAILY_ACTIONS = [app_commands.Choice(name='turn on in this channel', value='on'), app_commands.Choice(name='turn off in this channel', value='off')]


@tree.command(name='daily', description='Post the 1h/4h THOD/TLOD charts in this channel every weekday at 09:30 ET')
@app_commands.describe(action='turn the 09:30 post on or off for this channel', ticker='default NQ')
@app_commands.choices(action=DAILY_ACTIONS)
async def daily_cmd(inter: discord.Interaction, action: app_commands.Choice[str], ticker: str = 'NQ'):
    st = load_daily(); cid = str(inter.channel_id)
    if action.value == 'on':
        st['channels'][cid] = norm_ticker(ticker); save_daily(st)
        await inter.response.send_message(f"On: {st['channels'][cid]} THOD/TLOD charts here every weekday at 09:30 ET (only while the bot machine is running). If you turn it on between 09:30 and noon on a weekday, today's chart goes out within a minute; otherwise the first one is the next weekday at 09:30.")
    else:
        st['channels'].pop(cid, None); save_daily(st)
        await inter.response.send_message('Off for this channel.')


SCREENER_ONLY = os.getenv('SCREENER_ONLY', '1') == '1'


@client.event
async def on_ready():
    if not getattr(client, '_warm_started', False):
        client._warm_started = True
        asyncio.get_running_loop().create_task(keep_warm())                # keeps the website awake
        if not SCREENER_ONLY:
            asyncio.get_running_loop().create_task(daily_loop())
            asyncio.get_running_loop().create_task(after_loop())
            asyncio.get_running_loop().create_task(gex_prefetch_loop())
    if getattr(client, '_synced', False): return       # reconnects fire on_ready again: register commands once per process only
    client._synced = True
    if SCREENER_ONLY:                                  # Oct 21 2026: THOD/TLOD / levels / GEX bot retired - Discord is the screener only (/check, /history)
        for name in ('levels', 'chart', 'daily'): tree.remove_command(name)
    await sync_commands()
    asyncio.get_running_loop().create_task(command_guard())
    import socket
    print(f'ready as {client.user} on {"render" if ON_RENDER else socket.gethostname()}')


async def sync_commands():
    guild = os.environ.get('DISCORD_GUILD_ID', '').split('#')[0].strip()
    if guild:                                   # guild sync is instant; global sync can take up to an hour
        g = discord.Object(id=int(guild))
        tree.copy_global_to(guild=g)
        return await tree.sync(guild=g)
    return await tree.sync()


_warned = {'t': 0.0}
async def notify_owner(text):
    """post at most once every 6 hours to the screener channel"""
    import time as _t
    if _t.time() - _warned['t'] < 6 * 3600: return
    ch = client.get_channel(int(os.environ.get('VECTOR_CHANNEL_ID', '0').split('#')[0].strip() or 0))
    if ch: await ch.send(text); _warned['t'] = _t.time()


async def command_guard():
    """Every 5 min: if Discord's registered command list is missing any of ours (another bot copy with older code
    overwrote it), put ours back. Logs it so the culprit is visible."""
    ours = {c.name for c in tree.get_commands()}
    while True:
        await asyncio.sleep(60)                                       # Oct 26: every minute (was 5) - commands kept vanishing between checks
        try:
            guild = os.environ.get('DISCORD_GUILD_ID', '').split('#')[0].strip()
            have = {c.name for c in await tree.fetch_commands(guild=discord.Object(id=int(guild)) if guild else None)}
            if ours - have:
                print(f'commands missing on Discord {sorted(ours - have)}; the other bot copy registered {sorted(have)} - re-registering')
                try: await notify_owner(f"⚠️ Another copy of the bot overwrote the slash commands (it registered: {', '.join(sorted(have)) or 'nothing'}). Restored /{', /'.join(sorted(ours - have))}. Find and stop the other copy (old Mac launchd job, another Render service, or another computer using the same bot token).")
                except Exception: pass
                await sync_commands()
        except Exception as e: print('command guard:', str(e)[:120])


if __name__ == '__main__':
    client.run(os.environ['DISCORD_TOKEN'])
