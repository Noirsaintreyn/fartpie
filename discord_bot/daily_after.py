"""4 pm 'after' charts: the 09:30 charts again with the lines EXACTLY as they were issued (saved at 09:30) and every candle that printed since.
bot.py saves the issued THOD/TLOD per ticker at 09:30 (save_issue) and calls the renderers below at 16:05 ET."""
import json
import os

import pandas as pd

import vchart as VC
import vterrain as VT

DATA_DIR = os.environ.get('DATA_DIR') or os.path.dirname(os.path.abspath(__file__))
ISSUE_FILE = os.path.join(DATA_DIR, 'daily_issue.json')


def save_issue(date, issued):
    json.dump({'date': date, 'issued': issued}, open(ISSUE_FILE, 'w'))


def load_issue():
    try:
        return json.load(open(ISSUE_FILE))
    except Exception:
        return {'date': '', 'issued': {}}


def issue_meta(resp, h1, h4, tfs=('1h', '4h'), h3=None):
    """what has to be remembered about a 09:30 chart: the data time it was issued at (last 5m bar start + 5 min) and the two THOD/TLOD sets
    (h1 / h4 keys kept for old files; tfs says which timeframes they actually are)"""
    last = pd.Timestamp(resp['priceData'][-1]['date']) + pd.Timedelta(minutes=5)
    return dict(issue=last.strftime('%Y-%m-%d %H:%M'), h1=h1, h4=h4, h3=h3, tfs=list(tfs))


def _split(resp, issue):
    cut = pd.Timestamp(issue).strftime('%Y-%m-%d %H:%M'); bars = resp['priceData']
    return [b for b in bars if b['date'][:16] < cut], [b for b in bars if b['date'][:16] >= cut]


def render_after_nq(t, resp, meta):
    """NQ: the odds chart as issued (no cones) + the candles after; touches after issue are dots on the lines"""
    before, after = _split(resp, meta['issue']); r2 = dict(resp); r2['priceData'] = before; r2['currentPrice'] = before[-1]['close']
    note = f"AFTER · lines as issued {pd.Timestamp(meta['issue']):%H:%M} ET · candles since (right of the dashed 12:00 mark = afternoon)"
    return VT.render(t, '5m', r2, meta['h1'], None, hl_lines=meta['h1'], targets='t1t2', thod_tf='1h', extra_lines=[('4h', meta['h4'])], cones=False, after=after, replay_note=note)


def render_after_stock(t, resp, meta):
    """stocks: the THOD/TLOD chart as issued + the candles after; ring = first touch after issue"""
    before, after = _split(resp, meta['issue']); r2 = dict(resp); r2['currentPrice'] = before[-1]['close']
    tfs = meta.get('tfs', ['1h', '4h']); srcs = [dict(tf=tf, resp=r2, hl=h, levels=[]) for tf, h in zip(tfs, [meta['h1'], meta['h4'], meta.get('h3')]) if h]
    bands, xlines = [], []
    try:
        import qr_zones as QZ
        for i, tf in enumerate(tfs[:2]):
            b_, x_ = QZ.zones((meta.get('qr') or {}).get(tf), [meta['h1'], meta['h4']][i], meta.get('qr_px') or r2['currentPrice'], tf, VC.STYLE[i]['thod']); bands += b_; xlines += x_
    except Exception as ex: print(f'{t} after QR zones skipped: {ex}')
    return VC.render(t, '5m', 'thod', r2, srcs, nbars=min(110, len(before)) + len(after), issue_time=pd.Timestamp(meta['issue']), bands=bands, xlines=xlines).getvalue()
