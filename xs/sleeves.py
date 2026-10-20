"""
Signal sleeves. Each function returns a wide (date x ticker) score known at the
CLOSE of `date` (only data through that close). Higher score = expected to
outperform. Tearsheet sector-neutralizes, so raw scale doesn't matter.
"""
import os

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfg import D, VH  # noqa: E402
_cache = {}


def _load():
    if not _cache:
        r = pd.read_parquet(os.path.join(D, 'resid.parquet'))
        _cache['res'], _cache['res_on'], _cache['res_id'] = r['res'], r['res_on'], r['res_id']
        p = pd.read_parquet(os.path.join(D, 'panel.parquet'), columns=['date', 'ticker', 'ret'])
        _cache['R'] = p.pivot(index='date', columns='ticker', values='ret').sort_index()
    return _cache


def _window_sum(x, start, end, min_frac=0.8):
    """sum of x over [t-start, t-end] trading days (inclusive of t-end)."""
    n = start - end + 1
    return x.shift(end).rolling(n, min_periods=int(n * min_frac)).sum()


def mom12_1_raw():
    R = _load()['R']
    return np.expm1(_window_sum(np.log1p(R), 252, 21))


def rmom12_1():
    return _window_sum(_load()['res'], 252, 21)


def rmom6_1():
    return _window_sum(_load()['res'], 126, 21)


def rrev5():
    return -_window_sum(_load()['res'], 4, 0)


def rrev21():
    return -_window_sum(_load()['res'], 20, 0)


def on60():
    return _window_sum(_load()['res_on'], 59, 0)


def id60():
    return _window_sum(_load()['res_id'], 59, 0)


def on_minus_id60():
    c = _load()
    return _window_sum(c['res_on'], 59, 0) - _window_sum(c['res_id'], 59, 0)


def on20():
    return _window_sum(_load()['res_on'], 19, 0)


def _event_wide(col, max_age=60, src='events.parquet', when='avail'):
    ev = pd.read_parquet(os.path.join(D, src)).dropna(subset=[when, col])
    idx = _load()['res'].index
    w = ev.sort_values(when).drop_duplicates(['ticker', when], keep='last').pivot(
        index=when, columns='ticker', values=col)
    return w.reindex(idx).ffill(limit=max_age)


def _prank(w):
    return w.rank(axis=1, pct=True)


def pead_sue():
    return _event_wide('sue')


def pead_rev():
    return _event_wide('sue_rev')


def pead_ear():
    return _event_wide('ear')


def pead_combo():
    a, b, c = _prank(pead_sue()), _prank(pead_rev()), _prank(pead_ear())
    return pd.concat([a, b, c]).groupby(level=0).mean()


def pead_combo_fresh():
    return pead_combo().where(_event_wide('sue', max_age=15).notna())


# ---- exact 8-K release dates: usable at close of the reaction day ----
def pead_sue_x():
    return _event_wide('sue', when='rd_x')


def pead_rev_x():
    return _event_wide('sue_rev', when='rd_x')


def pead_ear_x():
    return _event_wide('ear_x', when='rd_x')


def pead_combo_x():
    a, b, c = _prank(pead_sue_x()), _prank(pead_rev_x()), _prank(pead_ear_x())
    return pd.concat([a, b, c]).groupby(level=0).mean()


def pead_combo_x_fresh():
    return pead_combo_x().where(_event_wide('sue', max_age=15, when='rd_x').notna())


def _ins(col):
    w = pd.read_parquet(os.path.join(D, 'insider_daily.parquet')).pivot(
        index='date', columns='ticker', values=col)
    return w.reindex(_load()['res'].index)


def ins_npr126():
    return _ins('npr126')


def ins_buyers63():
    # tie-break zeros by net purchase ratio so quintiles aren't alphabetical
    return _ins('buyers63') + 0.01 * _ins('npr126').fillna(-1)


def ins_offbuy63():
    return _ins('offbuy63') + 0.01 * _ins('npr126').fillna(-1)


_vh = {}


def _volhist(col):
    """DoltHub volatility_history (weekly, dated Sat = Fri close). Usable from the first
    trading day >= date (conservative), held until the next print (max 10 sessions)."""
    if 'df' not in _vh:
        v = pd.read_csv(VH,
                        usecols=['date', 'act_symbol', 'hv_current', 'iv_current', 'iv_week_ago',
                                 'iv_month_ago', 'iv_year_high', 'iv_year_low'])
        v['date'] = pd.to_datetime(v['date'])
        v['act_symbol'] = v['act_symbol'].str.replace('.', '-', regex=False)
        _vh['df'] = v
    v = _vh['df']
    idx = _load()['res'].index
    cols = set(_load()['res'].columns)
    from panel_build import ALIAS
    inv = {}
    for t in cols:
        inv.setdefault(ALIAS.get(t, t).replace('.', '-'), []).append(t)
        inv.setdefault(t.replace('.', '-'), []).append(t)
    vals = v.assign(x=col(v)).dropna(subset=['x'])
    k = idx.searchsorted(vals['date'])
    vals = vals[k < len(idx)].assign(avail=idx[k[k < len(idx)]])
    rows = [(a, t, x) for s_, a, x in zip(vals.act_symbol, vals.avail, vals.x) for t in set(inv.get(s_, []))]
    w = pd.DataFrame(rows, columns=['avail', 'ticker', 'x']).drop_duplicates(
        ['avail', 'ticker'], keep='last').pivot(index='avail', columns='ticker', values='x')
    return w.reindex(idx).ffill(limit=10)


# pre-registered directions (from the literature), NOT fitted:
def neg_iv_minus_rv():      # rich implied vs realized -> lower future returns
    return -_volhist(lambda v: v.iv_current - v.hv_current)


def neg_iv_chg_1m():        # rising IV -> lower future returns
    return -_volhist(lambda v: v.iv_current - v.iv_month_ago)


def neg_iv_chg_1w():
    return -_volhist(lambda v: v.iv_current - v.iv_week_ago)


def neg_iv_rank():          # IV high within its own 1y range -> lower future returns
    return -_volhist(lambda v: (v.iv_current - v.iv_year_low) / (v.iv_year_high - v.iv_year_low))


def neg_accruals():
    return -_event_wide('accruals', max_age=400, src='annual_sig.parquet')


def neg_asset_growth():
    return -_event_wide('asset_growth', max_age=400, src='annual_sig.parquet')


SLEEVES = {
    'mom12_1_raw': mom12_1_raw,     # benchmark
    'rmom12_1': rmom12_1,
    'rmom6_1': rmom6_1,
    'rrev5': rrev5,
    'rrev21': rrev21,
    'on20': on20,
    'on60': on60,
    'id60': id60,
    'on_minus_id60': on_minus_id60,
    'pead_sue': pead_sue,
    'pead_rev': pead_rev,
    'pead_ear': pead_ear,
    'pead_combo': pead_combo,
    'pead_combo_fresh': pead_combo_fresh,
    'pead_sue_x': pead_sue_x,
    'pead_rev_x': pead_rev_x,
    'pead_ear_x': pead_ear_x,
    'pead_combo_x': pead_combo_x,
    'pead_combo_x_fresh': pead_combo_x_fresh,
    'ins_npr126': ins_npr126,
    'ins_buyers63': ins_buyers63,
    'ins_offbuy63': ins_offbuy63,
    'neg_iv_minus_rv': neg_iv_minus_rv,
    'neg_iv_chg_1m': neg_iv_chg_1m,
    'neg_iv_chg_1w': neg_iv_chg_1w,
    'neg_iv_rank': neg_iv_rank,
    'neg_accruals': neg_accruals,
    'neg_asset_growth': neg_asset_growth,
}
