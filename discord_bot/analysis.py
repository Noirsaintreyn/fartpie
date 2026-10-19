"""Pure helpers: flatten/merge levels from /api/data, ATR, theory checks, chart rendering. No Discord/network here."""
import io

import numpy as np

MERGE_PCT = 0.0010          # levels within 0.10% of each other are shown as one


def atr14(bars):
    h = np.array([b['high'] for b in bars]); l = np.array([b['low'] for b in bars]); c = np.array([b['close'] for b in bars])
    if len(c) < 15:
        return float(np.mean(h - l))
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    a = tr[:14].mean()
    for x in tr[14:]:
        a = (a * 13 + x) / 14
    return float(a)


def flatten_levels(resp):
    """One row per distinct price: price, sources (categories), strength, touches, ml score, reversion prob."""
    rows = []
    lv = resp.get('levels', {})
    for key in ('structural', 'event', 'interaction', 'mlConfluence'):
        for l in lv.get(key, []) or []:
            p = l.get('price')
            if not isinstance(p, (int, float)):
                continue
            rows.append({'price': float(p), 'cats': [l.get('category') or l.get('source') or key],
                         'strength': float(l.get('strength') or l.get('level_strength') or 0),
                         'touches': int(l.get('touches') or 0),
                         'ml': l.get('ml_filter_score'), 'rev': l.get('reversionProb')})
    rows.sort(key=lambda r: r['price'])
    merged = []
    for r in rows:
        if merged and abs(r['price'] - merged[-1]['price']) / merged[-1]['price'] <= MERGE_PCT:
            m = merged[-1]
            top = r if r['strength'] > m['strength'] else m
            m.update(price=top['price'], strength=max(m['strength'], r['strength']), touches=max(m['touches'], r['touches']),
                     ml=m['ml'] if m['ml'] is not None else r['ml'], rev=top['rev'] if top['rev'] is not None else m['rev'])
            m['cats'] = sorted(set(m['cats'] + r['cats']))
            m['n'] += 1
        else:
            r['n'] = 1
            merged.append(r)
    return merged


def split_sides(levels, price):
    sup = sorted([l for l in levels if l['price'] < price], key=lambda l: -l['price'])
    res = sorted([l for l in levels if l['price'] >= price], key=lambda l: l['price'])
    return sup, res


def short_cat(cats):
    m = {'Density (HDBSCAN)': 'HDBSCAN', 'HDBSCAN-MultiScale': 'HDBSCAN-MS', 'HDBSCAN-TimeWeighted': 'HDBSCAN-TW',
         'Isolation-Forest': 'IsoForest', 'Local Interaction': 'Interaction'}
    return '+'.join(m.get(c, c) for c in cats)


def level_line(l, price, atr):
    d = l['price'] - price
    bits = [f"`{l['price']:,.2f}`", f"{d / price * 100:+.2f}%", f"{d / atr:+.1f} ATR", short_cat(l['cats'])]
    if l['n'] > 1:
        bits.append(f"x{l['n']} agree")
    return ' · '.join(bits)


def parse_numbers(text):
    import re
    return [float(x.replace(',', '')) for x in re.findall(r'-?\d[\d,]*\.?\d*', text)]


def theory_levels(prices, levels, price, atr, er=None):
    out = []
    for p in prices:
        near = min(levels, key=lambda l: abs(l['price'] - p)) if levels else None
        d_atr = abs(near['price'] - p) / atr if near else None
        side = 'support (below price)' if p < price else 'resistance (above price)'
        out.append({'price': p, 'side': side, 'dist_pct': (p - price) / price * 100, 'dist_atr': (p - price) / atr,
                    'near': near, 'near_atr': d_atr, 'on_level': d_atr is not None and d_atr <= 0.25,
                    'in_expected': (er['expected_low'] <= p <= er['expected_high']) if er else None})
    return out


def theory_range(lo, hi, levels, price, atr, er=None):
    lo, hi = min(lo, hi), max(lo, hi)
    inside = [l for l in levels if lo <= l['price'] <= hi]
    below = max([l for l in levels if l['price'] < lo], key=lambda l: l['price'], default=None)
    above = min([l for l in levels if l['price'] > hi], key=lambda l: l['price'], default=None)
    res = {'lo': lo, 'hi': hi, 'width': hi - lo, 'width_atr': (hi - lo) / atr, 'width_pct': (hi - lo) / price * 100,
           'inside': inside, 'below': below, 'above': above, 'contains_price': lo <= price <= hi}
    if er:
        ew = er['expected_high'] - er['expected_low']
        ov = max(0.0, min(hi, er['expected_high']) - max(lo, er['expected_low']))
        res['er'] = {'low': er['expected_low'], 'high': er['expected_high'], 'width_ratio': (hi - lo) / ew,
                     'overlap_of_yours': ov / (hi - lo) if hi > lo else 0, 'overlap_of_model': ov / ew}
    return res


def chart_png(resp, levels, ticker, tf, er=None, nbars=90):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bars = resp['priceData'][-nbars:]
    price = resp['currentPrice']
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=110)
    fig.patch.set_facecolor('#0f1116'); ax.set_facecolor('#0f1116')
    for i, b in enumerate(bars):
        up = b['close'] >= b['open']
        c = '#26a69a' if up else '#ef5350'
        ax.plot([i, i], [b['low'], b['high']], color=c, lw=0.8)
        ax.add_patch(plt.Rectangle((i - 0.3, min(b['open'], b['close'])), 0.6, max(abs(b['close'] - b['open']), 1e-9), color=c))
    lo_v = min(b['low'] for b in bars); hi_v = max(b['high'] for b in bars)
    span = hi_v - lo_v
    lo_v -= span * 0.25; hi_v += span * 0.25
    if er:
        ax.axhspan(er['expected_low'], er['expected_high'], color='#7e57c2', alpha=0.13, lw=0)
    for l in levels:
        if lo_v <= l['price'] <= hi_v:
            col = '#66bb6a' if l['price'] < price else '#ff7043'
            ax.axhline(l['price'], color=col, lw=0.9 + 1.4 * min(l['strength'], 1), alpha=0.75, ls='-' if l['n'] > 1 else '--')
            ax.text(len(bars) + 0.5, l['price'], f"{l['price']:,.1f}", color=col, va='center', fontsize=7.5)
    ax.axhline(price, color='#ffffff', lw=0.8, ls=':')
    ax.set_xlim(-1, len(bars) + 9); ax.set_ylim(lo_v, hi_v)
    ax.tick_params(colors='#9aa0a6', labelsize=8)
    for s in ax.spines.values():
        s.set_color('#2a2f3a')
    step = max(1, len(bars) // 6)
    ax.set_xticks(range(0, len(bars), step)); ax.set_xticklabels([bars[i]['date'][5:16] for i in range(0, len(bars), step)])
    ax.set_title(f'{ticker} {tf} - levels' + ('  (purple = expected range today)' if er else ''), color='#e8eaed', fontsize=11)
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor()); plt.close(fig)
    buf.seek(0)
    return buf


_SHORT = {'Density (HDBSCAN)': 'HDBSCAN', 'Isolation-Forest': 'IsoForest', 'Multiscale-HDBSCAN': 'MS-HDBSCAN', 'Time-Weighted HDBSCAN': 'TW-HDBSCAN'}


def all_levels(resp):
    """Every level as the site lists them, but ONE row per price (the backend repeats a merged level once per category).
    name = every detector that contributed (the backend's `sources`), strongest-first category leading, e.g. 'KDE+TDA+GMM+MeanShift+IsoForest (6)'.
    Sorted high -> low."""
    by_price = {}
    for key in ('structural', 'event', 'interaction', 'mlConfluence'):
        for l in resp.get('levels', {}).get(key, []) or []:
            p = l.get('price')
            if not isinstance(p, (int, float)):
                continue
            e = by_price.setdefault(round(float(p), 2), {'price': float(p), 'cats': [], 'strength': 0.0, 'merged': 0})
            for c in [l.get('type') or l.get('category') or l.get('source') or key] + list(l.get('sources') or []):
                c = _SHORT.get(str(c), str(c))
                if c not in e['cats']:
                    e['cats'].append(c)
            e['strength'] = max(e['strength'], float(l.get('strength') or l.get('level_strength') or 0) * 100)
            e['merged'] = max(e['merged'], int(l.get('merged_count') or 0))
    out = []
    for e in by_price.values():
        name = '+'.join(e['cats']) + (f" ({e['merged']})" if e['merged'] > len(e['cats']) else '')
        out.append({'price': e['price'], 'name': name, 'strength': e['strength']})
    out.sort(key=lambda x: -x['price'])
    return out


def level_rows(levels, price):
    return [f"{l['price']:>10,.2f}  {l['name'][:40]:<40} {l['strength']:>3.0f}%  {(l['price'] - price) / price * 100:+6.2f}%" for l in levels]


def chunk_block(header, rows, limit=3800):
    """Split rows into code-block strings that fit an embed description."""
    chunks, cur = [], header
    for r in rows:
        if len(cur) + len(r) + 1 > limit:
            chunks.append(cur); cur = ''
        cur += ('\n' if cur else '') + r
    chunks.append(cur)
    return [f'```\n{c}\n```' for c in chunks if c]


def tp_chart_png(j, ticker, tf, direction):
    """Simulated OU paths from entry, coloured by outcome, with entry/stop/target lines."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sp = j['sample_paths']
    paths = np.array(sp['paths_price']); out = sp['outcome']
    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
    fig.patch.set_facecolor('#0f1116'); ax.set_facecolor('#0f1116')
    col = {'target': '#66bb6a', 'stop': '#ef5350', '': '#6b7280'}
    for p, o, h in zip(paths, out, sp['hit_step']):
        end = len(p) - 1 if h < 0 else h
        ax.plot(range(end + 1), p[:end + 1], color=col.get(o, '#6b7280'), lw=0.9, alpha=0.55)
    for y, c_, lab in ((j['target'], '#66bb6a', 'target'), (j['entry'], '#ffffff', 'entry'), (j['stop'], '#ef5350', 'stop')):
        ax.axhline(y, color=c_, lw=1.3, ls='--' if lab != 'entry' else ':')
        ax.text(paths.shape[1] - 1, y, f' {lab} {y:,.2f}', color=c_, va='center', fontsize=8)
    ax.set_xlim(0, paths.shape[1] + paths.shape[1] * 0.18)
    ax.set_xlabel(f'{tf} bars ahead', color='#9aa0a6', fontsize=9)
    ax.tick_params(colors='#9aa0a6', labelsize=8)
    for s in ax.spines.values():
        s.set_color('#2a2f3a')
    ax.set_title(f"{ticker} {direction}: P(target first) = {j['calibrated_probability']:.0%} calibrated ({j['raw_probability']:.0%} raw)  |  {len(paths)} sample paths",
                 color='#e8eaed', fontsize=10)
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor()); plt.close(fig)
    buf.seek(0)
    return buf
