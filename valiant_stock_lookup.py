"""
Interactive per-stock diagnostic tool: search a ticker + timeframe, get a
classification of current state (bullish / bearish / consolidation / not
confident) based on AGREEMENT between the stock's own valiant (price
structure) and the broader market regime (HMM on SPY), plus a move-size
read (quiet / normal / elevated) from Yang-Zhang volatility - not a return
prediction, not a trade signal. This deliberately does NOT rank stocks or
generate picks - that was tested and retired (see memory / this session's
eligibility-gate test). This describes state and flags disagreement.

Also reports a CONFIRMATION-TIMEFRAME read: the valiant on a finer
timeframe than the one queried, scaled adaptively (daily query -> 4h
confirmation via LSE intraday data; weekly/monthly query -> daily
confirmation). Purely descriptive context ("does the faster timeframe
agree"), not a filter and not a new claim of predictive skill - the
automated version of this idea (requiring agreement) was tested earlier
tonight and made the automated picker worse, which is a different
question from whether it's useful CONTEXT to show a human.

Usage: .venv311/bin/python valiant_stock_lookup.py TICKER [daily|weekly|monthly]
"""
import json
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

from fetch_lse_ohlcv import fetch_ohlcv
from vector_fundamentals_backtest import fundamentals_asof, load_facts_cache, get_facts_for_ticker
from vector_screener_sp500_sectors import load_sectors, batch_download_cached

from backtest_pivot_ratchet import run_state_machine, regime_strip_agreement, ATR_LEN, SLOW_LEN, PIVOT_LEN

warnings.filterwarnings('ignore', category=RuntimeWarning)

HORIZON = 20
RECENT_FLIP_WINDOW = 20
MIN_FLIPS_FOR_QUALITY = 5
CHOPPY_FLIP_COUNT = 3      # this many flips within CHOPPY_WINDOW bars => consolidation
CHOPPY_WINDOW = 30
HMM_STRESS_THRESHOLD = 0.5
YZ_WINDOW = 20
YZ_LOOKBACK_FOR_PERCENTILE = 252
PEER_BREADTH_TREND_LOOKBACK = 20  # bars ago to compare breadth against, for a strengthening/weakening read
PEER_MAX_COUNT = 25                # cap on sector peers fetched, for very large sectors -
                                    # kept small deliberately: a live server request fetching
                                    # dozens of peers' full history at once is real memory
                                    # pressure on a small instance (this caused a production
                                    # OOM crash at PEER_MAX_COUNT=60 with period='max')
PEER_FETCH_PERIOD = '2y'           # plenty for run_state_machine's ~60-bar warmup + the
                                    # 90-day trailing correlation window, far less memory
                                    # than fetching each peer's full decades-long history
PEER_CORR_WINDOW = 90


def resample_ohlc(daily, freq):
    d = daily.copy()
    if freq == 'weekly':
        iso = d['datetime'].dt.isocalendar()
        d['key1'], d['key2'] = iso['year'], iso['week']
    elif freq == 'monthly':
        d['key1'], d['key2'] = d['datetime'].dt.year, d['datetime'].dt.month
    else:
        return d
    bars = d.groupby(['key1', 'key2'], sort=False).agg(
        datetime=('datetime', 'min'), open=('open', 'first'),
        high=('high', 'max'), low=('low', 'min'), close=('close', 'last'),
    ).reset_index(drop=True)
    return bars.sort_values('datetime').reset_index(drop=True)


def yang_zhang_vol(df, window=YZ_WINDOW, annualize_periods=252):
    """Rolling Yang-Zhang volatility estimate - uses full OHLC, accounts
    for overnight gaps and drift, more sample-efficient than close-to-close
    historical vol. Returns an array aligned to df's index (NaN until the
    window fills)."""
    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    n = len(df)
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan

    log_o_prevc = np.log(o / prev_c)
    log_c_o = np.log(c / o)
    log_h_c = np.log(h / c)
    log_h_o = np.log(h / o)
    log_l_c = np.log(l / c)
    log_l_o = np.log(l / o)
    rs = log_h_c * log_h_o + log_l_c * log_l_o

    yz = np.full(n, np.nan)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    for i in range(window, n):
        seg_o = log_o_prevc[i - window + 1:i + 1]
        seg_c = log_c_o[i - window + 1:i + 1]
        seg_rs = rs[i - window + 1:i + 1]
        var_o = np.nanvar(seg_o, ddof=1)
        var_c = np.nanvar(seg_c, ddof=1)
        var_rs = np.nanmean(seg_rs)
        yz_var = var_o + k * var_c + (1 - k) * max(var_rs, 0)
        yz[i] = np.sqrt(yz_var * annualize_periods)
    return yz


def fetch_daily(ticker):
    # raw (dividend-unadjusted) prices for flip detection - confirmed
    # tonight to match TradingView's actual displayed OHLC, unlike
    # auto_adjust=True which this tool used until now
    df = yf.download(ticker, period='max', interval='1d', auto_adjust=False, progress=False)
    df = df.reset_index()
    df.columns = [str(c[0] if isinstance(c, tuple) else c).lower() for c in df.columns]
    df = df.rename(columns={'date': 'datetime', 'adj close': 'adjclose'})
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    return df


def fit_hmm_stress_prob(weekly_log_rets):
    if len(weekly_log_rets) < 52:
        return np.nan
    X = weekly_log_rets.reshape(-1, 1)
    try:
        model = GaussianHMM(n_components=2, covariance_type='diag', n_iter=50, random_state=42)
        model.fit(X)
        probs = model.predict_proba(X)
        means = model.means_[:, 0]
        stress_state = int(np.argmin(means))
        return float(probs[-1, stress_state])
    except Exception:
        return np.nan


def market_regime_context():
    spy = fetch_daily('SPY')
    spy_weekly = spy.set_index('datetime')['close'].resample('W-FRI').last().dropna()
    spy_weekly_logret = np.log(spy_weekly / spy_weekly.shift(1)).dropna()
    stress_prob = fit_hmm_stress_prob(spy_weekly_logret.values)
    return stress_prob


CONFIRMATION_TIMEFRAME = {'daily': '4h', 'weekly': 'daily', 'monthly': 'daily'}


def confirmation_regime(ticker, primary_timeframe, daily_df):
    """Valiant regime on a finer timeframe than the one queried, scaled
    adaptively: daily -> 4h (fetched live via LSE intraday), weekly/
    monthly -> daily (already have it). Returns (regime, label) or
    (None, reason) if unavailable."""
    conf_tf = CONFIRMATION_TIMEFRAME[primary_timeframe]
    if conf_tf == 'daily':
        bars = daily_df
    else:
        end = datetime.now()
        start = end - timedelta(days=720)  # LSE intraday depth is limited; ~2 years of 4h
        try:
            bars = fetch_ohlcv(ticker, conf_tf, start, end)
        except Exception as e:
            return None, f"could not fetch {conf_tf} data ({e})"
        if bars is None or len(bars) < 150:
            return None, f"insufficient {conf_tf} history available"
        bars = bars.rename(columns=str.lower)

    if len(bars) < 100:
        return None, f"insufficient {conf_tf} history"
    regimes, levels, atr, flips = run_state_machine(bars)
    warmup = max(ATR_LEN, SLOW_LEN, PIVOT_LEN * 2) + 5
    idx = len(bars) - 1
    if idx < warmup + 10:
        return None, f"insufficient {conf_tf} history past warmup"
    return int(regimes[idx]), conf_tf


def regime_maturity(flips, current_regime, bars_since_flip):
    """How far along the current move is, relative to this stock's own
    historical episodes of the same direction - completed episodes only
    (the current one isn't finished, so it's excluded from the stats it's
    being compared against)."""
    completed = []
    for i in range(len(flips) - 1):
        if flips[i][1] == current_regime:
            completed.append(flips[i + 1][0] - flips[i][0])
    if len(completed) < 3:
        return None
    completed = np.array(completed)
    mean_len = float(completed.mean())
    pct_of_avg = (bars_since_flip / mean_len * 100) if mean_len > 0 else np.nan
    percentile = float((completed < bars_since_flip).mean() * 100)
    return {'mean': mean_len, 'median': float(np.median(completed)),
            'pct_of_avg': pct_of_avg, 'percentile': percentile, 'n': len(completed)}


def fundamentals_rating(ticker, as_of_date):
    facts_by_ticker = load_facts_cache()
    gaap = get_facts_for_ticker(ticker, facts_by_ticker)
    if gaap is None:
        return None
    result = fundamentals_asof(gaap, as_of_date)
    return result


def fundamentals_narrative(fund):
    """Plain-language read of the actual fundamentals numbers, not just
    a pass/fail count - the tool's reasoning should read like a
    fundamentals+macro analyst's note, not a technical-indicator log."""
    if fund is None:
        return "fundamentals data unavailable"
    parts = []
    pm = fund.get('profit_margin')
    if pm is not None:
        parts.append(f"{'strong' if pm > 0.15 else 'modest' if pm > 0 else 'negative'} profit margin ({pm*100:+.1f}%)")
    rg = fund.get('revenue_growth')
    if rg is not None:
        parts.append(f"{'strong' if rg > 0.10 else 'modest' if rg > 0 else 'shrinking'} revenue growth ({rg*100:+.1f}% YoY)")
    dte = fund.get('debt_to_equity')
    if dte is not None:
        parts.append(f"{'conservative' if dte < 50 else 'moderate' if dte < 150 else 'high'} leverage (D/E {dte:.0f}%)")
    cr = fund.get('current_ratio')
    if cr is not None:
        parts.append(f"{'strong' if cr >= 1.5 else 'adequate' if cr >= 1.0 else 'weak'} liquidity (current ratio {cr:.1f}x)")
    if not parts:
        return "fundamentals data incomplete"
    return ", ".join(parts)


def macro_narrative(market_stress_prob, market_stressed):
    if np.isnan(market_stress_prob):
        return "macro backdrop unavailable"
    if market_stressed:
        return f"a stressed macro backdrop (broad-market stress probability {market_stress_prob*100:.0f}%)"
    return f"a calm macro backdrop (broad-market stress probability {market_stress_prob*100:.0f}%)"


def candle_shape_stats(bars, regimes, current_regime, idx):
    """Typical candle shape for bars historically in the SAME regime
    direction as right now - decomposed into four ATR-normalized
    excursions: open->high and low->close (upside character, "green"),
    high->close and open->low (downside character, "red")."""
    o, h, l, c = bars['open'].values, bars['high'].values, bars['low'].values, bars['close'].values
    atr = compute_bar_atr(bars)
    mask = (regimes[:idx] == current_regime) & (atr[:idx] > 0) & ~np.isnan(atr[:idx])
    if mask.sum() < 20:
        return None
    oh = (h[:idx][mask] - o[:idx][mask]) / atr[:idx][mask]
    lc = (c[:idx][mask] - l[:idx][mask]) / atr[:idx][mask]
    hc = (h[:idx][mask] - c[:idx][mask]) / atr[:idx][mask]
    ol = (o[:idx][mask] - l[:idx][mask]) / atr[:idx][mask]
    return {
        'n': int(mask.sum()),
        'open_to_high': float(np.mean(oh)),
        'low_to_close': float(np.mean(lc)),
        'high_to_close': float(np.mean(hc)),
        'open_to_low': float(np.mean(ol)),
    }


def compute_bar_atr(bars, window=ATR_LEN):
    h, l, c = bars['high'].values, bars['low'].values, bars['close'].values
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).ewm(span=window, min_periods=window).mean().values


def expected_move(bars):
    """GJR-GARCH(1,1,1) skew-t 1-bar-ahead quantile forecast, anchored to
    the current close - the exact same methodology already validated
    for NQ/ES in backend.py's macro-regime expected-range feature
    (backtest_returns_distribution.py: best of 5 candidates tested,
    ~5-8% lower pinball loss than a symmetric normal, fewest calibration
    violations), reimplemented standalone here rather than imported
    from backend.py - avoids pulling in that whole module's own import
    chain for two small, genuinely asset-agnostic functions.

    "1 bar ahead" means whatever timeframe `bars` already is - next
    day for daily, next week for weekly, next month for monthly, same
    bar-resolution semantics as everything else in this tool. Not
    validated specifically on individual equities (only NQ/ES) - this
    is the same model applied to a new asset class, not a re-proof."""
    closes = bars['close'].values
    highs, lows = bars['high'].values, bars['low'].values
    if len(closes) < 80:
        return None
    try:
        from arch import arch_model
        from arch.univariate import SkewStudent
    except ImportError:
        return None

    returns_pct = np.diff(np.log(closes)) * 100
    try:
        model = arch_model(returns_pct, vol='GARCH', p=1, o=1, q=1, dist='skewt', rescale=False)
        result = model.fit(disp='off', show_warning=False)
        forecast = result.forecast(horizon=1, reindex=False)
        mean = float(forecast.mean.values[-1, 0])
        sigma = float(np.sqrt(forecast.variance.values[-1, 0]))
        if not np.isfinite(mean) or not np.isfinite(sigma) or sigma <= 0:
            return None
        eta, lam = result.params['eta'], result.params['lambda']
        dist_obj = SkewStudent()
        quantiles_pct = {q: mean + sigma * float(dist_obj.ppf(q, [eta, lam]))
                          for q in [0.10, 0.30, 0.50, 0.70, 0.90]}
    except Exception:
        return None

    current_price = float(closes[-1])
    bounds = {q: current_price * np.exp(v / 100) for q, v in quantiles_pct.items()}
    range_width_pct = quantiles_pct[0.70] - quantiles_pct[0.30]

    # context: this bar-size's own trailing realized range, as the
    # "normal" baseline - grounded in the stock's own history, not VIX
    # (which doesn't apply to an individual equity the way it does to
    # index futures)
    realized_log_range = np.log(highs[-60:] / lows[-60:])
    typical_range_pct = float(np.mean(realized_log_range)) * 100 if len(realized_log_range) >= 20 else None
    vs_typical_pct = ((range_width_pct - typical_range_pct) / typical_range_pct * 100
                       if typical_range_pct and typical_range_pct > 0 else None)

    return {
        'current_price': current_price,
        'low_30': bounds[0.30], 'high_70': bounds[0.70],
        'low_10': bounds[0.10], 'high_90': bounds[0.90],
        'median': bounds[0.50],
        'range_width_pct': range_width_pct,
        'typical_range_pct': typical_range_pct,
        'vs_typical_pct': vs_typical_pct,
    }


def peer_correlation_regime(peer_closes, window=PEER_CORR_WINDOW):
    """Rough average pairwise correlation among sector peers' trailing
    daily returns - a correlation/volatility-regime read (high = peers
    moving as one block, "risk-on/risk-off" driven; low = idiosyncratic,
    stock-specific drivers dominate). Approximate: aligns peers by
    position (last `window` closes each), not by calendar date - fine
    for same-exchange US equities over a ~90-day window, not exact."""
    series = {}
    for t, closes in peer_closes.items():
        if len(closes) < window + 1:
            continue
        rets = pd.Series(closes[-(window + 1):]).pct_change().dropna().reset_index(drop=True)
        series[t] = rets
    if len(series) < 3:
        return None
    mat = pd.DataFrame(series)
    corr = mat.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return float(np.nanmean(upper.values))


def load_subindustries():
    """Real GICS-level sub-industry (e.g. "Semiconductors"), from
    yfinance's .info - a one-time batch build (build_sp500_subindustries.py),
    not fetched live here. Missing/empty if that build hasn't been run
    yet or a given ticker wasn't resolved; callers fall back to the
    broad 11-sector grouping in that case."""
    try:
        with open('sp500_subindustries.json') as f:
            raw = json.load(f)
        return {k: v['industry'] for k, v in raw.items() if v.get('industry')}
    except FileNotFoundError:
        return {}


def peer_breadth(ticker, timeframe):
    """Peer/sector breadth: how many of this stock's peers currently
    agree with its valiant state - context, not a vote. Uses real GICS
    sub-industry (e.g. "Semiconductors") when available - true peers,
    not "all of Information Technology" - and falls back to the broad
    11-sector grouping only for tickers the sub-industry build didn't
    resolve."""
    subindustries = load_subindustries()
    sectors = load_sectors()

    group = subindustries.get(ticker)
    grouping_label = 'sub-industry'
    if group is None:
        group = sectors.get(ticker)
        grouping_label = 'broad sector'
    if group is None:
        return None, f"no sector or sub-industry classification available for {ticker}"

    lookup = subindustries if grouping_label == 'sub-industry' else sectors
    peers = [t for t, g in lookup.items() if g == group and t != ticker]
    if len(peers) < 3:
        # sub-industry group too small (or ticker missing from it) - widen
        # to the broad sector rather than give up on breadth entirely
        broad = sectors.get(ticker)
        if broad is not None and grouping_label == 'sub-industry':
            group, grouping_label = broad, 'broad sector'
            peers = [t for t, s in sectors.items() if s == broad and t != ticker]
    if len(peers) < 3:
        return None, f"too few peers ({len(peers)}) for a meaningful breadth read"
    if len(peers) > PEER_MAX_COUNT:
        peers = peers[:PEER_MAX_COUNT]

    raw = batch_download_cached(peers, period=PEER_FETCH_PERIOD)
    warmup = max(ATR_LEN, SLOW_LEN, PIVOT_LEN * 2) + 5
    peer_states, peer_closes = [], {}

    for t, df in raw.items():
        bars = resample_ohlc(df, timeframe) if timeframe != 'daily' else df
        if len(bars) < 100:
            continue
        try:
            regimes, levels, atr, flips = run_state_machine(bars)
        except Exception:
            continue
        n = len(bars)
        idx = n - 1
        if idx < warmup + 10:
            continue

        cur_regime = int(regimes[idx])
        last_flip = flips[-1] if flips else None
        bars_since = idx - last_flip[0] if last_flip else None
        a = atr[idx]
        lvl = levels[idx]
        dist_atr = abs(bars['close'].values[idx] - lvl) / a if (not np.isnan(a) and a > 0 and not np.isnan(lvl)) else np.nan

        idx_prior = idx - PEER_BREADTH_TREND_LOOKBACK
        prior_regime = int(regimes[idx_prior]) if idx_prior >= warmup else None

        peer_states.append({'ticker': t, 'regime': cur_regime, 'bars_since_flip': bars_since,
                             'dist_atr': dist_atr, 'prior_regime': prior_regime})
        peer_closes[t] = bars['close'].values

    if len(peer_states) < 3:
        return None, f"too few peers had usable data ({len(peer_states)}/{len(peers)})"

    n_peers = len(peer_states)
    n_bullish = sum(1 for p in peer_states if p['regime'] == 1)
    pct_bullish = n_bullish / n_peers * 100

    recent_bullish_flips = sum(1 for p in peer_states
                                if p['bars_since_flip'] is not None
                                and p['bars_since_flip'] <= RECENT_FLIP_WINDOW and p['regime'] == 1)
    recent_bearish_flips = sum(1 for p in peer_states
                                if p['bars_since_flip'] is not None
                                and p['bars_since_flip'] <= RECENT_FLIP_WINDOW and p['regime'] == -1)

    dists = [p['dist_atr'] for p in peer_states if not np.isnan(p['dist_atr'])]
    median_dist = float(np.median(dists)) if dists else np.nan

    prior_valid = [p for p in peer_states if p['prior_regime'] is not None]
    if prior_valid:
        pct_bullish_prior = sum(1 for p in prior_valid if p['prior_regime'] == 1) / len(prior_valid) * 100
        delta = pct_bullish - pct_bullish_prior
        breadth_trend = 'STRENGTHENING' if delta > 10 else 'WEAKENING' if delta < -10 else 'STABLE'
    else:
        pct_bullish_prior, breadth_trend = None, None

    dispersion = 'BROAD AGREEMENT' if (pct_bullish >= 75 or pct_bullish <= 25) else 'MIXED'
    avg_corr = peer_correlation_regime(peer_closes)

    return {
        'sector': group, 'grouping': grouping_label, 'n_peers': n_peers, 'pct_bullish': pct_bullish,
        'recent_bullish_flips': recent_bullish_flips, 'recent_bearish_flips': recent_bearish_flips,
        'median_dist_atr': median_dist, 'pct_bullish_prior': pct_bullish_prior,
        'breadth_trend': breadth_trend, 'dispersion': dispersion, 'avg_corr': avg_corr,
    }, None


def build_report(ticker, timeframe, verbose=False):
    """Computes everything analyze() prints, returned as a plain dict
    instead - used by the CLI (analyze) and by the backend API route.
    Returns (report_dict, None) on success, (None, error_message) if
    there isn't enough data/history to give a reliable read."""
    if verbose:
        print(f"Fetching {ticker} and computing broader market regime...")
    daily = fetch_daily(ticker)
    if daily.empty or len(daily) < 200:
        return None, f"Not enough data for {ticker}."

    bars = resample_ohlc(daily, timeframe)
    if len(bars) < 100:
        return None, f"Not enough {timeframe} bars for {ticker} (have {len(bars)}, need >=100)."

    regimes, levels, atr, flips = run_state_machine(bars)
    close = bars['close'].values
    n = len(bars)
    idx = n - 1
    warmup = max(ATR_LEN, SLOW_LEN, PIVOT_LEN * 2) + 5

    if idx < warmup + HORIZON + 10:
        return None, f"Not enough history past warmup for a reliable read on {ticker} ({timeframe})."

    current_regime = int(regimes[idx])
    last_flip = flips[-1] if flips else None
    bars_since_flip = idx - last_flip[0] if last_flip else None
    flips_so_far = sum(1 for f in flips if f[0] <= idx)

    recent_flips_in_window = [f for f in flips if idx - CHOPPY_WINDOW <= f[0] <= idx]
    is_choppy = len(recent_flips_in_window) >= CHOPPY_FLIP_COUNT

    strip = regime_strip_agreement(close[:idx + 1], regimes[:idx + 1], [HORIZON], warmup)
    quality_z = float(strip.iloc[0]['z'])
    sufficient_history = flips_so_far >= MIN_FLIPS_FOR_QUALITY and quality_z > 0

    market_stress_prob = market_regime_context()
    market_stressed = (not np.isnan(market_stress_prob)) and market_stress_prob > HMM_STRESS_THRESHOLD

    if verbose:
        print(f"Fetching {CONFIRMATION_TIMEFRAME[timeframe]} confirmation timeframe...")
    conf_regime, conf_label = confirmation_regime(ticker, timeframe, daily)

    fund = fundamentals_rating(ticker, bars['datetime'].iloc[idx].strftime('%Y-%m-%d'))
    fund_note = fundamentals_narrative(fund)
    macro_note = macro_narrative(market_stress_prob, market_stressed)

    # --- classification ---
    # Leads with fundamentals + macro context (what an analyst would
    # cite), with the underlying structural read as the concluding
    # verdict rather than the headline mechanism.
    if not sufficient_history:
        classification = "NOT CONFIDENT"
        reason = (f"{ticker} doesn't have a reliable enough track record here to call a structural direction "
                  f"({flips_so_far} historical regime changes, low agreement with forward direction) - "
                  f"for context, fundamentals show {fund_note}, against {macro_note}")
    elif is_choppy:
        classification = "CONSOLIDATION"
        reason = (f"price structure is choppy, not trending ({len(recent_flips_in_window)} direction changes "
                  f"in the last {CHOPPY_WINDOW} bars) - fundamentals show {fund_note}, against {macro_note}")
    elif current_regime == 1 and not market_stressed:
        classification = "BULLISH"
        reason = f"fundamentals show {fund_note}, and against {macro_note}, the stock's structure is confirmed bullish"
    elif current_regime == -1 and market_stressed:
        classification = "BEARISH"
        reason = f"fundamentals show {fund_note}, and against {macro_note}, the stock's structure is confirmed bearish"
    else:
        classification = "NOT CONFIDENT"
        reason = (f"structure and market backdrop disagree - price structure reads "
                  f"{'bullish' if current_regime == 1 else 'bearish'} but the macro backdrop is "
                  f"{'stressed' if market_stressed else 'calm'} - fundamentals show {fund_note}")

    # --- move-size read (Yang-Zhang, backward-looking realized vol) ---
    daily_yz = yang_zhang_vol(daily)
    valid_yz = daily_yz[~np.isnan(daily_yz)]
    current_yz = valid_yz[-1] if len(valid_yz) else np.nan
    lookback = valid_yz[-YZ_LOOKBACK_FOR_PERCENTILE:] if len(valid_yz) >= YZ_LOOKBACK_FOR_PERCENTILE else valid_yz
    if len(lookback) >= 30 and not np.isnan(current_yz):
        pct = float((lookback < current_yz).mean() * 100)
        if pct < 33:
            move_size = "QUIET"
        elif pct < 67:
            move_size = "NORMAL"
        else:
            move_size = "ELEVATED"
    else:
        pct = np.nan
        move_size = "UNKNOWN (insufficient history)"

    maturity = regime_maturity(flips, current_regime, bars_since_flip) if bars_since_flip is not None else None
    candle_stats = candle_shape_stats(bars, regimes, current_regime, idx)
    move = expected_move(bars.iloc[:idx + 1])

    if verbose:
        print(f"\n  Computing sector peer breadth...")
    breadth, breadth_err = peer_breadth(ticker, timeframe)

    agrees_with_peers = None
    if breadth is not None:
        agrees_with_peers = (current_regime == 1 and breadth['pct_bullish'] >= 50) or \
                             (current_regime == -1 and breadth['pct_bullish'] < 50)

    report = {
        'ticker': ticker, 'timeframe': timeframe,
        'classification': classification, 'reason': reason,
        'current_regime': 'bullish/support' if current_regime == 1 else 'bearish/resistance',
        'bars_since_flip': bars_since_flip, 'flips_so_far': flips_so_far, 'quality_z': quality_z,
        'maturity': maturity,
        'market_stress_prob': None if np.isnan(market_stress_prob) else market_stress_prob,
        'market_stressed': market_stressed,
        'confirmation_timeframe': conf_label,
        'confirmation_regime': None if conf_regime is None else ('bullish/support' if conf_regime == 1 else 'bearish/resistance'),
        'confirmation_agrees': None if conf_regime is None else (conf_regime == current_regime),
        'sector_breadth': breadth, 'sector_breadth_error': breadth_err,
        'sector_breadth_agrees': agrees_with_peers,
        'fundamentals': fund,
        'candle_shape': candle_stats,
        'expected_move': move,
        'move_size': move_size,
        'current_yz_vol': None if np.isnan(current_yz) else float(current_yz),
        'yz_percentile': None if np.isnan(pct) else float(pct),
        'as_of_date': bars['datetime'].iloc[idx].strftime('%Y-%m-%d'),
    }
    return report, None


def print_report(report):
    ticker, timeframe = report['ticker'], report['timeframe']
    print(f"\n{'='*90}\n{ticker}  ({timeframe})\n{'='*90}")
    print(f"  Classification: {report['classification']}")
    print(f"  Reason: {report['reason']}")
    print(f"  Valiant regime: {report['current_regime']}  "
          f"({report['bars_since_flip']} bars since last flip, {report['flips_so_far']} total flips, "
          f"quality z={report['quality_z']:+.2f})")

    maturity = report['maturity']
    if maturity is not None:
        stage = ("FRESH" if maturity['percentile'] < 33 else
                 "TYPICAL" if maturity['percentile'] < 67 else
                 "EXTENDED" if maturity['percentile'] < 90 else "VERY EXTENDED")
        bull = report['current_regime'].startswith('bullish')
        print(f"  Regime maturity: {stage}  (current move is {report['bars_since_flip']} bars old; this stock's "
              f"{'bullish' if bull else 'bearish'} moves historically run "
              f"{maturity['mean']:.0f} bars on average, {maturity['median']:.0f} median, n={maturity['n']} - "
              f"current move is longer than {maturity['percentile']:.0f}% of past ones)")
    else:
        print(f"  Regime maturity: unavailable (not enough completed episodes yet)")

    msp = report['market_stress_prob']
    print(f"  Market regime (SPY-based HMM): {'STRESSED' if report['market_stressed'] else 'calm'}  "
          f"(P(stress)={msp:.2f})" if msp is not None else "  Market regime: unavailable")
    if report['confirmation_regime'] is not None:
        print(f"  Confirmation timeframe ({report['confirmation_timeframe']}): {report['confirmation_regime']}  "
              f"({'AGREES' if report['confirmation_agrees'] else 'DISAGREES'} with the {timeframe} valiant)")
    else:
        print(f"  Confirmation timeframe: unavailable ({report['confirmation_timeframe']})")

    breadth = report['sector_breadth']
    if breadth is not None:
        label = "peer" if breadth['grouping'] == 'sub-industry' else "broad-sector"
        print(f"  {label.capitalize()} breadth ('{breadth['sector']}', {breadth['grouping']}, "
              f"n={breadth['n_peers']}): "
              f"{breadth['pct_bullish']:.0f}% bullish  [{breadth['dispersion']}]"
              + (f", {breadth['breadth_trend'].lower()}" if breadth['breadth_trend'] else ""))
        print(f"    {breadth['recent_bullish_flips']} peers flipped bullish and "
              f"{breadth['recent_bearish_flips']} flipped bearish in the last {RECENT_FLIP_WINDOW} bars")
        if not np.isnan(breadth['median_dist_atr']):
            print(f"    median peer distance from its own valiant level: {breadth['median_dist_atr']:.2f} ATR")
        if breadth['avg_corr'] is not None:
            print(f"    avg pairwise peer correlation (trailing {PEER_CORR_WINDOW}d returns): {breadth['avg_corr']:+.2f} "
                  f"({'high - correlated/risk-driven regime' if breadth['avg_corr'] > 0.5 else 'moderate' if breadth['avg_corr'] > 0.25 else 'low - idiosyncratic, stock-specific drivers'})")
        print(f"    this stock's own valiant {'AGREES' if report['sector_breadth_agrees'] else 'DISAGREES'} with majority sector breadth")
    else:
        print(f"  Sector/peer breadth: unavailable ({report['sector_breadth_error']})")

    fund = report['fundamentals']
    if fund is not None:
        print(f"  Fundamentals (SEC EDGAR, as of {report['as_of_date']}): "
              f"{fund['n_passed']}/{fund['n_total']} checks passed  "
              f"({'PASS' if fund['strict_pass'] else 'fail'} strict gate)")
    else:
        print(f"  Fundamentals: unavailable (no cached SEC EDGAR data for {ticker})")

    candle_stats = report['candle_shape']
    if candle_stats is not None:
        print(f"\n  Typical candle shape when in this regime (n={candle_stats['n']} bars, ATR-normalized):")
        print(f"    GREEN (upside character):  open->high {candle_stats['open_to_high']:+.2f}   "
              f"low->close {candle_stats['low_to_close']:+.2f}")
        print(f"    RED (downside character):  high->close {-candle_stats['high_to_close']:+.2f}   "
              f"open->low {-candle_stats['open_to_low']:+.2f}")
    else:
        print(f"\n  Typical candle shape: unavailable (not enough history in this regime)")

    print(f"\n  Move-size (Yang-Zhang realized vol, annualized): {report['move_size']}")
    if report['current_yz_vol'] is not None:
        print(f"    current YZ vol: {report['current_yz_vol']*100:.1f}%   percentile vs own history: {report['yz_percentile']:.0f}th")

    move = report['expected_move']
    if move is not None:
        bar_noun = {'daily': 'day', 'weekly': 'week', 'monthly': 'month'}.get(timeframe, timeframe)
        print(f"\n  Expected move (GJR-GARCH skew-t, next {bar_noun}'s bar, from ${move['current_price']:.2f}):")
        print(f"    30-70% range:  ${move['low_30']:.2f}  -  ${move['high_70']:.2f}")
        print(f"    10-90% range:  ${move['low_10']:.2f}  -  ${move['high_90']:.2f}   (median ${move['median']:.2f})")
        if move['vs_typical_pct'] is not None:
            tag = ('WIDER than typical' if move['vs_typical_pct'] > 15 else
                   'NARROWER than typical' if move['vs_typical_pct'] < -15 else 'close to typical')
            print(f"    range width vs this bar-size's own trailing typical range: {move['vs_typical_pct']:+.0f}% ({tag})")
        print(f"    same methodology validated on NQ/ES in the macro-regime forecast - not separately validated on individual equities.")
    else:
        print(f"\n  Expected move: unavailable (not enough history for a stable GARCH fit)")

    print(f"\n  This is a state description, not a prediction. 'Bullish'/'Bearish' means the valiant and")
    print(f"  market regime AGREE, not that a positive/negative return is expected with any stated odds.")


def analyze(ticker, timeframe):
    report, err = build_report(ticker, timeframe, verbose=True)
    if report is None:
        print(err)
        return
    print_report(report)


if __name__ == '__main__':
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    timeframe = sys.argv[2] if len(sys.argv) > 2 else 'daily'
    analyze(ticker.upper(), timeframe)
