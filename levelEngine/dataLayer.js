/**
 * dataLayer.js - Alpaca API communication for options data
 */

const DATA_BASE = 'https://data.alpaca.markets/v1beta1';
const IV_CACHE_TTL_MS = 15 * 60_000;
const _ivCache = {};

export async function fetchChain(ticker, authHeaders, expiryCount = 3) {
  const expiries = nextExpiries(expiryCount);
  const allContracts = {};
  await Promise.all(expiries.map(async (expiry) => {
    let pageToken = null;
    let page = 0;
    do {
      const params = new URLSearchParams({
        expiration_date: expiry,
        limit: '200',
        feed: 'indicative',
        ...(pageToken ? { page_token: pageToken } : {}),
      });
      let res, data;
      try {
        res = await fetch(`${DATA_BASE}/options/snapshots/${ticker}?${params}`, { headers: authHeaders });
        data = await res.json();
      } catch (err) {
        console.warn(`[dataLayer] Fetch failed ${ticker} ${expiry}:`, err.message);
        break;
      }
      if (!res.ok) {
        console.warn(`[dataLayer] HTTP ${res.status} for ${ticker} ${expiry}:`, data?.message ?? '');
        break;
      }
      Object.assign(allContracts, data.snapshots ?? {});
      pageToken = data.next_page_token ?? null;
      page++;
    } while (pageToken && page < 5);
  }));
  return allContracts;
}

export async function fetchIVHistory(ticker, authHeaders) {
  const cached = _ivCache[ticker];
  if (cached && Date.now() - cached.fetchedAt < IV_CACHE_TTL_MS) {
    return cached.values;
  }
  const today = new Date();
  const start = new Date(today - 30 * 86_400_000).toISOString().split('T')[0];
  const end = today.toISOString().split('T')[0];
  const params = new URLSearchParams({
    expiration_date_gte: start,
    expiration_date_lte: end,
    limit: '500',
    feed: 'indicative',
  });
  let values = [];
  try {
    const res = await fetch(`${DATA_BASE}/options/snapshots/${ticker}?${params}`, { headers: authHeaders });
    const data = await res.json();
    values = Object.values(data.snapshots ?? {})
      .map(s => s.implied_volatility)
      .filter(v => typeof v === 'number' && v > 0);
  } catch (err) {
    console.warn('[dataLayer] IV history fetch failed:', err.message);
  }
  _ivCache[ticker] = { values, fetchedAt: Date.now() };
  return values;
}

export function nextExpiries(n) {
  const expiries = [];
  const d = new Date();
  while (expiries.length < n) {
    d.setDate(d.getDate() + 1);
    if (d.getDay() === 5) expiries.push(d.toISOString().split('T')[0]);
  }
  return expiries;
}
