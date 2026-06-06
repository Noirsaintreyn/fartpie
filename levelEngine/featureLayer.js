/**
 * featureLayer.js - Derive options context from Alpaca chain
 */

export function parseContracts(rawChain, scaler, currentPrice) {
  return Object.values(rawChain)
    .map(s => {
      const strike = parseFloat(s.details?.strike_price ?? 0) * scaler;
      const type = s.details?.type ?? '';
      const gamma = s.greeks?.gamma ?? 0;
      const delta = s.greeks?.delta ?? 0;
      const theta = s.greeks?.theta ?? 0;
      const vega = s.greeks?.vega ?? 0;
      const iv = s.implied_volatility ?? 0;
      const mid = ((s.latest_quote?.ap ?? 0) + (s.latest_quote?.bp ?? 0)) / 2;
      const vannaApprox = (iv > 0 && currentPrice > 0)
        ? (vega * delta) / (currentPrice * iv)
        : 0;
      const charmApprox = -(theta * delta);
      return { strike, type, gamma, delta, theta, vega, iv, mid, vannaApprox, charmApprox };
    })
    .filter(c => c.strike > 0 && (c.type === 'call' || c.type === 'put'));
}

export function calcGammaProfile(contracts) {
  const byStrike = new Map();
  for (const c of contracts) {
    const k = c.strike.toFixed(2);
    const dealerGamma = c.type === 'call' ? -c.gamma : c.gamma;
    byStrike.set(k, (byStrike.get(k) ?? 0) + dealerGamma);
  }
  const strikeLevels = [...byStrike.entries()]
    .map(([k, gamma]) => ({ strike: parseFloat(k), gamma }))
    .sort((a, b) => a.strike - b.strike);
  const netGamma = strikeLevels.reduce((sum, l) => sum + l.gamma, 0);
  let gammaFlip = null;
  let runningGamma = 0;
  for (let i = 0; i < strikeLevels.length - 1; i++) {
    const prev = runningGamma;
    runningGamma += strikeLevels[i].gamma;
    if (prev !== 0 && (prev < 0) !== (runningGamma < 0)) {
      const s0 = strikeLevels[i].strike;
      const s1 = strikeLevels[i + 1].strike;
      const w = Math.abs(prev) / (Math.abs(prev) + Math.abs(runningGamma));
      gammaFlip = s0 + w * (s1 - s0);
      break;
    }
  }
  return { gammaFlip, netGamma, strikeLevels };
}

export function calcFlowPressure(contracts) {
  let vannaNet = 0;
  let charmNet = 0;
  for (const c of contracts) {
    vannaNet += c.type === 'call' ? c.vannaApprox : -c.vannaApprox;
    charmNet += c.type === 'call' ? c.charmApprox : -c.charmApprox;
  }
  return { vannaNet, charmNet };
}

export function calcIVContext(contracts, currentPrice, ivHistory) {
  const ATM_BAND = 0.015;
  const atmContracts = contracts.filter(c => Math.abs(c.strike - currentPrice) / currentPrice < ATM_BAND);
  const atmIV = atmContracts.length
    ? atmContracts.reduce((s, c) => s + c.iv, 0) / atmContracts.length
    : (contracts[0]?.iv ?? 0);
  const ivZScore = zScore(atmIV, ivHistory);
  return { atmIV, ivZScore };
}

export function buildContext(rawChain, ivHistory, currentPrice, scaler) {
  const contracts = parseContracts(rawChain, scaler, currentPrice);
  if (!contracts.length) {
    return {
      gammaFlip: null, netGamma: 0, vannaNet: 0, charmNet: 0,
      ivZScore: 0, atmIV: 0, scaler,
      contractCount: 0, dataQuality: 'empty',
    };
  }
  const { gammaFlip, netGamma } = calcGammaProfile(contracts);
  const { vannaNet, charmNet } = calcFlowPressure(contracts);
  const { atmIV, ivZScore } = calcIVContext(contracts, currentPrice, ivHistory);
  return {
    gammaFlip, netGamma, vannaNet, charmNet,
    ivZScore, atmIV, scaler,
    contractCount: contracts.length,
    dataQuality: contracts.length > 20 ? 'good' : 'sparse',
  };
}

export function zScore(value, arr) {
  if (!arr.length || value === 0) return 0;
  const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
  const std = Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length);
  return std > 0 ? (value - mean) / std : 0;
}
