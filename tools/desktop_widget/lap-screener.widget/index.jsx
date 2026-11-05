// L'Argent Parle - the live vector screener, as an engraved desk statement. Übersicht widget (https://tracesof.net/uebersicht/), refreshes every minute
// from the site's snapshot. Type: Times New Roman (roman for figures, italic for prose) + Snell Roundhand for the masthead (both ship with macOS).
import { css } from "uebersicht";

export const command = "bash lap-screener.widget/fetch.sh";
export const refreshFrequency = 60 * 1000;

const C = { ink: "rgba(13,19,34,0.94)", vellum: "#ece2c9", vellum2: "#c9bfa6", pewter: "#8b8f99", gold: "#c6a465", rule: "rgba(198,164,101,0.28)", faint: "rgba(236,226,201,0.08)",
            gain: "#9cc3a0", loss: "#d4776f", amber: "#e0b25a" };
const SERIF = `"Times New Roman", Times, serif`, SCRIPT = `"Snell Roundhand", "Apple Chancery", cursive`;
export const className = `
  left: 24px; bottom: 24px; width: 392px; color: ${C.vellum}; font-family: ${SERIF}; font-size: 13px; line-height: 1.35;
  background: ${C.ink}; border: 1px solid ${C.rule}; border-radius: 6px; padding: 0 20px 18px; overflow: hidden;
  box-shadow: 0 18px 50px rgba(0,0,0,0.45); -webkit-font-smoothing: antialiased;
`;
const row = css`display: flex; align-items: baseline; padding: 5px 0; border-top: 1px solid ${C.faint};`;
const title = css`display: flex; justify-content: space-between; align-items: baseline; margin: 16px 0 4px; font-style: italic; color: ${C.gold}; font-size: 15px;`;
const it = css`font-style: italic;`;

const MON = { "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun", "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec" };
const DAY = { Mon: "Monday", Tue: "Tuesday", Wed: "Wednesday", Thu: "Thursday", Fri: "Friday", Sat: "Saturday", Sun: "Sunday" };
const MONTH = { Jan: "January", Feb: "February", Mar: "March", Apr: "April", May: "May", Jun: "June", Jul: "July", Aug: "August", Sep: "September", Oct: "October", Nov: "November", Dec: "December" };
const sgn = (x, d = 2) => (x == null || isNaN(x) ? "–" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(Number(x)).toFixed(d)}%`);
const hue = (x) => (x == null || isNaN(x) ? C.pewter : x >= 0 ? C.gain : C.loss);
const shortDay = (s) => { const m = /(\d\d)-(\d\d)/.exec(s || ""); return m ? `${Number(m[2])} ${MON[m[1]]}` : ""; };          // 'Wed 10-21 11:00  343.18' -> '21 Oct'
const longSince = (s) => { const p = (s || "").split(" "); return p.length === 3 ? `${DAY[p[0]] || p[0]} ${Number(p[2])} ${MONTH[p[1]] || p[1]}` : s || "the start"; };
const ago = (iso) => { if (!iso) return ""; const m = Math.round((Date.now() - new Date(iso).getTime()) / 60000); return m < 1 ? "just now" : m < 60 ? `${m} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} days ago`; };
const SLOTS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight"];

const Guilloche = ({ w, h }) => {                 // banknote-style engraved wave band behind the masthead
  const lines = [];
  for (let k = 0; k < 16; k++) {
    const ph = (k / 16) * Math.PI * 2, pts = [];
    for (let x = 0; x <= w; x += 3) { const t = (x / w) * Math.PI * 5; pts.push(`${x},${(h / 2 + Math.sin(t + ph) * h * 0.3 + Math.sin(t * 2.6 - ph * 1.4) * h * 0.13).toFixed(1)}`); }
    lines.push(<polyline key={k} points={pts.join(" ")} fill="none" stroke={C.gold} strokeOpacity="0.13" strokeWidth="0.6" />);
  }
  return <svg width={w} height={h} style={{ position: "absolute", left: 0, top: 0 }}>{lines}</svg>;
};
const Masthead = ({ right }) => (
  <div style={{ position: "relative", height: 62, margin: "0 -20px", padding: "0 20px", borderBottom: `1px solid ${C.rule}` }}>
    <Guilloche w={432} h={62} />
    <div style={{ position: "relative", display: "flex", justifyContent: "space-between", alignItems: "flex-end", height: 54 }}>
      <span style={{ fontFamily: SCRIPT, fontSize: 30, color: C.gold, lineHeight: 1 }}>L'Argent Parle</span>
      <span className={it} style={{ color: C.vellum2, fontSize: 12, textAlign: "right" }}>{right}</span>
    </div>
  </div>
);
const Spark = ({ curve }) => {
  if (!curve || !curve.t || curve.t.length < 2) return null;
  const W = 392, H = 46, n = curve.t.length, vals = [...(curve.book || []), ...(curve.qqq || [])].filter((v) => v != null);
  if (!vals.length) return null;
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), span = hi - lo || 1, X = (i) => (i / (n - 1)) * W, Y = (v) => 3 + (H - 6) * (1 - (v - lo) / span);
  const pts = (a) => (a || []).map((v, i) => (v == null ? null : `${X(i).toFixed(1)},${Y(v).toFixed(1)}`)).filter(Boolean).join(" ");
  const b = curve.book || [], li = b.length - 1;
  return (
    <svg width={W} height={H} style={{ display: "block", margin: "8px 0 4px" }}>
      <line x1="0" x2={W} y1={Y(0)} y2={Y(0)} stroke={C.pewter} strokeOpacity="0.35" strokeDasharray="1 3" />
      <polyline points={pts(curve.qqq)} fill="none" stroke={C.pewter} strokeOpacity="0.7" strokeWidth="1" strokeDasharray="2 2" />
      <polyline points={pts(b)} fill="none" stroke={C.vellum} strokeWidth="1.4" />
      {b[li] != null && <circle cx={X(li)} cy={Y(b[li])} r="2.6" fill={C.gold} />}
    </svg>
  );
};

export const render = ({ output, error }) => {
  let d = null; try { d = JSON.parse(output); } catch (e) {}
  const today = new Date().toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long" });
  if (error || !d || d.error || !d.snapshot)
    return (<div><Masthead right={today} /><div className={it} style={{ color: C.vellum2, marginTop: 14 }}>{(d && d.error) || "Waiting for the site. Render can take a minute to wake up."}</div></div>);
  const s = d.snapshot, h = s.history || {}, pos = s.positions || [], sig = s.signals || [], watch = (s.watch || []).slice(0, 8);
  const trades = h.trades || [], since = {}; trades.filter((t) => t.open).forEach((t) => { since[t.name] = shortDay(t.entry); });
  const closed = trades.filter((t) => !t.open).slice(0, 3);
  const age = s.asof ? (Date.now() - new Date(s.asof).getTime()) / 60000 : null, stale = s.market_open && age > 30;
  return (
    <div>
      <Masthead right={<span>{today}<br /><span style={{ color: stale ? C.amber : C.pewter }}>{s.market_open ? "Market open" : "Market closed"}, updated {ago(s.asof)}</span></span>} />
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginTop: 14 }}>
        <div>
          <div style={{ fontSize: 42, lineHeight: 1, color: hue(h.book), letterSpacing: "-0.01em" }}>{sgn(h.book)}</div>
          <div className={it} style={{ color: C.vellum2, marginTop: 4 }}>since {longSince(h.since)}</div>
        </div>
        <div style={{ textAlign: "right", color: C.vellum2, lineHeight: 1.5 }}>
          {h.qqq != null && <div><span className={it} style={{ color: C.pewter }}>QQQ </span>{sgn(h.qqq)}</div>}
          {h.spy != null && <div><span className={it} style={{ color: C.pewter }}>SPY </span>{sgn(h.spy)}</div>}
        </div>
      </div>
      <Spark curve={h.curve} />
      <div className={it} style={{ color: C.pewter, fontSize: 12 }}>
        Realized {sgn(h.realized)}, open {sgn(h.open)}. {SLOTS[pos.length] ? SLOTS[pos.length][0].toUpperCase() + SLOTS[pos.length].slice(1) : pos.length} of {SLOTS[s.slots] || s.slots} slots filled.
      </div>

      <div className={title}><span>Positions</span><span style={{ color: C.pewter, fontSize: 11 }}>gain, peak, room to stop</span></div>
      {pos.length === 0 && <div className={it} style={{ color: C.pewter }}>No open positions. The book is in cash.</div>}
      {pos.map((p) => {
        const last = Number(p.last), stop = Number(p.stop), room = last && stop ? ((last - stop) / last) * 100 : null;
        return (
          <div className={row} key={p.name}>
            <span style={{ flex: 1 }}>
              <span style={{ fontSize: 14, letterSpacing: "0.04em", color: p.selling ? C.loss : C.vellum }}>{p.name}</span>
              <span className={it} style={{ color: C.pewter, fontSize: 11.5 }}> {p.selling ? "selling at the next open" : since[p.name] ? `since ${since[p.name]}` : ""}</span>
            </span>
            <span style={{ width: 62, textAlign: "right", color: hue(p.pnl), fontSize: 14 }}>{sgn(p.pnl)}</span>
            <span style={{ width: 58, textAlign: "right", color: C.vellum2 }}>{sgn(p.peak)}</span>
            <span style={{ width: 52, textAlign: "right", color: room != null && room < 1 ? C.amber : C.vellum2 }}>{room == null ? "–" : `${room.toFixed(1)}%`}</span>
          </div>
        );
      })}

      {sig.length > 0 && <div className={title}><span>Today</span></div>}
      {sig.slice(0, 5).map((x, i) => (
        <div className={row} key={i}>
          <span style={{ width: 44, color: C.pewter }}>{x.time}</span>
          <span className={it} style={{ flex: 1, color: C.vellum2 }}>
            {x.action === "SELL" ? "Sold " : "Bought "}<span style={{ fontStyle: "normal", letterSpacing: "0.04em", color: C.vellum }}>{x.ticker}</span> at {x.price}{x.grade ? `, grade ${x.grade}` : ""}
          </span>
          {x.pnl != null && <span style={{ color: hue(x.pnl) }}>{sgn(x.pnl)}</span>}
        </div>
      ))}

      {closed.length > 0 && <div className={title}><span>Recently closed</span></div>}
      {closed.map((t, i) => (
        <div className={row} key={i}>
          <span style={{ width: 64, letterSpacing: "0.04em" }}>{t.name}</span>
          <span className={it} style={{ flex: 1, color: C.pewter, fontSize: 12 }}>{t.lock ? "profit lock, " : ""}{shortDay(t.exit)}</span>
          <span style={{ color: hue(t.ret) }}>{sgn(t.ret)}</span>
        </div>
      ))}

      <div className={title}><span>Ranked by options ratio</span><span style={{ color: C.pewter, fontSize: 11 }}>underlined: held</span></div>
      <div style={{ display: "flex", flexWrap: "wrap", columnGap: 16, rowGap: 4, marginTop: 2 }}>
        {watch.map((w) => (
          <span key={w.tk} style={{ whiteSpace: "nowrap" }}>
            <span style={{ letterSpacing: "0.04em", color: w.held ? C.vellum : C.vellum2, borderBottom: w.held ? `1px solid ${C.gold}` : "none" }}>{w.tk}</span>
            <span className={it} style={{ color: C.pewter, fontSize: 12 }}> {Number(w.ratio).toFixed(1)}</span>
          </span>
        ))}
      </div>
    </div>
  );
};
