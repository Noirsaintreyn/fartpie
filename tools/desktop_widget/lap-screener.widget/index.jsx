// L'Argent Parle - the live vector screener, as an engraved desk statement. Übersicht widget (https://tracesof.net/uebersicht/), refreshes every minute
// from the site's snapshot. Type: Times New Roman (roman for figures, italic for prose) + Snell Roundhand for the masthead (both ship with macOS).
// MOVE IT: drag it anywhere (the spot is remembered). Double-click puts it back at the default spot.
// Data: the Mac screener's own ~/fartpie/live/site_snapshot.json (fetch.sh), so overnight it shows the last close and goes live at the open.
import { css } from "uebersicht";

export const command = "bash lap-screener.widget/fetch.sh";
export const refreshFrequency = 60 * 1000;
export const className = `left: 0; top: 0;`;            // the panel positions itself (draggable), see Panel below

const C = { ink: "rgba(13,19,34,0.94)", vellum: "#ece2c9", vellum2: "#c9bfa6", pewter: "#8b8f99", gold: "#c6a465", rule: "rgba(198,164,101,0.28)", faint: "rgba(236,226,201,0.08)",
            gain: "#9cc3a0", loss: "#d4776f", amber: "#e0b25a" };
const SERIF = `"Times New Roman", Times, serif`, SCRIPT = `"Snell Roundhand", "Apple Chancery", cursive`;
const box = css`
  position: absolute; width: 392px; color: ${C.vellum}; font-family: ${SERIF}; font-size: 13px; line-height: 1.35; cursor: grab; user-select: none;
  background: ${C.ink}; border: 1px solid ${C.rule}; border-radius: 6px; padding: 0 20px 18px; overflow: hidden;
  box-shadow: 0 18px 50px rgba(0,0,0,0.45); -webkit-font-smoothing: antialiased;
  &:active { cursor: grabbing; }
`;
const row = css`display: flex; align-items: baseline; padding: 5px 0; border-top: 1px solid ${C.faint};`;
const title = css`display: flex; justify-content: space-between; align-items: baseline; margin: 16px 0 4px; font-style: italic; color: ${C.gold}; font-size: 15px;`;
const it = css`font-style: italic;`;

// ---- dragging: the panel's spot lives in localStorage, so it survives refreshes and restarts
const KEY = "lap-screener-pos", DEF = { x: 190, y: 64 };
const getPos = () => { try { const p = JSON.parse(window.localStorage.getItem(KEY)); return p && isFinite(p.x) && isFinite(p.y) ? p : DEF; } catch (e) { return DEF; } };
const startDrag = (e) => {
  if (e.button !== 0) return;
  const el = e.currentTarget, p = getPos(), dx = e.clientX - p.x, dy = e.clientY - p.y;
  const at = (ev) => ({ x: Math.max(0, ev.clientX - dx), y: Math.max(0, ev.clientY - dy) });
  const move = (ev) => { const q = at(ev); el.style.left = `${q.x}px`; el.style.top = `${q.y}px`; };
  const up = (ev) => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); try { window.localStorage.setItem(KEY, JSON.stringify(at(ev))); } catch (err) {} };
  window.addEventListener("mousemove", move); window.addEventListener("mouseup", up); e.preventDefault();
};
const resetPos = (e) => { try { window.localStorage.removeItem(KEY); } catch (err) {} e.currentTarget.style.left = `${DEF.x}px`; e.currentTarget.style.top = `${DEF.y}px`; };
const Panel = ({ children }) => { const p = getPos(); return <div className={box} style={{ left: p.x, top: p.y }} onMouseDown={startDrag} onDoubleClick={resetPos}>{children}</div>; };

const MON = { "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun", "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec" };
const sgn = (x, d = 2) => (x == null || isNaN(x) ? "–" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(Number(x)).toFixed(d)}%`);
const hue = (x) => (x == null || isNaN(x) ? C.pewter : x >= 0 ? C.gain : C.loss);
const shortDay = (s) => { const m = /(\d\d)-(\d\d)/.exec(s || ""); return m ? `${Number(m[2])} ${MON[m[1]]}` : ""; };          // 'Wed 10-21 11:00  343.18' -> '21 Oct'
const ago = (iso) => { if (!iso) return ""; const m = Math.round((Date.now() - new Date(iso).getTime()) / 60000); return m < 1 ? "just now" : m < 60 ? `${m} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} days ago`; };
const SLOTS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight"];
const marketOpenNow = () => {                      // New York clock: weekdays 09:30-16:00 (holidays show as open but stale)
  const p = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false }).formatToParts(new Date());
  const g = (t) => (p.find((x) => x.type === t) || {}).value, m = (Number(g("hour")) % 24) * 60 + Number(g("minute"));
  return !["Sat", "Sun"].includes(g("weekday")) && m >= 570 && m < 960;
};

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
    return (<Panel><Masthead right={today} /><div className={it} style={{ color: C.vellum2, marginTop: 14 }}>{(d && d.error) || "Waiting for the site. Render can take a minute to wake up."}</div></Panel>);
  const s = d.snapshot, h = s.history || {}, pos = s.positions || [], sig = s.signals || [], watch = (s.watch || []).slice(0, 8);
  const trades = h.trades || [], since = {}; trades.filter((t) => t.open).forEach((t) => { since[t.name] = shortDay(t.entry); });
  const closed = trades.filter((t) => !t.open).slice(0, 3);
  const age = s.asof ? (Date.now() - new Date(s.asof).getTime()) / 60000 : null, open = marketOpenNow(), stale = open && age > 30;
  const asofDay = s.asof ? new Date(s.asof).toLocaleDateString("en-GB", { weekday: "long" }) : "", sameDay = s.asof && new Date(s.asof).toDateString() === new Date().toDateString();
  const status = !open ? `Market closed, showing ${sameDay ? "today" : asofDay}'s close` : stale ? `Screener not updating, last ${ago(s.asof)}` : `Market open, updated ${ago(s.asof)}`;
  return (
    <Panel>
      <Masthead right={<span>{today}<br /><span style={{ color: stale ? C.amber : C.pewter }}>{status}</span></span>} />
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginTop: 14 }}>
        <div style={{ fontSize: 42, lineHeight: 1, color: hue(h.book), letterSpacing: "-0.01em" }}>{sgn(h.book)}</div>
        <div style={{ textAlign: "right", color: C.vellum2, lineHeight: 1.5 }}>
          <div className={it} style={{ color: C.gold, fontSize: 14 }}>since launch</div>
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
    </Panel>
  );
};
