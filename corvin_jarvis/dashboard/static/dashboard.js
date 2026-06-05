const $ = (id) => document.getElementById(id);
const fmtKRW = (v) => v == null ? "—" : "₩" + Math.round(v).toLocaleString();
const pct = (v) => v == null ? "—" : (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
const cls = (v) => v == null ? "dim" : v >= 0 ? "up" : "down";
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));

function renderHero(t) {
  const pnl = t.equity_pnl_krw, p = t.equity_pnl_pct;
  $("p-hero").innerHTML = `
    <div><div class="big ${cls(pnl)}">${pnl != null && pnl >= 0 ? '+' : ''}${fmtKRW(pnl)}</div>
      <div class="sub ${cls(p)}">P&L ${pct(p)}</div></div>
    <div style="margin-left:auto;text-align:right">
      <div class="dim" style="font-size:11px">TOTAL ASSETS</div>
      <div class="tot">${fmtKRW(t.total_assets_krw)}</div>
      <div class="dim">CASH ${fmtKRW(t.deployable_krw)}</div></div>`;
}

// 실시간 수익률(%) 추이 — 매 폴링마다 equity_pnl_pct를 누적해 0% 기준선 위로 라이브 드로잉.
let RETURN_SERIES = [];
function renderReturnCurve(series, nowPct) {
  const el = $("p-curve");
  const pts = series.filter(v => v != null);
  const nowLbl = nowPct == null ? "—" : (nowPct >= 0 ? "+" : "") + nowPct.toFixed(2) + "%";
  const nowCls = nowPct == null ? "dim" : nowPct >= 0 ? "up" : "down";
  if (pts.length < 1) {
    el.innerHTML = `<div class="curve-meta"><span class="dim">LIVE · intraday return</span>
      <span class="now ${nowCls}">${nowLbl}</span></div>
      <div class="dim" style="flex:1;display:flex;align-items:center">accumulating live…</div>`;
    return;
  }
  const lo = Math.min(0, ...pts), hi = Math.max(0, ...pts), span = (hi - lo) || 1;
  const W = 600, H = 110;
  const y = (v) => H - ((v - lo) / span) * (H - 12) - 6;
  const x = (i) => pts.length < 2 ? W : (i / (pts.length - 1)) * W;
  const line = pts.map((v, i) => `${x(i).toFixed(0)},${y(v).toFixed(1)}`).join(" ");
  const y0 = y(0).toFixed(1);
  const ex = x(pts.length - 1).toFixed(0), ey = y(pts[pts.length - 1]).toFixed(1);
  el.innerHTML = `
    <div class="curve-meta"><span class="dim">LIVE · intraday return · ${pts.length}pt</span>
      <span class="now ${nowCls}">${nowLbl}</span></div>
    <svg class="curve-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <line x1="0" y1="${y0}" x2="${W}" y2="${y0}" stroke="#5a4520" stroke-width="1" stroke-dasharray="4 4"/>
      <polyline fill="none" stroke="#ffae42" stroke-width="2" points="${line}"/>
      <circle class="curve-tip" cx="${ex}" cy="${ey}" r="3.5" fill="#ffd27f"/>
    </svg>
    <div class="curve-axis"><span>basis 0%</span><span>${hi >= 0 ? '+' : ''}${hi.toFixed(2)}% peak</span></div>`;
}

function renderPositions(rows) {
  const price = (r) => r.price == null ? "—" : (r.ccy === "USD" ? "$" + r.price.toFixed(2) : "₩" + Math.round(r.price).toLocaleString());
  $("p-positions").innerHTML = `<table>
    <tr><th>TICKER</th><th>PRICE</th><th>DAY</th><th>PNL%</th><th>VALUE(₩)</th></tr>
    ${rows.map(r => `<tr><td><span class="sym">${esc(r.sym)}</span></td>
      <td>${price(r)}</td><td class="${cls(r.day_pct)}">${pct(r.day_pct)}</td>
      <td class="${cls(r.pnl_pct)}">${pct(r.pnl_pct)}</td>
      <td>${r.value_krw?Math.round(r.value_krw/1000).toLocaleString()+'k':'—'}</td></tr>`).join("")}
  </table>`;
}

function renderIndices(rows) {
  $("p-indices").innerHTML = rows.map(r =>
    `<div class="idx"><span>${esc(r.label)}</span><span>${r.price?.toLocaleString()} <span class="${cls(r.pct)}">${pct(r.pct)}</span></span></div>`).join("");
}

function renderAlloc(rows) {
  $("p-alloc").innerHTML = rows.map(r =>
    `<div class="bar"><div class="lab"><span>${esc(r.label)}</span><span>${r.pct}%</span></div>
     <div class="track"><div class="fill" style="width:${Math.min(r.pct,100)}%"></div></div></div>`).join("");
}

function renderSignals(rows) {
  const dot = (c) => c === "green" ? "g" : c === "red" ? "r" : "a";
  $("p-signals").innerHTML = rows.map(r =>
    `<div><span><span class="dot ${dot(r.color)}"></span>${esc(r.sym)}</span><span class="dim">${esc(r.zone)}</span></div>`).join("")
    || '<div class="dim">— no signals</div>';
}

function renderLog(rows) {
  $("p-log").innerHTML = rows.map(r => `<div><b>${esc(r.ts)}</b> ${esc(r.text)}</div>`).join("");
}

let POLY_ACTIVE = null;
function renderPoly(rows) {
  const tabsEl = $("p-poly-tabs");
  if (!rows || !rows.length) { tabsEl.innerHTML = ""; $("p-poly").innerHTML = '<div class="dim">— no data</div>'; return; }
  const cats = {};
  rows.forEach(r => { const c = r.category || "Other"; (cats[c] = cats[c] || []).push(r); });
  const names = Object.keys(cats);
  if (POLY_ACTIVE === null || !cats[POLY_ACTIVE]) POLY_ACTIVE = names[0];
  const vol = (v) => v >= 1e6 ? (v/1e6).toFixed(1)+"M" : Math.round(v/1e3)+"k";
  const body = () => {
    $("p-poly").innerHTML = (cats[POLY_ACTIVE] || []).map(r => `
      <div><span class="q">${esc(r.question)}</span> ${Math.round(r.prob*100)}%
        <div class="track"><div class="fill" style="width:${r.prob*100}%"></div></div>
        <span class="dim" style="font-size:10px">interest $${vol(r.volume_usd)}</span></div>`).join("");
  };
  tabsEl.innerHTML = names.map((n, i) =>
    `<span class="poly-tab ${n===POLY_ACTIVE?'active':''}" data-idx="${i}">${esc(n)}</span>`).join("");
  tabsEl.querySelectorAll(".poly-tab").forEach((t, i) => {
    t.onclick = () => { POLY_ACTIVE = names[i]; body();
      tabsEl.querySelectorAll(".poly-tab").forEach((x, j) => x.classList.toggle("active", j === i)); };
  });
  body();
}

function renderTicker(rows) {
  $("p-ticker").innerHTML = rows.map(r =>
    `<span><b>${esc(r.label)}</b> ${r.price} <span class="${cls(r.pct)}">${pct(r.pct)}</span></span>`).join("");
}

// 실시간 틱 테이프 — 보유·지수의 최신 시세를 회전시켜 시퀀스 애니메이션으로 흘림.
let TAPE_DATA = [];
let TAPE_TIMER = null;
function renderTape(snap) {
  const items = [];
  (snap.positions || []).forEach(p => items.push({ s: p.sym, v: p.price, c: p.day_pct, ccy: p.ccy }));
  (snap.indices || []).forEach(i => items.push({ s: i.label, v: i.price, c: i.pct }));
  TAPE_DATA = items;
}
function startTape() {
  if (TAPE_TIMER) return;
  const el = $("p-matrix");
  let off = 0;
  const VIS = 5;
  const fmt = (d) => d.v == null ? "—" : d.ccy === "USD" ? "$" + d.v.toFixed(2)
    : d.v >= 1000 ? Math.round(d.v).toLocaleString() : String(d.v);
  TAPE_TIMER = setInterval(() => {
    if (!TAPE_DATA.length) return;
    off = (off + 1) % TAPE_DATA.length;
    const rows = [];
    for (let i = 0; i < Math.min(VIS, TAPE_DATA.length); i++) {
      const d = TAPE_DATA[(off + i) % TAPE_DATA.length];
      const cls = d.c == null ? "dim" : d.c >= 0 ? "up" : "down";
      const arr = d.c == null ? "·" : d.c >= 0 ? "▲" : "▼";
      const chg = d.c == null ? "" : Math.abs(d.c).toFixed(1);
      rows.push(`<div class="tape-row"><span class="sym">${esc(d.s)}</span><span>${fmt(d)}</span><span class="${cls}">${arr}${chg}</span></div>`);
    }
    el.innerHTML = rows.join("");
  }, 900);
}

async function refresh() {
  try {
    const snap = await (await fetch("/api/snapshot")).json();
    const state = snap.market_state || "open";
    renderHero(snap.totals || {});
    const rp = snap.totals ? snap.totals.equity_pnl_pct : null;
    if (rp != null) { RETURN_SERIES.push(rp); if (RETURN_SERIES.length > 120) RETURN_SERIES.shift(); }
    renderReturnCurve(RETURN_SERIES, rp);
    renderTape(snap);
    renderPositions(snap.positions || []);
    renderIndices(snap.indices || []);
    renderAlloc(snap.allocation || []);
    renderSignals(snap.signals || []);
    renderLog(snap.log || []);
    renderTicker(snap.macro_ticker || []);
    renderPoly(snap.polymarket || []);
    $("topmeta").textContent = `${snap.ts} · ${state.toUpperCase()} · FX ${snap.fx_usdkrw?.toFixed?.(2) ?? "—"}`;
    return state;
  } catch (e) {
    $("topmeta").textContent = "connection error — retrying";
    return "open";
  }
}

async function loop() {
  const state = await refresh();
  setTimeout(loop, state === "open" ? 30000 : 300000);
}

// 시작 버스트: 캐시된 스냅샷을 빠르게 몇 번 샘플해 수익률 곡선을 즉시 띄운 뒤 정상 주기로.
async function bootstrap() {
  for (let i = 0; i < 6; i++) {
    await refresh();
    await new Promise(r => setTimeout(r, 1200));
  }
  loop();
}

startTape();
bootstrap();
