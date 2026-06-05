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

function renderCurve(series) {
  if (!series || series.length < 2) { $("p-curve").innerHTML = '<div class="dim">— no curve data</div>'; return; }
  const vals = series.map(p => p.value), min = Math.min(...vals), max = Math.max(...vals);
  const W = 600, H = 150, span = (max - min) || 1;
  const pts = series.map((p, i) => {
    const x = (i / (series.length - 1)) * W;
    const y = H - ((p.value - min) / span) * (H - 10) - 5;
    return `${x.toFixed(0)},${y.toFixed(0)}`;
  }).join(" ");
  $("p-curve").innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
       <polygon fill="rgba(255,174,66,.10)" points="${pts} ${W},${H} 0,${H}"/>
       <polyline fill="none" stroke="#ffae42" stroke-width="2" points="${pts}"/></svg>`;
}

function renderPositions(rows) {
  const price = (r) => r.price == null ? "—" : (r.ccy === "USD" ? "$" + r.price.toFixed(2) : "₩" + Math.round(r.price).toLocaleString());
  $("p-positions").innerHTML = `<table>
    <tr><th>종목</th><th>현재가</th><th>일간</th><th>손익%</th><th>평가(₩)</th></tr>
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

let MATRIX_TIMER = null;
function startMatrix() {
  if (MATRIX_TIMER) return;
  const el = $("p-matrix");
  MATRIX_TIMER = setInterval(() => {
    let s = "";
    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 8; c++) s += Math.random() > 0.5 ? "1 " : "0 ";
      s += "\n";
    }
    el.textContent = s;
  }, 400);
}

async function tick() {
  let state = "open";
  try {
    const snap = await (await fetch("/api/snapshot")).json();
    state = snap.market_state || "open";
    renderHero(snap.totals || {});
    renderCurve(snap.equity_curve);
    renderPositions(snap.positions || []);
    renderIndices(snap.indices || []);
    renderAlloc(snap.allocation || []);
    renderSignals(snap.signals || []);
    renderLog(snap.log || []);
    renderTicker(snap.macro_ticker || []);
    renderPoly(snap.polymarket || []);
    $("topmeta").textContent = `${snap.ts} · ${state.toUpperCase()} · FX ${snap.fx_usdkrw?.toFixed?.(2) ?? "—"}`;
  } catch (e) {
    $("topmeta").textContent = "connection error — retrying";
  }
  const delay = state === "open" ? 30000 : 300000;
  setTimeout(tick, delay);
}

startMatrix();
tick();
