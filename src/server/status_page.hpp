#pragma once

namespace mako::server {

// GET /status: a page that polls /v1/status every 2 s (with an API key, kept in the browser, when the server needs one).
inline constexpr const char* status_page = R"html(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mako status</title>
<style>
:root { --bg: #fafaf9; --fg: #1c1917; --dim: #78716c; --line: #e7e5e4; --card: #fff; --ok: #15803d; --warn: #b45309;
        --bad: #b91c1c; --accent: #2563eb; color-scheme: light; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0c0a09; --fg: #e7e5e4; --dim: #a8a29e; --line: #292524; --card: #1c1917; --ok: #4ade80;
          --warn: #fbbf24; --bad: #f87171; --accent: #60a5fa; color-scheme: dark; }
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body { margin: 0; padding: 20px 16px 40px; background: var(--bg); color: var(--fg);
       font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .06em; color: var(--dim); margin: 24px 0 8px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 16px; }
.sub { color: var(--dim); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 14px; }
.tile { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
.tile .k { color: var(--dim); font-size: 12px; }
.tile .v { font-size: 20px; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; background: var(--card); border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--line); }
th { color: var(--dim); font-weight: 500; font-size: 12px; }
th:first-child, td:first-child, .l { text-align: left; }
tr:last-child td { border-bottom: 0; }
.empty { color: var(--dim); padding: 10px 12px; }
.pill { display: inline-block; padding: 0 7px; border-radius: 9px; font-size: 12px; border: 1px solid currentColor; }
.generating, .live, .nominal { color: var(--ok); }
.reading, .waiting, .fair { color: var(--warn); }
.serious, .critical { color: var(--bad); }
#err { color: var(--bad); margin-top: 10px; }
#key { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
#key input { flex: 1; min-width: 200px; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px;
             background: var(--card); color: var(--fg); font: inherit; }
#key button { padding: 6px 12px; border: 0; border-radius: 6px; background: var(--accent); color: #fff; font: inherit; }
</style>
</head>
<body>
<main>
<header><h1 id="model">mako</h1><span class="sub" id="facts"></span></header>
<form id="key" hidden><input id="keyin" type="password" placeholder="API key" autocomplete="current-password">
<button>Use key</button></form>
<div id="err" hidden></div>
<div class="tiles" id="tiles"></div>
<h2>Requests</h2><div class="scroll" id="requests"></div>
<h2>Conversations</h2><div class="scroll" id="conversations"></div>
<h2>Recent</h2><div class="scroll" id="recent"></div>
</main>
<script>
const $ = id => document.getElementById(id);
let key = "";
try { key = localStorage.getItem("mako-key") || ""; } catch (e) {}
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[c]);
const num = (v, d = 0) => v == null ? "–" : Number(v).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d});
const dur = s => s < 90 ? num(s) + " s" : s < 5400 ? num(s / 60) + " min" : num(s / 3600, 1) + " h";
const pill = s => `<span class="pill ${esc(s)}">${esc(s)}</span>`;
function table(el, cols, rows, none) {
  if (!rows.length) { el.innerHTML = `<div class="empty">${none}</div>`; return; }
  el.innerHTML = "<table><tr>" + cols.map(c => `<th${c[2] ? ' class="l"' : ""}>${c[0]}</th>`).join("") + "</tr>" +
    rows.map(r => "<tr>" + cols.map(c => `<td${c[2] ? ' class="l"' : ""}>${c[1](r)}</td>`).join("") + "</tr>").join("") +
    "</table>";
}
function render(s) {
  $("model").textContent = s.model;
  document.title = s.model + " · mako";
  $("facts").innerHTML = `context ${num(s.context)} tokens · up ${dur(s.uptime_s)} · thermal ${pill(s.thermal)}`;
  const c = s.expert_cache;
  const tiles = [
    ["in flight", num(s.in_flight)], ["waiting for a turn", num(s.waiting)], ["requests served", num(s.served)],
    ["expert cache", num(c.gb, 1) + " GB"], ["experts resident", num(c.resident * 100) + "%"],
    ["cache hits", c.hit_share == null ? "–" : num(c.hit_share * 100, 1) + "%"],
    ["parked", num(s.conversations.parked_gb, 2) + " / " + num(s.conversations.budget_gb, 1) + " GB"],
  ];
  $("tiles").innerHTML = tiles.map(t => `<div class="tile"><div class="k">${t[0]}</div><div class="v">${t[1]}</div></div>`).join("");
  table($("requests"), [
    ["id", r => esc(r.id), 1], ["phase", r => pill(r.phase), 1], ["prompt", r => num(r.prompt)],
    ["generated", r => num(r.generated)], ["tok/s", r => num(r.tokens_per_s, 1)], ["for", r => dur(r.elapsed_s)],
  ], s.requests, "none");
  table($("conversations"), [
    ["#", r => r.index, 1], ["state", r => pill(r.state) + (r.busy ? " busy" : ""), 1], ["tokens", r => num(r.tokens)],
    ["parked", r => r.state == "live" ? "–" : num(r.parked_mb) + " MB"], ["last used", r => dur(r.idle_s) + " ago"],
  ], s.conversations.list, "none yet");
  table($("recent"), [
    ["id", r => esc(r.id), 1], ["end", r => esc(r.finish), 1], ["conv.", r => r.conversation],
    ["prompt", r => num(r.prompt)], ["reused", r => num(r.reused)], ["prefill tok/s", r => num(r.prefill_tokens_per_s)],
    ["generated", r => num(r.generated)], ["tok/s", r => num(r.tokens_per_s, 1)],
    ["misses/token", r => num(r.misses_per_token, 1)], ["ended", r => dur(r.ago_s) + " ago"],
  ], s.recent, "none yet");
}
async function poll() {
  try {
    const res = await fetch("/v1/status", {headers: key ? {"Authorization": "Bearer " + key} : {}});
    if (res.status == 401) { $("key").hidden = false; throw new Error("this server needs an API key"); }
    if (!res.ok) throw new Error("HTTP " + res.status);
    render(await res.json());
    $("err").hidden = true;
  } catch (e) {
    $("err").textContent = e.message;
    $("err").hidden = false;
  }
}
$("key").addEventListener("submit", e => {
  e.preventDefault();
  key = $("keyin").value.trim();
  try { localStorage.setItem("mako-key", key); } catch (e) {}
  $("key").hidden = true;
  poll();
});
poll();
setInterval(poll, 2000);
</script>
</body>
</html>
)html";

} // namespace mako::server
