/* Fills every number from data/values.json, draws the model-risk bar, and runs the "try it" panel
   (live POST /api/estimate; precomputed data/fallback.json when the function is unreachable). */
(function () {
  "use strict";
  var SUP = { "⁻": "−", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "ⁿ": "n" };
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;"); }
  function fancy(s) {   // unicode superscripts -> <sup>
    return esc(s).replace(/[⁻⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ]+/g, function (m) { return "<sup>" + m.split("").map(function (c) { return SUP[c]; }).join("") + "</sup>"; });
  }
  function $(id) { return document.getElementById(id); }

  /* ---------- values ---------- */
  var values = {};
  fetch("data/values.json").then(function (r) { return r.json(); }).then(function (v) {
    values = v;
    document.querySelectorAll("[data-v]").forEach(function (el) {
      var k = el.getAttribute("data-v");
      if (k in v) el.innerHTML = fancy(v[k]);
    });
    var bar = function (id, left, width) { var e = $(id); if (e) { e.style.left = left; if (width) e.style.width = width; } };
    bar("rb-band", v.POS_LO, v.POS_W); bar("rb-brief", v.POS_BRIEF); bar("rb-med", v.POS_MED);
    bar("rc-brief", v.POS_BRIEF); bar("rc-med", v.POS_MED); bar("rc-ci", v.POS_MED);
  }).catch(function () { /* page still reads without numbers */ });

  /* ---------- try it ---------- */
  var BUDGETS = [2048, 4096, 8192, 16384, 32768, 65536];
  var TNAME = { expected_loss: "expected principal loss", joint_attachment: "both bonds pay", exceedance: "pool loss above US$80m",
                stop_loss: "pool stop-loss above US$80m", joint_exhaustion: "both bonds exhausted" };
  var fallback = null, seq = 0;

  function fmt(x, money) {
    if (x === null || x === undefined || isNaN(x)) return "–";
    if (money) return "US$" + (x >= 100 ? x.toFixed(1) : x.toFixed(3)) + "m";
    if (x === 0) return "0";
    if (x >= 1e-3) return (x * 100).toPrecision(3) + "%";
    var e = Math.floor(Math.log10(x)), m = x / Math.pow(10, e);
    return m.toPrecision(3) + "×10<sup>" + String(e).replace("-", "−") + "</sup>";
  }
  function range(c, money) { return "95% interval " + fmt(c[0], money) + " to " + fmt(c[1], money); }

  function budgetLabel() { $("budgetLab").textContent = BUDGETS[+$("budget").value].toLocaleString("en-US"); }

  function loadFallback() {
    if (fallback) return Promise.resolve(fallback);
    return fetch("data/fallback.json").then(function (r) { return r.json(); }).then(function (j) { fallback = j; return j; });
  }

  function request(body) {
    var ctl = ("AbortController" in window) ? new AbortController() : null;
    var timer = setTimeout(function () { if (ctl) ctl.abort(); }, 20000);
    return fetch("api/estimate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
                                   signal: ctl ? ctl.signal : undefined })
      .then(function (r) {
        clearTimeout(timer);
        var ct = r.headers.get("content-type") || "";
        if (!r.ok || ct.indexOf("json") < 0) throw new Error("function unavailable");
        return r.json();
      });
  }

  function run(randomSeed) {
    var my = ++seq;
    var body = { scenario: $("scenario").value, target: $("target").value, budget: BUDGETS[+$("budget").value],
                 threshold_musd: 80, seed: randomSeed ? Math.floor(Math.random() * 1e6) : 0 };
    $("status").className = "status"; $("status").textContent = "Running…";
    request(body).then(function (r) {
      if (my !== seq) return;
      render(r, "Live: computed on the server in " + r.seconds.toFixed(2) + " s (RDPO, training included)" + (randomSeed ? ", new random seed." : "."), false);
    }).catch(function () {
      return loadFallback().then(function (fb) {
        if (my !== seq) return;
        var r = fb[body.scenario + "|" + body.target + "|" + body.budget];
        if (!r) throw new Error("no precomputed cell");
        render(r, "The live function is not reachable from this page, so this is a precomputed answer (seed 0, same code and settings).", true);
      });
    }).catch(function () {
      if (my !== seq) return;
      $("status").className = "status warn"; $("status").textContent = "Could not compute this case. Please try again.";
    });
  }

  function row(name, sub, color, est, ci, money, extra) {
    return '<div class="mrow" style="--c:' + color + '"><div class="nm">' + name + "<small>" + sub + '</small></div>' +
           '<div class="val">' + fmt(est, money) + "</div><div class=\"ci\">" + range(ci, money) + (extra ? "<br>" + extra : "") + "</div></div>";
  }

  function render(r, status, isFallback) {
    var money = r.money, N = r.paths_used;
    $("status").className = "status" + (isFallback ? " warn" : ""); $("status").textContent = status;
    var mcx = money ? "" : (r.mc.hits === 0 ? "No path hit the event." : r.mc.hits + " of " + N.toLocaleString("en-US") + " paths hit the event.");
    $("rows").innerHTML =
      row("Exact", "dense enumeration", "var(--navy)", r.exact, [r.exact, r.exact], money, "no sampling error") +
      row("Plain Monte Carlo", N.toLocaleString("en-US") + " paths", "var(--ink2)", r.mc.estimate, r.mc.ci, money, mcx) +
      row("RDPO", N.toLocaleString("en-US") + " paths, training included", "var(--terra)", r.estimate, r.ci, money, "");
    plot(r, money);
    var wMC = r.mc.ci[1] - r.mc.ci[0], wR = r.ci[1] - r.ci[0];
    var inside = r.exact >= r.ci[0] && r.exact <= r.ci[1];
    $("verdict").textContent = "RDPO’s interval is " + (wR > 0 ? (wMC / wR).toFixed(wMC / wR >= 10 ? 0 : 1) : "∞") + "× narrower than plain Monte Carlo’s on the same budget" +
      (inside ? " and contains the exact value." : "; this run’s interval misses the exact value (about 1 run in 20 should).") +
      " Wall-clock aside: at equal CPU time simple methods win many cells, see the table above.";
  }

  /* interval plot: log axis for probabilities, linear for money */
  function plot(r, money) {
    var P = $("plot"), pos = [], vals = [r.exact, r.estimate].concat(r.ci, r.mc.ci, [r.mc.estimate]);
    var lo, hi, tx, ticks = [];
    if (!money) {
      var pv = vals.filter(function (v) { return v > 0; });
      var l = Math.floor(Math.log10(Math.min.apply(null, pv)) - 0.15), h = Math.min(0, Math.ceil(Math.log10(Math.max.apply(null, pv)) + 0.05));
      if (h - l < 1) l = h - 1;
      tx = function (v) { return v <= 0 ? 0 : Math.min(1, Math.max(0, (Math.log10(v) - l) / (h - l))); };
      var step = Math.max(1, Math.ceil((h - l) / 6));
      for (var e = l; e <= h; e += step) ticks.push([e, "10<sup>" + String(e).replace("-", "−") + "</sup>"]);
      ticks = ticks.map(function (t) { return [tx(Math.pow(10, t[0])), t[1]]; });
    } else {
      lo = Math.min.apply(null, vals); hi = Math.max.apply(null, vals);
      var pad = (hi - lo) * 0.18 || hi * 0.1; lo = Math.max(0, lo - pad); hi = hi + pad;
      tx = function (v) { return Math.min(1, Math.max(0, (v - lo) / (hi - lo))); };
      var raw = (hi - lo) / 4, mag = Math.pow(10, Math.floor(Math.log10(raw))), st = [1, 2, 5, 10].map(function (k) { return k * mag; }).filter(function (k) { return k >= raw; })[0];
      for (var t = Math.ceil(lo / st) * st; t <= hi; t += st) ticks.push([tx(t), "US$" + t.toFixed(st < 0.1 ? 3 : st < 1 ? 2 : 1) + "m"]);
    }
    var html = '<div class="axis">';
    ticks.forEach(function (t) { html += '<div class="tick" style="left:' + (100 * t[0]) + '%"></div><div class="tl" style="left:' + (100 * t[0]) + '%;bottom:-24px">' + t[1] + "</div>"; });
    function bar(y, name, color, ci, est) {
      var a = tx(ci[0]), b = tx(ci[1]), clipped = !money && ci[0] <= 0;
      var h = '<div class="rl" style="left:0;top:' + (y - 22) + 'px;color:' + color + '">' + name + "</div>";
      h += '<div class="iv" style="--c:' + color + ";left:" + (100 * a) + "%;width:max(" + (100 * (b - a)) + "%,8px);top:" + (y - 6) + 'px"></div>';
      if (clipped) h += '<div class="arrowl" style="--c:' + color + ";left:" + (100 * a) + "%;top:" + y + 'px">◀ 0</div>';
      if (est > 0 || money) h += '<div class="pt" style="--c:' + color + ";left:" + (100 * tx(est)) + "%;top:" + y + 'px"></div>';
      return h;
    }
    html += bar(52, "Plain Monte Carlo", "var(--ink2)", r.mc.ci, r.mc.estimate);
    html += bar(104, "RDPO", "var(--terra)", r.ci, r.estimate);
    html += '<div class="ex" style="left:' + (100 * tx(r.exact)) + '%"></div><div class="exl" style="left:' + (100 * tx(r.exact)) + '%">exact</div>';
    html += "</div>";
    P.innerHTML = html;
    P.setAttribute("aria-label", "Interval plot. Exact " + (money ? r.exact.toFixed(3) + " million US dollars" : (r.exact * 100).toPrecision(3) + " percent") +
      ". RDPO interval narrower than Monte Carlo's.");
  }

  ["scenario", "target", "budget"].forEach(function (id) {
    $(id).addEventListener("change", function () { budgetLabel(); run(false); });
  });
  $("budget").addEventListener("input", budgetLabel);
  $("form").addEventListener("submit", function (e) { e.preventDefault(); run(true); });
  budgetLabel();
  run(false);
})();
