"""Whole-system test: is each strong add-on usable in practice? Ten dimensions, thresholds set before running.

  1 reproducible   a clean rerun (empty output folder, the friend's current main) gives the committed numbers
  2 seed-stable    two more seeds: same verdict, headline numbers move less than their own reported interval half-width
  3 input-robust   the headline conclusion survives >= 80% of the listed perturbations of its inputs
  4 timely         live mark <= 10 s; event response <= 5 min; yearly twin refresh <= 60 min; everything <= 2 h
  5 predictive     an out-of-sample or exact check exists (listed, not re-run here)
  6 decision       moves the indicative price by >= 0.5 point, or changes the ranking of what data to collect
  7 honest         every reported interval either has a tested coverage or is marked "not tested"
  8 portable       without the data drive: public data is fetched into a local cache, or the error says what to run
  9 consistent     numbers that must agree across modules agree
 10 end-to-end     replay 2017-2026 as if live (desk.py, phase 2 of this work)

    python scorecard.py rerun <label> <seed offset>     full pipeline into results_scorecard/<label> (hours)
    python scorecard.py perturb                          dimension 3 and 9, 8 (minutes) -> results/scorecard_perturb.json
    python scorecard.py                                  the table from everything available -> results/scorecard.json
Reruns never touch addons/results/: the modules write to ADDON_OUT and shift their seeds by ADDON_SEED_OFFSET (_src.py).
"""
import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results_scorecard"
RES = HERE / "results"
PIPELINE = ["rotations", "q2_ae", "combine", "eq_cluster", "rates_unc", "climate_cc", "sensitivity", "nested_uq", "voi", "model_risk",
            "twin", "event_response", "spread_band", "live_mark"]
J = "term_both_bonds_exhausted_probability"


def rerun(label, offset):
    out = RUNS / label
    (out / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ADDON_OUT=str(out), ADDON_SEED_OFFSET=str(offset), PYTHONIOENCODING="utf-8")
    if int(offset):                                         # seed runs reuse the yearly counts (deterministic, 20 minutes of declustering)
        shutil.copy(HERE / "results" / "twin_counts.json", out / "twin_counts.json")
    timing = json.loads((out / "_timing.json").read_text()) if (out / "_timing.json").exists() else {}
    for m in PIPELINE:
        if timing.get(m, {}).get("exit") == 0:
            continue
        t0 = time.perf_counter()
        with open(out / "logs" / f"{m}.log", "w", encoding="utf-8") as log:
            code = subprocess.run([sys.executable, "-W", "ignore", f"{m}.py"], cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        timing[m] = {"seconds": time.perf_counter() - t0, "exit": code}
        (out / "_timing.json").write_text(json.dumps(timing, indent=1))
        print(f"{label} {m}: exit {code}, {timing[m]['seconds']:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------- helpers
def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def claim(text, rows):
    """rows: (perturbation, value shown, holds). The first row is the unperturbed case."""
    pert = rows[1:]
    return {"claim": text, "base": {"value": rows[0][1], "holds": bool(rows[0][2])},
            "perturbations": [{"what": w, "value": v, "holds": bool(h)} for w, v, h in pert],
            "survival": float(sum(bool(h) for _, _, h in pert) / max(len(pert), 1)), "n": len(pert)}


@contextlib.contextmanager
def patched(mod, **kw):
    old = {k: getattr(mod, k) for k in kw}
    for k, v in kw.items():
        setattr(mod, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(mod, k, v)


# ------------------------------------------------------------------------------------------- dimension 3: perturbations
def p_eq_cluster():
    import numpy as np
    from scipy import stats
    import eq_cluster as EC
    import variant
    from _src import in_repo
    EQ = EC.EQ
    with in_repo():
        ev, polys = EQ.load()
    t = np.array([e[0].toordinal() for e in ev]); la = np.array([e[1] for e in ev]); lo = np.array([e[2] for e in ev]); m = np.array([e[3] for e in ev])
    yr = np.array([e[0].year for e in ev])
    d = np.minimum(*[np.array([EQ.dist_to_country(a, o, polys[c]) for a, o in zip(la, lo)]) for c in ("KG", "TJ")])

    def kept(idx, scale):                                   # Gardner-Knopoff windows as in the friend's decluster, scaled
        mm, removed = m[idx], np.zeros(len(idx), bool)
        for i in np.argsort(-mm, kind="stable"):
            if removed[i]:
                continue
            L = 10 ** (0.1238 * mm[i] + 0.983) * scale
            T = (10 ** (0.032 * mm[i] + 2.7389) if mm[i] >= 6.5 else 10 ** (0.5409 * mm[i] - 0.547)) * scale
            hit = (~removed) & (mm <= mm[i]) & (np.abs(t[idx] - t[idx][i]) <= T) & (EQ.haversine(la[idx][i], lo[idx][i], la[idx], lo[idx]) <= L)
            hit[i] = False
            removed |= hit
        return idx[~removed]

    full = np.flatnonzero((yr >= EQ.YEARS[0]) & (yr <= EQ.YEARS[1]))
    assert len(kept(full, 1.0)) == len(EQ.decluster([ev[i] for i in full]))     # same windows as the friend's function
    base = variant.targets()[J]

    def factor(radius=50.0, M=6.5, years=EQ.YEARS, scale=1.0, low=False):
        idx = np.flatnonzero((yr >= years[0]) & (yr <= years[1]))
        k = kept(idx, scale)
        raw, dc = int(((d[idx] <= radius) & (m[idx] >= M)).sum()), int(((d[k] <= radius) & (m[k] >= M)).sum())
        nu = stats.gamma.ppf(0.025, raw - dc + 0.5) / max(dc, 1) if low else (raw - dc) / max(dc, 1)
        return variant.targets(nu=float(nu))[J] / base, float(nu)

    cases = [("base: 50 km, nu at M>=6.5, 1965-2016", {}), ("radius 25 km", {"radius": 25.0}), ("radius 100 km", {"radius": 100.0}),
             ("nu measured at M>=6.0", {"M": 6.0}), ("nu measured at M>=6.9", {"M": 6.9}), ("drop the first 10 years", {"years": (1975, 2016)}),
             ("drop the last 10 years", {"years": (1965, 2006)}), ("aftershock windows x0.5", {"scale": 0.5}), ("aftershock windows x2", {"scale": 2.0}),
             ("nu at the low end of its 95% interval", {"low": True})]
    rows = []
    for name, kw in cases:
        f, nu = factor(**kw)
        rows.append((name, f"x{f:.2f} (nu {nu:.2f})", f >= 1.5))
    return claim("aftershocks paid separately raise joint exhaustion by >= 1.5x", rows)


def p_rates_unc():
    import numpy as np
    from scipy import stats
    import rates_unc as RU
    from _src import load
    eq = load("realdata_eq.json"); gr = eq["gutenberg_richter"]
    cd0 = RU.country_data()
    gml0 = RU.gev_gml

    def width(radius="50", bse=1.0, shift=0.0, prior=True, drop=None):
        def eqm(rng, n):
            b = rng.normal(gr["b"], gr["b_se"] * bse, n)
            out = {}
            for c in ("KG", "TJ"):
                k, mt = eq["radii_km"][radius]["5.5"][c]["events"], gr[c]["implied_trigger_magnitude"] + shift
                out[c] = rng.gamma(k + 0.5, 1.0, n) / k * 10 ** (-(b - gr["b"]) * (mt - 5.5))
            return out, {}
        cd = cd0 if drop is None else {c: dict(v, years=sorted(v["years"])[10:] if drop == "first" else sorted(v["years"])[:-10]) for c, v in cd0.items()}
        with patched(RU, N=400, NB=120, save=lambda *a: None, eq_multipliers=eqm, country_data=lambda: cd,
                     gev_gml=gml0 if prior else (lambda x: stats.genextreme.fit(x, 0.0))):
            o = quiet(RU.run)
        s, f = o["sweeps"]["all"]["summary"]["joint_exhaustion"]["ci95"], o["friend_shares_only_joint_exhaustion"]["ci95"]
        return (s[1] - s[0]) / (f[1] - f[0])

    cases = [("base (400 draws)", {}), ("radius 25 km", {"radius": "25"}), ("radius 100 km", {"radius": "100"}), ("b standard error x0.5", {"bse": 0.5}),
             ("b standard error x2", {"bse": 2.0}), ("trigger magnitude +0.2", {"shift": 0.2}), ("trigger magnitude -0.2", {"shift": -0.2}),
             ("rain GEV without the shape prior", {"prior": False}), ("rain: drop the first 10 years", {"drop": "first"}), ("rain: drop the last 10 years", {"drop": "last"})]
    rows = []
    for name, kw in cases:
        w = width(**kw)
        rows.append((name, f"x{w:.2f}", w >= 2.0))
    return claim("with base-rate data uncertainty the joint-exhaustion 95% interval is >= 2x the shares-only one", rows)


def p_climate():
    import numpy as np
    from scipy import stats
    import climate_cc as CC
    import rates_unc as RU
    import variant
    cd = RU.country_data()

    def factor(rate=1.07, prior=True, drop=None, scen="brief (adb-base)", dT=2.0):
        mult = []
        for c in ("KG", "TJ"):
            ys = sorted(cd[c]["years"])
            ys = set(ys[10:] if drop == "first" else ys[:-10] if drop == "last" else ys)
            a = {}
            for key, v in cd[c]["idx"].items():
                y = int(key[:4])
                if y in ys:
                    a[y] = max(a.get(y, 0.0), v)
            x = np.array(list(a.values()))
            g = RU.gev_gml(x) if prior else stats.genextreme.fit(x, 0.0)
            p = stats.genextreme.sf(stats.genextreme.isf(0.02, *g) / rate ** dT, *g)
            mult.append(-math.log1p(-p) / -math.log1p(-0.02))
        sc = CC.scenarios()[scen]
        base = CC.targets_of(sc, CC.term_distribution(variant.annual(sc)))[J]
        return CC.targets_of(sc, CC.term_with(sc, [mult] * CC.YEARS))[J] / base

    cases = [("base: 7% per degree, brief shares", {}), ("5% per degree", {"rate": 1.05}), ("9% per degree", {"rate": 1.09}), ("GEV without the shape prior", {"prior": False}),
             ("drop the first 10 years", {"drop": "first"}), ("drop the last 10 years", {"drop": "last"}), ("rain share from data", {"scen": "rain share from data"})]
    rows = []
    for name, kw in cases:
        f = factor(**kw)
        rows.append((name, f"x{f:.2f}", f >= 2.0))
    return claim("+2 C raises joint exhaustion by >= 2x", rows)


def p_event():
    import numpy as np
    import event_response as ER
    cat = ER.load()

    def verdict(**kw):
        with patched(ER, **kw):
            d = ER.design(cat)
            tr = (d["year"] >= ER.TRAIN[0]) & (d["year"] < ER.TRAIN[1] + 1); te = (d["year"] >= ER.TEST[0]) & (d["year"] < ER.TEST[1] + 1)
            par = ER.fit(d, tr)
            ev, ok1, ok2 = ER.evaluate(d, par, te, np.random.default_rng(1))
            imp = ER.event_impact(cat, par)
        return f"gain {ev['30']['log_score_gain']:+.2f}/{ev['365']['log_score_gain']:+.2f}, bins {ev['30']['bins_ok']}/{ev['365']['bins_ok']}", ok1 and ok2, imp

    cases = [("base", {}), ("radius 25 km", {"R_KM": 25.0}), ("radius 100 km", {"R_KM": 100.0}), ("target M>=6.0", {"M_TARGET": 6.0}), ("triggers M>=5.5 only", {"M_TRIG": 5.5}),
             ("train 1973-1994, test 1995-2024", {"TRAIN": (1973, 1994), "TEST": (1995, 2024)}), ("train 1973-2004, test 2005-2024", {"TRAIN": (1973, 2004), "TEST": (2005, 2024)})]
    rows, extra = [], {}
    for name, kw in cases:
        v, ok, imp = verdict(**kw)
        rows.append((name, v, ok))
    for mag in (5.2, 5.5):                                  # USGS value and the Kyrgyz agency's value for the 3 Nov 2026 event
        _, _, imp = verdict(EVENT=dict(ER.EVENT, mag=mag))
        extra[f"event_M{mag}_added_pay_probability_KG"] = imp["pay_size_KG"]["p_added_by_event"]
    out = claim("the after-event forecast beats the stationary Poisson baseline out of sample and is calibrated (both rules)", rows)
    out["event_magnitude_reading"] = extra
    return out


def p_backtest():
    import numpy as np
    from scipy.stats import nbinom
    c = json.loads((RES / "twin_counts.json").read_text())
    K, Kp = np.asarray(c["K_yearly"]), np.asarray(c["Kp_yearly"])
    r0 = 52
    fk = np.array([x["friend"] for x in c["check_1965_2016"]["countries"]]); fp = np.array([x["friend"] for x in c["check_1965_2016"]["pairs"]])

    def cov(k0, Y, a, level):
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        k, n, out = np.asarray(k0, float).copy(), float(r0), []
        for t in range(Y.shape[0]):
            d = nbinom(k + a, n / (n + 1.0))
            lo, hi = d.cdf(Y[t] - 1), d.cdf(Y[t])
            out.append(np.clip(np.minimum(hi, hi_q) - np.maximum(lo, lo_q), 0, None) / np.maximum(hi - lo, 1e-300))
            k, n = k + Y[t], n + 1
        return float(np.concatenate(out).mean())

    cases = [("base: Jeffreys prior, 90% interval, friend's counts to 2016", 0.5, 0.90, "friend"), ("prior weight 1 (uniform-like)", 1.0, 0.90, "friend"), ("prior weight 0.05", 0.05, 0.90, "friend"),
             ("80% interval", 0.5, 0.80, "friend"), ("95% interval", 0.5, 0.95, "friend"), ("ComCat counts to 2016 as the start", 0.5, 0.90, "comcat")]
    rows = []
    for name, a, level, base in cases:
        k0, p0 = (fk, fp) if base == "friend" else (K[:r0].sum(0), Kp[:r0].sum(0))
        cc, cp = cov(k0, K[r0:], a, level), cov(p0, Kp[r0:], a, level)
        rows.append((name, f"{cc:.3f} / {cp:.3f}", all(level - 0.05 <= x <= min(level + 0.05, 1.0) for x in (cc, cp))))
    return claim("next-year count forecasts are calibrated: coverage within 5 points of the nominal level, countries and pairs", rows)


def p_live_mark():
    from datetime import date, timedelta
    import live_mark as LM
    nu_ci = json.loads((RES / "quality_eq_cluster.json").read_text())["by_magnitude"]["6.5"]["KG_or_TJ"]["nu_ci95"]

    def deltas(**kw):
        with patched(LM, save=lambda *a: None, **kw):
            r = quiet(LM.run)
        b = r["cases"]["base (no event, no payout)"]
        d = lambda name: max(abs(r["cases"][name][c]["price"] - b[c]["price"]) for c in LM.BONDS)
        x = {"rate": max(d("earthquake rate at 2.5% of data interval"), d("earthquake rate at 97.5% of data interval")), "clause": d("aftershocks paid separately (nu = 0.5)"),
             "warming": d("rain +2C"), "event": d("3 Nov 2026 border earthquake")}
        ok = x["rate"] > x["clause"] > x["event"] and x["warming"] > x["event"] and min(x["rate"], x["clause"], x["warming"]) >= 0.5 and x["event"] < 0.5
        return f"rate {x['rate']:.1f}, clause {x['clause']:.1f}, +2C {x['warming']:.1f}, event {x['event']:.2f} points", ok

    cases = [("base", {}), ("maturity 90 days earlier", {"MATURITY": LM.MATURITY - timedelta(days=90)}), ("maturity 90 days later", {"MATURITY": LM.MATURITY + timedelta(days=90)}),
             ("one month after issue", {"ASOF": date(2026, 5, 24)}), ("one year before maturity", {"ASOF": date(2028, 4, 24)}), ("nu at the low end (%.2f)" % nu_ci[0], {"NU_B": nu_ci[0]}),
             ("nu at the high end (%.2f)" % nu_ci[1], {"NU_B": nu_ci[1]}), ("warming factors at the low end", {"CC": {"+1C": 1.5, "+2C": 2.2}}), ("warming factors at the high end", {"CC": {"+1C": 1.6, "+2C": 2.6}})]
    rows = [(name, *deltas(**kw)) for name, kw in cases]
    out = claim("in price points: rate uncertainty > aftershock clause > one M5 event; +2 C > the event; the three big ones >= 0.5 point, the event < 0.5", rows)
    # a stress that no add-on covered: the contract reads GFZ magnitudes, the data here are USGS; +-0.2 magnitude units
    T, tau = LM.years(LM.ISSUE, LM.MATURITY), LM.years(LM.ASOF, LM.MATURITY)
    at_issue, now = LM.remaining(T), LM.remaining(tau)
    stress = {}
    for off in (-0.2, 0.2):
        f = 10 ** (0.92 * off)
        r = LM.remaining(tau, eq_mult=(f, f))
        stress[f"{off:+.1f}"] = {c: LM.price(r[c]["el_pct"], tau, LM.COUPON / (at_issue[c]["el_pct"] / T)) - LM.price(now[c]["el_pct"], tau, LM.COUPON / (at_issue[c]["el_pct"] / T)) for c in LM.BONDS}
    out["magnitude_agency_offset_price_points"] = stress
    return out


def p_spread():
    import spread_band as SB

    def envelope(mult=SB.BG["mult"], amb=SB.AMB, use=("bg", "ibrd", "amb")):
        ok, shown = True, []
        for c, el in SB.ISSUED.items():
            v = ([SB.bodoff_gan(el, mult)] if "bg" in use else []) + ([SB.ibrd(el)] if "ibrd" in use else []) + ([m * el for m in amb] if "amb" in use else [])
            ok &= min(v) <= SB.SPREAD <= max(v)
            shown.append(f"{c} {min(v):.2f}-{max(v):.2f}%")
        return ", ".join(shown), ok

    cases = [("base: three curves", {}), ("Bodoff-Gan multiplier 1.41 (low end)", {"mult": 1.41}), ("Bodoff-Gan multiplier 1.85 (high end)", {"mult": 1.85}),
             ("AM Best 2026 only (2.4-2.5)", {"amb": (2.4, 2.5)}), ("without AM Best", {"use": ("bg", "ibrd")}), ("without Bodoff-Gan", {"use": ("ibrd", "amb")})]
    return claim("the 6.00% paid lies inside the range of the market curves at the issued expected loss, both notes", [(n, *envelope(**kw)) for n, kw in cases])


def p_quantum():
    from rotations import price
    from _src import load

    def minfactor(adv, q2, models=("qrom_fallback", "qrom_mixed"), scale=1.0, use_q2=True):
        out = []
        for r in adv:
            calls = q2[(r["topology"], r["n"])]["chebae_odd"]["calls_mean"] * scale if use_q2 else None
            out += [r["required_speedup"] / price(r, mdl, calls)["s_star"] for mdl in models]
        return min(out)

    q2 = lambda path: {(r["topology"], r["n"]): r for r in json.loads(path.read_text())["rows"]}
    adv, q = load("quantum_advantage.json")["advantage"], q2(RES / "q2_ae.json")
    rows = [("base: author's baseline, both tiers", minfactor(adv, q)), ("ChebAE needs 50% more calls", minfactor(adv, q, scale=1.5)),
            ("no QROM state-preparation layer", minfactor(adv, q, models=("fallback", "mixed")))]
    rep, old = HERE.parent / "adb-repro" / "results" / "quantum_advantage.json", HERE / "results_d29bd43" / "q2_ae.json"
    if rep.exists() and old.exists():
        rows.append(("our reproduction's baseline instead of the author's", minfactor(json.loads(rep.read_text())["advantage"], q2(old))))
    out = claim("the required hardware speed-up shrinks >= 10x on every setting", [(n, f"min x{v:.1f}", v >= 10) for n, v in rows])
    out["ablation_min_factor"] = {"rotation synthesis only": minfactor(adv, q, use_q2=False), "ChebAE only": minfactor(adv, q, models=("rs",))}
    return out


# ------------------------------------------------------------------------------------------- dimensions 9 and 8
def consistency():
    import live_mark as LM
    import model_risk as MR
    import spread_band as SB
    rows = []
    add = lambda what, value, ok: rows.append({"what": what, "value": value, "ok": bool(ok)})
    try:
        quiet(LM.selftest); add("live mark over 3 years = the friend's term expected loss", "equal", True)
    except AssertionError as e:
        add("live mark over 3 years = the friend's term expected loss", str(e)[:80], False)
    try:
        quiet(MR.selftest); add("CAREC engine at the observed data = the friend's MPS references", "< 1e-6", True)
    except AssertionError as e:
        add("CAREC engine at the observed data = the friend's MPS references", str(e)[:80], False)
    tw = json.loads((RES / "twin.json").read_text())["friend_2016"]
    d = max(abs(tw["engine_at_observed"][str(T)] / tw["exact_refs"][f"exceedance|{T}"] - 1) for T in (450, 850))
    add("twin at the 2016 state vs the friend's exact Asia-Pacific references", f"max rel diff {d:.4f}", d <= 0.01)
    ru, vo = json.loads((RES / "quality_rates_unc.json").read_text()), json.loads((RES / "voi.json").read_text())
    f, s = ru["friend_shares_only_joint_exhaustion"], ru["sweeps"]["shares_only"]["summary"]["joint_exhaustion"]
    d = abs(s["median"] / f["median"] - 1)
    add("rates_unc with shares only vs the friend's own posterior (different draws)", f"median rel diff {d:.3f}", d <= 0.05)
    same = all(ru["posteriors"][a][k] == vo["posteriors"][b][k] for a, b in (("earthquake_share", "eq_share"), ("rain_share", "rain_share")) for k in ("a", "b"))
    add("voi and rates_unc use the same share posteriors", "identical" if same else "differ", same)
    sb = json.loads((RES / "spread_band.json").read_text())
    d = max(abs(sb["bonds"][c]["el_interval_pct"][i] - SB.ISSUED[c] * ru["multiplier_q"][f"eq{c}"][i]) for c in SB.ISSUED for i in range(3))
    add("spread band's expected-loss interval = issued EL x rates_unc multipliers", f"max diff {d:.1e}", d < 1e-9)
    er, tc = json.loads((RES / "event_response.json").read_text())["event_2026_11_03"], json.loads((RES / "twin_counts.json").read_text())
    kg_tj = sum(x["friend"] for x in tc["check_1965_2016"]["countries"] if x["iso"] in ("KG", "TJ")) / 52
    add("event response's local M>=5.5 rate (one 50 km disc) is below the KG + TJ country rate in the twin", f"{er['local_rate_per_year']:.2f} vs {kg_tj:.2f} per year", er["local_rate_per_year"] < kg_tj)
    known = {"KG-TJ joint exhaustion differs across the three models by design": {"brief two-country model (25% rain share)": ru["brief_point"]["joint_exhaustion"],
             "Asia-Pacific model (2.2% rain share, return-period cover)": tw["exact_refs"]["joint_exhaustion|None"]}}
    return {"checks": rows, "all_ok": all(r["ok"] for r in rows), "known_differences": known}


def portability():
    """No data drive: do the two data-dependent modules fetch the public catalogues themselves?"""
    tmp = tempfile.mkdtemp(prefix="addon_data_")
    env = dict(os.environ, DATA_ROOT=tmp, PYTHONIOENCODING="utf-8")
    out = {}
    for name, code, ref in (("event_response", "import event_response as e; print(len(e.load()['t']))", "import event_response as e; print(len(e.load()['t']))"),
                            ("twin", "import fetch_usgs as f, twin; print(len(twin.comcat_events(f.asia_pacific())))", "import fetch_usgs as f, twin; print(len(twin.comcat_events(f.asia_pacific())))")):
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=HERE, env=env, capture_output=True, text=True)
        here = subprocess.run([sys.executable, "-W", "ignore", "-c", ref], cwd=HERE, env=dict(os.environ, PYTHONIOENCODING="utf-8"), capture_output=True, text=True)
        n_new = int(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else None
        n_ref = int(here.stdout.strip().splitlines()[-1]) if here.returncode == 0 and here.stdout.strip() else None
        out[name] = {"fetched_without_data_drive": r.returncode == 0, "seconds": time.perf_counter() - t0, "events_fetched_now": n_new, "events_in_library": n_ref,
                     "error": None if r.returncode == 0 else r.stderr.strip().splitlines()[-1][:200]}
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def perturb():
    res = {}
    for name, f in (("eq_cluster", p_eq_cluster), ("rates_unc", p_rates_unc), ("climate_cc", p_climate), ("event_response", p_event), ("twin backtest", p_backtest),
                    ("live_mark", p_live_mark), ("spread_band", p_spread), ("rotations + ChebAE", p_quantum)):
        t0 = time.perf_counter()
        res[name] = f()
        r = res[name]
        print(f"{name}: {r['claim']}\n   base {r['base']['value']} ({'holds' if r['base']['holds'] else 'FAILS'}); survival {r['survival']:.0%} of {r['n']} ({time.perf_counter() - t0:.0f}s)", flush=True)
        for p in r["perturbations"]:
            print(f"     {'ok ' if p['holds'] else 'NO '} {p['what']}: {p['value']}", flush=True)
    res["_consistency"] = consistency()
    for r in res["_consistency"]["checks"]:
        print(f"consistency {'ok ' if r['ok'] else 'NO '} {r['what']}: {r['value']}", flush=True)
    res["_portability"] = portability()
    print("portability:", res["_portability"], flush=True)
    (RES / "scorecard_perturb.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "rerun":
        rerun(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "0")
    elif len(sys.argv) >= 2 and sys.argv[1] == "perturb":
        perturb()
