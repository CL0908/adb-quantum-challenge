"""Quality 6: data-gap audit of the friend's GHCN rain pipeline (lessons from QContagion's data/interface.py:
"no data" must not become "no event", fills must be bounded, statistics need observation counts). Additive.

What the friend's rain_check already does right: no forward fill, quality-flagged values and the [99, 100) mm source
codes are dropped, a year counts only with >= 300 valid days. What is measured here, with the friend's own functions:
  1. Missing days inside the valid years, per country, and the events they could hide (event rate per valid day ×
     missing days).
  2. Shared-share bias: an event of the rarer country can only be "shared" if the other country has data in the
     ±1-day window. Share among events whose window is fully observed on the other side vs the friend's share.
  3. Threshold sensitivity: the (1 − q) quantile of annual maxima from years with >= 300 valid days (friend) vs only
     near-complete years (>= 355 days).
"""
from datetime import date, timedelta

import numpy as np

from _src import load, save
from rain_lag import country_data
from src.realdata import rain_check as RC


def valid_days(ci):
    return {c: {date(int(k[:4]), int(k[4:6]), int(k[6:])) for k in v["idx"] if int(k[:4]) in set(v["years"])}
            for c, v in ci.items()}


def run():
    ci = country_data()
    vd = valid_days(ci)
    out = {"countries": {}, "pairs": {}, "pooled": {}, "threshold": {}}
    for c, v in ci.items():
        ys = v["years"]
        total = sum(366 if (y % 4 == 0 and (y % 100 or y % 400 == 0)) else 365 for y in ys)
        miss = total - len(vd[c])
        ev, _ = RC.events(v["idx"], ys, 0.5)
        rate = len(ev) / max(len(vd[c]), 1)
        out["countries"][c] = {"valid_years": len(ys), "missing_days": miss, "missing_share": miss / total,
                               "events_q50": len(ev), "events_possibly_hidden_q50": rate * miss}
        # threshold sensitivity
        amax_all, amax_full = [], []
        per_year = {}
        for d in vd[c]:
            per_year[d.year] = per_year.get(d.year, 0) + 1
        for y in ys:
            vals = [x for k, x in v["idx"].items() if int(k[:4]) == y]
            amax_all.append(max(vals))
            if per_year.get(y, 0) >= 355:
                amax_full.append(max(vals))
        out["threshold"][c] = {}
        for q in ("0.5", "0.2"):
            out["threshold"][c][q] = {"friend": float(np.quantile(amax_all, 1 - float(q))),
                                      "near_complete_years": float(np.quantile(amax_full, 1 - float(q))) if len(amax_full) >= 10 else None,
                                      "n_near_complete": len(amax_full)}
    for q in (0.5, 0.2):
        K = N = Kc = Nc = inc = 0
        for a, b in RC.NEIGHBOURS:
            common = set(ci[a]["years"]) & set(ci[b]["years"])
            if len(common) < 20:
                continue
            ea, _ = RC.events(ci[a]["idx"], ci[a]["years"], q)
            eb, _ = RC.events(ci[b]["idx"], ci[b]["years"], q)
            A = [e for e in ea if e[0].year in common]; B = [e for e in eb if e[0].year in common]
            (rare, rc), (other, oc) = ((A, a), (B, b)) if len(A) <= len(B) else ((B, b), (A, a))
            k = n = kc = nc = 0
            for s0, s1 in rare:
                hit = any((o0 - s1).days <= 1 and (s0 - o1).days <= 1 for o0, o1 in other)
                window = [s0 + timedelta(days=d) for d in range(-1, (s1 - s0).days + 2)]
                full = all(d in vd[oc] for d in window)
                k += hit; n += 1
                if full:
                    kc += hit; nc += 1
            K += k; N += n; Kc += kc; Nc += nc; inc += n - nc
            out["pairs"].setdefault(f"{a}-{b}", {})[str(q)] = {"shared": k, "events": n, "complete_window_events": nc,
                                                               "shared_complete": kc}
        out["pooled"][str(q)] = {"friend_share": K / max(N, 1), "events": N, "events_with_gap_in_other_window": inc,
                                 "share_complete_windows": Kc / max(Nc, 1), "complete_events": Nc}
    for c, r in out["countries"].items():
        t = out["threshold"][c]["0.5"]
        print(f"{c}: {r['valid_years']} valid years, missing {r['missing_days']} days ({r['missing_share']:.1%}) -> "
              f"~{r['events_possibly_hidden_q50']:.2f} q=0.5 events possibly hidden of {r['events_q50']};  "
              f"q=0.5 threshold {t['friend']:.3f} vs near-complete years {t['near_complete_years'] if t['near_complete_years'] is None else round(t['near_complete_years'], 3)} "
              f"(n={t['n_near_complete']})")
    for q, r in out["pooled"].items():
        print(f"pooled q={q}: friend share {r['friend_share']:.1%} ({r['events']} events, {r['events_with_gap_in_other_window']} "
              f"with a gap on the other side) vs complete-window share {r['share_complete_windows']:.1%} ({r['complete_events']})")
    save("quality_data_gaps.json", out)
    return out


if __name__ == "__main__":
    run()
