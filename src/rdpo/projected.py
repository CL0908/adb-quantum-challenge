"""Projected Doob sampler: RDPO with an exact tabular value function on a small set of countries.

For a target f that depends only on the terminal states of a subset R of countries (|R| <= 3; for example the
joint exhaustion of a pair), the exact value function of Proposition 3 depends only on the R-projection of the
state, s_R ∈ {0..9}^|R|:

    V_k(s_R) = E[f | state of R after sources 1..k-1 is s_R],     V_{K+1} = f,
    V_k(s_R) = Σ_x p_k(x) V_{k+1}(s_R ⊕ x_R),

where x_R is the part of source k's increment that lands on R (sources touching no country in R leave V
unchanged). This is RDPO with the learned log-linear soft value replaced by an exact table of 10^|R| entries per
step, computed by backward induction in O(K · 10^|R| · 25). Its Boltzmann policy

    q_k(x | s) = p_k(x) V_{k+1}(s_R ⊕ x_R) / V_k(s_R)

is the zero-variance Doob sampler: every path has w·f = μ exactly. Because sources outside R keep their true
law, every sampled path is an exact draw from the portfolio *conditional on the event* (for an indicator f), so
the sampler also produces stress scenarios: e.g. the distribution of the pool loss given that both bonds of a
pair are exhausted.
"""
import itertools

import numpy as np

from .problem import PortfolioProblem

A_OF, R_OF = np.arange(10) // 2, np.arange(10) % 2


class ProjectedDoob:
    def __init__(self, prob: PortfolioProblem, sites=None):
        P = self.P = prob
        self.R = list(sites if sites is not None else prob.pair)
        r = len(self.R)
        assert 1 <= r <= 3, "projected tables are 10^|R| per step; use |R| <= 3"
        self.shape = (10,) * r
        # next-code table on R for every source: nxt[k][v] -> tuple of per-site code maps (10,) for each R site
        self.touch = []
        for src in P.src:
            V = len(src["p"])
            maps = np.zeros((V, r, 10), dtype=np.int64)
            for v in range(V):
                for t, site in enumerate(self.R):
                    da = sum(inc[v] for s, inc in src["da"] if s == site)
                    dr = sum(inc[v] for s, inc in src["dr"] if s == site)
                    maps[v, t] = 2 * np.minimum(A_OF + da, 4) + np.minimum(R_OF + dr, 1)
            touches = any(site in src["sites"] for site in self.R)
            self.touch.append(maps if touches else None)
        # terminal payoff on R codes
        L = P.site_loss(A_OF, R_OF)
        grids = np.meshgrid(*([np.arange(10)] * r), indexing="ij")
        losses = [L[g] for g in grids]
        if P.target == "joint_exhaustion":
            f = np.all([l == P.F for l in losses], axis=0).astype(float)
        elif P.target == "joint_attachment":
            f = np.all([l > 0 for l in losses], axis=0).astype(float)
        else:
            raise ValueError("projected sampler supports pair/subset indicator targets")
        # backward induction
        self.V = [None] * (P.K + 1)
        self.V[P.K] = f
        for k in range(P.K - 1, -1, -1):
            maps = self.touch[k]
            if maps is None:
                self.V[k] = self.V[k + 1]
                continue
            nxt = np.zeros(self.shape)
            for v, pv in enumerate(P.src[k]["p"]):
                if pv:
                    nxt += pv * self.V[k + 1][np.ix_(*[maps[v, t] for t in range(r)])]
            self.V[k] = nxt
        self.mu = float(self.V[0][(0,) * r])

    def rollout(self, N, rng, full_state=True):
        """Sample N terms under the projected Doob policy. Returns f, w, the pool loss and the R-codes.
        With full_state, all other sources are sampled at their true law, so pool-level quantities are exact
        conditional draws."""
        P, r = self.P, len(self.R)
        sR = np.zeros((N, r), dtype=np.int64)
        logw = np.zeros(N)
        A = np.zeros((N, P.n), dtype=np.int64); Rr = np.zeros((N, P.n), dtype=np.int64)
        idx = np.arange(N)
        for k, src in enumerate(P.src):
            p = src["p"]
            maps = self.touch[k]
            if maps is None:
                v = rng.choice(len(p), size=N, p=p) if full_state else None
            else:
                nxt = np.stack([self.V[k + 1][tuple(maps[vv, t][sR[:, t]] for t in range(r))] for vv in range(len(p))], 1)
                q = p[None, :] * nxt
                q /= np.maximum(q.sum(1, keepdims=True), 1e-300)
                u = rng.random((N, 1))
                v = np.minimum((q.cumsum(1) < u).sum(1), len(p) - 1)
                logw += np.log(np.maximum(p[v], 1e-300)) - np.log(np.maximum(q[idx, v], 1e-300))
                sR = np.stack([maps[v, t, sR[:, t]] for t in range(r)], 1)
            if full_state and v is not None:
                for site, inc in src["da"]:
                    A[:, site] = np.minimum(A[:, site] + inc[v], 4)
                for site, inc in src["dr"]:
                    Rr[:, site] = np.minimum(Rr[:, site] + inc[v], 1)
        L = P.site_loss(A_OF[sR], R_OF[sR])
        if P.target == "joint_exhaustion":
            f = np.all(L == P.F, axis=1).astype(float)
        else:
            f = np.all(L > 0, axis=1).astype(float)
        pool = P.site_loss(A, Rr).sum(1) * 2.5e6 if full_state else None
        return {"f": f, "w": np.exp(logw), "pool_loss_usd": pool, "codes": sR}


def demo():
    """Zero-variance check and conditional stress scenarios on the ADB pair and the calibrated CAREC pool."""
    import json
    import time
    from pathlib import Path
    from .problem import adb_model
    out = {}
    cases = [("ADB benchmark (2 countries)", PortfolioProblem(adb_model("adb-base"), "joint_exhaustion", threshold=80e6), None)]
    try:
        from src.realdata.carec_pool import calibrate
        cal = calibrate()
        Pc = PortfolioProblem(cal["model"], "joint_exhaustion", pair=cal["pair"])
        cases.append(("CAREC pool (10 countries)", Pc, cal))
    except Exception as ex:  # data cache missing
        print("CAREC skipped:", ex)
    for name, P, cal in cases:
        t0 = time.perf_counter(); d = ProjectedDoob(P); o = d.rollout(20000, np.random.default_rng(7)); sec = time.perf_counter() - t0
        g = o["w"] * o["f"]
        rec = {"mu": d.mu, "exact": P.exact_pair(), "paths": 20000, "seconds": sec,
               "rel_sd_per_path": float(g.std() / g.mean()), "all_paths_hit": bool(np.all(o["f"] == 1))}
        pl, w = o["pool_loss_usd"], o["w"] / o["w"].sum()
        rec["conditional_pool_loss_mean_musd"] = float(pl @ w / 1e6)
        rec["conditional_pool_loss_p95_musd"] = float(np.quantile(pl, 0.95) / 1e6)
        if cal is not None:
            others = [cal["order"].index(c) for c in ("UZ", "KZ")]
            # chance that a third country also pays, given the pair is exhausted (exact conditional draws)
            rec["P(pool > 400m | pair exhausted)"] = float(np.mean(pl > 400e6))
            t3 = ProjectedDoob(P, sites=[cal["order"].index(c) for c in ("KG", "TJ", "UZ")])
            rec["triple_exhaustion_KG_TJ_UZ"] = t3.mu
        out[name] = rec
        print(name, rec)
    Path("results/projected_doob.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    demo()
