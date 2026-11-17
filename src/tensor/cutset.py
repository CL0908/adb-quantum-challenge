"""Cutset conditioning (Rao-Blackwellised sampling) for portfolio targets on dense dependence graphs.

The shared sources (edges) are split into a cutset D and the remainder R, with R of small pathwidth. The terminal
state is the capped sum of independent source increments (capped addition commutes), so the increments of the D
sources can be applied first, as initial (a, r) offsets of the countries they touch. With
        g(w_D) = E[ f | w_D ]
computed *exactly* by the frontier elimination of src/tensor/junction.py on the R graph (initial vector of country i =
row `code(offset_i)` of its solo kernel), we have  mu = E_{w_D}[ g(w_D) ]  and

  * sampled:     draw w_D from an importance law q_D (exponential twist of the D sources, optionally mixed with p_D),
                 estimate mu by the mean of  (p_D / q_D)(w_D) g(w_D).  Unbiased; every contraction is cached by the
                 per-country offset codes (g depends on w_D only through them).
  * enumerated:  sum p_D(w_D) g(w_D) over the most probable D-configurations (best-first), with a rigorous bound on the
                 neglected part (neglected probability, and a Chernoff-type tilted bound, whichever is smaller).

One contraction yields both targets (exceedance and stop-loss at the problem threshold T). Contractions reuse the
zero-offset frontier states of the sweep up to the first step that touches an offset country (prefix sharing).
The solver does not modify junction.py: kernels and orderings are imported from it.
"""
import heapq
import math
import time

import numpy as np
from scipy.stats import t as _tdist

from src.rdpo.problem import PortfolioProblem, UNIT
from src.tensor.junction import A_OF, R_OF, best_order, source_kernel

CODE_A, CODE_R = A_OF, R_OF


def split_edges(edges, keep):
    """Split an edge list into (R, D): R = edges for which keep(i, j) is true (low-width part), D = the rest."""
    R = [tuple(sorted(e)) for e in edges if keep(*sorted(e))]
    Rs = set(R)
    D = [tuple(sorted(e)) for e in edges if tuple(sorted(e)) not in Rs]
    return R, D


class CutsetSolver:
    def __init__(self, prob: PortfolioProblem, d_edges, order=None):
        self.P = prob
        n = self.n = prob.n
        self.d_edges = {tuple(sorted(e)) for e in d_edges}
        self.r_edges = [tuple(sorted(e)) for e in prob.edges if tuple(sorted(e)) not in self.d_edges]
        self.Tc = int(np.floor(prob.T)) + 1
        self.loss = prob.site_loss(A_OF, R_OF).astype(int)
        self.D_idx, self.R_idx = [], []
        self.solo = {i: np.eye(10) for i in range(n)}      # product of the solo kernels as 10x10 matrix
        self.pair_src = {}
        for k, s in enumerate(prob.src):
            if len(set(s["sites"])) == 2 and tuple(sorted(s["sites"])) in self.d_edges:
                self.D_idx.append(k)
            else:
                self.R_idx.append(k)
                sites, K = source_kernel(s)
                if len(sites) == 1:
                    self.solo[sites[0]] = self.solo[sites[0]] @ K
                else:
                    self.pair_src.setdefault(tuple(sorted(sites)), []).append((sites, K))
        for e, lst in list(self.pair_src.items()):      # merge the (commuting) kernels of one edge into one 100x100
            sites0 = lst[0][0]
            Mt = np.eye(100)
            for sites, K in lst:
                Kk = K if sites[0] == sites0[0] else K.transpose(1, 0, 3, 2)
                Mt = Mt @ Kk.reshape(100, 100)
            self.pair_src[e] = [(sites0, Mt.reshape(10, 10, 10, 10))]
        self.order = list(order) if order is not None else best_order(n, self.r_edges)
        pos = {v: t for t, v in enumerate(self.order)}
        self.nbr = {i: [] for i in range(n)}
        for i, j in self.r_edges:
            self.nbr[i].append(j); self.nbr[j].append(i)
        self.pos = pos
        # intro step of each country (first step at which its initial vector is used)
        self.intro = {}
        act = set()
        peak = 0
        for t, v in enumerate(self.order):
            for c in [v] + [u for u in sorted(self.nbr[v], key=lambda x: pos[x]) if pos[u] > pos[v]]:
                if c not in act:
                    act.add(c); self.intro[c] = t
            peak = max(peak, len(act)); act.discard(v)
        self.width = peak - 1
        self.cache_states = None
        self.n_contractions = 0
        self.t_contract = 0.0
        self.g_cache = {}
        # D source tables
        self.KD = len(self.D_idx)
        self.pD = [prob.src[k]["p"] for k in self.D_idx]
        self.ellD = []
        for k in self.D_idx:
            s = prob.src[k]
            e = np.zeros(len(s["p"]))
            for _, inc in s["da"]:
                e = e + prob.a_u * inc
            for _, inc in s["dr"]:
                e = e + prob.r_u * inc
            self.ellD.append(e)
        # per-source offset tables (V_k, n): a and r increments per site
        self.offA, self.offR = [], []
        for k in self.D_idx:
            s = prob.src[k]
            V = len(s["p"])
            a = np.zeros((V, n), np.int16); r = np.zeros((V, n), np.int16)
            for site, inc in s["da"]:
                a[:, site] += np.asarray(inc, np.int16)
            for site, inc in s["dr"]:
                r[:, site] += np.asarray(inc, np.int16)
            self.offA.append(a); self.offR.append(r)
        # uncapped loss ell for the R sources (for Chernoff bounds)
        self.ellR = []
        for k in self.R_idx:
            s = prob.src[k]
            e = np.zeros(len(s["p"]))
            for _, inc in s["da"]:
                e = e + prob.a_u * inc
            for _, inc in s["dr"]:
                e = e + prob.r_u * inc
            self.ellR.append(e)

    # ---- one contraction ------------------------------------------------------------------------------
    def _init_vec(self, c, code0):
        return self.solo[c][code0]

    def _shift_mats(self):
        """Per-state finalisation matrices on the combined vector Z = [mass bins 0..Tc-1, overflow mass, overflow
        first moment]: adding loss d moves mass, spills into the overflow and updates the overflow moment."""
        Tc = self.Tc
        mats = []
        for d in self.loss:
            d = int(d)
            M = np.zeros((Tc + 2, Tc + 2))
            for l in range(Tc):
                if l + d < Tc:
                    M[l, l + d] = 1.0
                else:
                    M[l, Tc] = 1.0
                    M[l, Tc + 1] = l + d - Tc
            M[Tc, Tc] = 1.0; M[Tc, Tc + 1] = d
            M[Tc + 1, Tc + 1] = 1.0
            mats.append(M)
        self._M = np.stack(mats)                        # (10, Tc+2, Tc+2)

    def _step(self, t, Z, act, codes):
        v = self.order[t]
        pos = self.pos
        if not hasattr(self, "_M"):
            self._shift_mats()

        def introduce(c, Z):
            vec = self._init_vec(c, int(codes[c])).reshape((10, 1))
            return np.expand_dims(Z, -2) * vec

        if v not in act:
            Z = introduce(v, Z); act.append(v)
        for u in sorted(self.nbr[v], key=lambda x: pos[x]):
            if pos[u] < pos[v]:
                continue
            if u not in act:
                Z = introduce(u, Z); act.append(u)
            a, b = act.index(v), act.index(u)
            for sites, K in self.pair_src.get(tuple(sorted((u, v))), []):
                Kv = K if sites[0] == v else K.transpose(1, 0, 3, 2)
                Z = np.moveaxis(np.tensordot(Z, Kv, axes=([a, b], [0, 1])), [-2, -1], [a, b])
        a = act.index(v)
        Zm = np.moveaxis(Z, a, 0)                       # (10, ..., Tc+2)
        out = np.matmul(Zm.reshape(10, -1, self.Tc + 2), self._M).sum(0)
        act.remove(v)
        return out.reshape(Zm.shape[1:])

    def _zero_states(self):
        """Zero-offset sweep, storing the state before every step."""
        Z = np.zeros(self.Tc + 2); Z[0] = 1.0
        act = []
        zeros = np.zeros(self.n, int)
        st = []
        for t in range(len(self.order)):
            st.append((Z, list(act)))
            Z = self._step(t, Z, act, zeros)
        self.cache_states = st
        self.g0 = (float(Z[self.Tc]), float(Z[self.Tc + 1]))
        return st

    def contract(self, codes):
        """Exact (exceedance, stop-loss in USD) given initial offset codes (n,) ints. Prefix-shared sweep."""
        if self.cache_states is None:
            self._zero_states()
        t0 = time.perf_counter()
        nz = np.nonzero(codes)[0]
        if len(nz) == 0:
            O, M = self.g0
        else:
            t_start = min(self.intro[int(c)] for c in nz)
            Z, act = self.cache_states[t_start]
            act = list(act)
            for t in range(t_start, len(self.order)):
                Z = self._step(t, Z, act, codes)
            O, M = float(Z[self.Tc]), float(Z[self.Tc + 1])
        self.n_contractions += 1
        self.t_contract += time.perf_counter() - t0
        return O, (M + O * (self.Tc - self.P.T)) * UNIT

    def g(self, codes):
        key = np.asarray(codes, np.uint8).tobytes()
        v = self.g_cache.get(key)
        if v is None:
            v = self.contract(codes)
            self.g_cache[key] = v
        return v

    # ---- helper: offset codes of D configurations --------------------------------------------------------
    def codes_of(self, vals):
        """vals (N, KD) int -> offset codes (N, n) (capped)."""
        N = vals.shape[0]
        A = np.zeros((N, self.n), np.int16); R = np.zeros((N, self.n), np.int16)
        for j in range(self.KD):
            A += self.offA[j][vals[:, j]]; R += self.offR[j][vals[:, j]]
        np.minimum(A, 4, out=A); np.minimum(R, 1, out=R)
        return (2 * A + R).astype(np.uint8)

    # ---- sampling law ---------------------------------------------------------------------------------------
    def twist_tables(self, eta):
        q = []
        for p, e in zip(self.pD, self.ellD):
            lg = np.log(np.maximum(p, 1e-300)) + eta * e
            lg -= lg.max(); w = np.exp(lg); q.append(w / w.sum())
        return q

    def sample_configs(self, N, rng, eta, delta):
        """Draw N D-configurations from (1-delta) q_eta + delta p. Returns vals (N,KD) and weights p/q_mix."""
        q = self.twist_tables(eta)
        vals = np.empty((N, self.KD), np.int16)
        lw = np.zeros(N)
        use_p = rng.random(N) < delta
        for j in range(self.KD):
            U = rng.random(N)
            cq = np.cumsum(q[j]); cq[-1] = 1.0
            cp = np.cumsum(self.pD[j]); cp[-1] = 1.0
            v = np.where(use_p, np.searchsorted(cp, U), np.searchsorted(cq, U))
            vals[:, j] = v
            lw += np.log(np.maximum(self.pD[j][v], 1e-300)) - np.log(np.maximum(q[j][v], 1e-300))
        # log of p/q for each draw; mixture law: q_mix/p = (1-delta) q/p + delta
        ratio = (1 - delta) * np.exp(-lw) + delta
        return vals, 1.0 / ratio

    # ---- estimators ---------------------------------------------------------------------------------------------
    def sampled(self, rng, seconds=None, n_samples=None, eta=None, delta=0.1, eta_scale=1.0, chunk=512, t_start=None):
        """Importance-sampled cutset estimator of E[1{L>T}] and E[(L-T)+] (USD). Budget: wall-clock `seconds`
        (everything from t_start charged, contractions included) or `n_samples`."""
        t0 = time.perf_counter() if t_start is None else t_start
        if eta is None:
            from src.rdpo.train import _best_twist_sources
            eta = eta_scale * _best_twist_sources(self.P)
        zs = [[], []]
        ws = []
        n_done = 0
        if self.cache_states is None:
            self._zero_states()
        # chunks small enough that the contractions of one chunk cost <= ~5% of the budget (full sweep time is an
        # upper bound for one contraction); the chunk size depends on timings only, never on the g values
        if seconds is not None:
            spc0 = max(time.perf_counter() - t0, 1e-4)
            chunk = int(np.clip(0.05 * seconds / spc0, 4, chunk))
        while True:
            if n_samples is not None:
                if n_done >= n_samples:
                    break
                m = min(chunk, n_samples - n_done)
            else:
                rem = seconds - (time.perf_counter() - t0)
                if rem <= 0.02 * seconds:
                    break
                m = chunk
            vals, w = self.sample_configs(m, rng, eta, delta)
            codes = self.codes_of(vals)
            uc, inv = np.unique(codes, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            gv = np.empty((len(uc), 2))
            for r in range(len(uc)):
                key = uc[r].tobytes()
                hit = self.g_cache.get(key)
                if hit is None:
                    if seconds is not None:
                        # stop before blowing the budget: only start a contraction if time remains
                        if time.perf_counter() - t0 > seconds:
                            gv[r] = np.nan
                            continue
                    hit = self.contract(uc[r].astype(int)); self.g_cache[key] = hit
                gv[r] = hit
            ok = ~np.isnan(gv[inv, 0])
            # configurations whose contraction was skipped (budget exhausted) are dropped as the *last* partial chunk,
            # keeping a prefix-stopping rule that is independent of the g values
            if not ok.all():
                first_bad = int(np.argmin(ok))
                inv, w = inv[:first_bad], w[:first_bad]
                if len(w) == 0:
                    break
                zs[0].append(w * gv[inv, 0]); zs[1].append(w * gv[inv, 1]); ws.append(w); n_done += len(w)
                break
            zs[0].append(w * gv[inv, 0]); zs[1].append(w * gv[inv, 1]); ws.append(w); n_done += m
        out = {"n_samples": n_done, "seconds": time.perf_counter() - t0, "n_contractions": self.n_contractions,
               "distinct_cached": len(self.g_cache), "sec_per_contraction": self.t_contract / max(self.n_contractions, 1),
               "eta": eta, "delta": delta, "KD": self.KD, "width_R": self.width}
        if not ws:
            out["exceedance"] = out["stop_loss"] = {"estimate": float("nan"), "se": float("nan"), "ci": (float("nan"),) * 2,
                                                    "var_per_sample": float("nan"), "rel_var_per_sample": float("nan")}
            out["ess_frac"] = float("nan")
            return out
        w = np.concatenate(ws)
        out["ess_frac"] = float(w.sum() ** 2 / (w ** 2).sum() / len(w))
        for name, z in zip(("exceedance", "stop_loss"), zs):
            z = np.concatenate(z)
            N = len(z)
            mean = float(z.mean()); var = float(z.var(ddof=1)); se = math.sqrt(var / N)
            zq = float(_tdist.ppf(0.975, max(N - 1, 1)))
            out[name] = {"estimate": mean, "se": se, "ci": (mean - zq * se, mean + zq * se),
                         "var_per_sample": var, "rel_var_per_sample": var / mean ** 2 if mean > 0 else float("inf")}
        return out

    # ---- deterministic enumeration ---------------------------------------------------------------------------
    def enumerate_best_first(self, max_configs=200000, max_seconds=None, eta_order=0.0, bound_etas=None,
                             rel_target=None, check_every=2000, target="exceedance", log_every=None):
        """Sum p_D(w) g(w) over D-configurations in order of decreasing probability (or of decreasing tilted
        probability p e^{eta_order * ell} if eta_order > 0), with rigorous remainder bounds.

        Bounds on the neglected part  sum_{w not enumerated} p(w) g(w):
          (a) neglected probability mass  (exceedance; for stop-loss times the sup, not available -> use (b)),
          (b) Chernoff: for every eta > 0, 1{L > T} <= exp(eta (S - T)) and (L-T)^+ <= exp(eta (S - T))/(e eta) with
              S = uncapped total increment >= L, hence
              remainder <= c(eta) e^{-eta T} prod_{k in R} M_k(eta) * [ prod_{k in D} M_k(eta) - sum_E p e^{eta ell} ].
        The best (smallest) valid bound over the eta grid is reported.
        """
        t0 = time.perf_counter()
        T = self.P.T
        if bound_etas is None:
            bound_etas = np.linspace(0.05, 1.5, 30)
        bound_etas = np.asarray(bound_etas)
        # per-source ranking by (tilted) probability
        orders, lp = [], []
        for p, e in zip(self.pD, self.ellD):
            sc = np.log(np.maximum(p, 1e-300)) + eta_order * e
            sc = np.where(p > 0, sc, -np.inf)
            od = np.argsort(-sc, kind="stable"); od = od[np.isfinite(sc[od])]
            orders.append(od); lp.append(sc[od])
        KD = self.KD
        perm = sorted(range(KD), key=lambda j: (lp[j][0] - lp[j][1]) if len(lp[j]) > 1 else np.inf)
        orders = [orders[j] for j in perm]; lp = [lp[j] for j in perm]
        nopen = sum(1 for l in lp if len(l) > 1)           # sources with at least two support values
        permarr = np.array(perm)
        # per-source moment tables for the bounds
        logM_D = np.array([[np.log(np.sum(p * np.exp(eta * e))) for eta in bound_etas] for p, e in zip(self.pD, self.ellD)]).sum(0)
        logM_R = np.array([[np.log(np.sum(self.P.src[k]["p"] * np.exp(eta * e))) for eta in bound_etas]
                           for k, e in zip(self.R_idx, self.ellR)]).sum(0)
        # Best-first over rank vectors (zero rank = most probable value of each source)
        start = (0,) * KD
        score0 = float(sum(l[0] for l in lp))
        heap = [(-score0, start, -1)]            # (-score, ranks, last nonzero index)
        # canonical scheme: children increase the last nonzero rank or open a later source at rank 1
        p_sum, tot_sum = 0.0, 0.0
        tilt_sum = np.zeros(len(bound_etas))
        n_enum = 0
        n_unique_codes0 = len(self.g_cache)
        trace = []
        acc_exc, acc_sl = [], []
        pvals_buf = []
        best_bounds = None
        while heap:
            sc, ranks, last = heapq.heappop(heap)
            vals = np.zeros(KD, int)
            vals[permarr] = [orders[j][r] for j, r in enumerate(ranks)]
            logp = float(sum(np.log(max(self.pD[j][vals[j]], 1e-300)) for j in range(KD)))
            pw = math.exp(logp)
            ell = float(sum(self.ellD[j][vals[j]] for j in range(KD)))
            codes = self.codes_of(vals[None, :].astype(np.int16))[0]
            ge, gs = self.g(codes)
            acc_exc.append(pw * ge); acc_sl.append(pw * gs)
            p_sum += pw
            tilt_sum += pw * np.exp(bound_etas * ell)
            n_enum += 1
            # children (sources sorted by opening cost): raise the last nonzero rank / open the next source /
            # shift a freshly opened rank-1 source to the next one; every state has a unique parent
            if last >= 0 and ranks[last] + 1 < len(orders[last]):
                r = list(ranks); r[last] += 1
                heapq.heappush(heap, (sc - (lp[last][r[last]] - lp[last][r[last] - 1]), tuple(r), last))
            if last + 1 < nopen:
                j = last + 1
                r = list(ranks); r[j] = 1
                heapq.heappush(heap, (sc - (lp[j][1] - lp[j][0]), tuple(r), j))
                if last >= 0 and ranks[last] == 1:
                    r = list(ranks); r[last] = 0; r[j] = 1
                    heapq.heappush(heap, (sc - (lp[j][1] - lp[j][0]) + (lp[last][1] - lp[last][0]), tuple(r), j))
            stop = n_enum >= max_configs or (max_seconds is not None and time.perf_counter() - t0 > max_seconds)
            if stop or n_enum % check_every == 0 or not heap:
                est_e, est_s = math.fsum(acc_exc), math.fsum(acc_sl)
                bd = self._remainder_bounds(p_sum, tilt_sum, logM_D, logM_R, bound_etas, T)
                trace.append({"n": n_enum, "mass_enumerated": p_sum, "exceedance": est_e, "stop_loss": est_s,
                              "rem_exc": bd["exceedance"], "rem_sl": bd["stop_loss"], "seconds": time.perf_counter() - t0,
                              "distinct": len(self.g_cache) - n_unique_codes0})
                if rel_target is not None:
                    rem, est = (bd["exceedance"], est_e) if target == "exceedance" else (bd["stop_loss"], est_s)
                    if est > 0 and rem <= rel_target * est:
                        stop = True
                if stop or not heap:
                    break
        est_e, est_s = math.fsum(acc_exc), math.fsum(acc_sl)
        bd = self._remainder_bounds(p_sum, tilt_sum, logM_D, logM_R, bound_etas, T)
        return {"n_enumerated": n_enum, "mass_enumerated": p_sum, "exceedance_lower": est_e, "stop_loss_lower": est_s,
                "remainder_bound_exceedance": bd["exceedance"], "remainder_bound_stop_loss": bd["stop_loss"],
                "bound_kind_exceedance": bd["kind_exc"], "seconds": time.perf_counter() - t0,
                "distinct_contractions": len(self.g_cache) - n_unique_codes0, "trace": trace,
                "eta_order": eta_order}

    def _remainder_bounds(self, p_sum, tilt_sum, logM_D, logM_R, etas, T):
        neglected = max(1.0 - p_sum, 0.0)
        # tilted neglected mass: prod_D M(eta) - sum_E p e^{eta ell}, clamp at 0
        tilt_neg = np.maximum(np.exp(logM_D) - tilt_sum, 0.0)
        # rounding guard: the difference is only trusted down to ~1e-12 of prod_D M
        tilt_neg = np.maximum(tilt_neg, 1e-12 * np.exp(logM_D))
        with np.errstate(over="ignore", invalid="ignore"):
            ch_e = np.exp(-etas * T + logM_R) * tilt_neg
            ch_s = np.exp(-etas * T + logM_R) * tilt_neg / (math.e * etas) * UNIT
        be, bs = float(ch_e.min()), float(ch_s.min())
        kind = "chernoff"
        if neglected < be:
            be, kind = neglected, "mass"
        return {"exceedance": be, "stop_loss": bs, "kind_exc": kind}
