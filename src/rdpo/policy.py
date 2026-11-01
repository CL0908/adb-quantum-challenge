"""RDPO policy: a soft-RL (Doob) policy over Poisson sources.

Sources are visited in a fixed order k = 1..K. At step k the state s_k (current capped levels
a_i, r_i of every country) is updated by the increment x ~ q_k(· | s_k), where

    q_k(x | s) = (1 − α) q̃_k(x | s) + α p_k(x),      q̃_k(x | s) ∝ p_k(x) · exp(θ · ΔΦ_k(s, x)),

ΔΦ_k(s, x) = Φ(s ⊕_k x) − Φ(s), and log Ṽ_θ(s) = θ·Φ(s) is the learned soft value function.
Φ has three blocks:
    country features   ψ(a_i, r_i) = [1{a_i>=1}, 1{a_i>=2}, 1{a_i>=3}, 1{a_i>=4}, r_i]     (5 / country)
    edge features      χ(s_i, s_j) = [1{L_i>0}·1{L_j>0}, 1{a_i=4}·1{a_j=4}, r_i·r_j]       (3 / edge)
    target progress    h(L, L_i, L_j)                                                     (1-3)
Only the touched countries (1 or 2), their incident edges and h change at step k, so ΔΦ_k lives on
a fixed coordinate set idx_k and the logit is ΔΦ_k · θ[idx_k]: O(n + |E|) work per path.
The defensive mixture weight α bounds every likelihood ratio by 1/α per step.
"""
import numpy as np

from .problem import PortfolioProblem

NPSI, NCHI = 5, 3


def psi(a, r):
    return np.stack([a >= 1, a >= 2, a >= 3, a >= 4, r >= 1], -1).astype(float)


def chi(ai, ri, aj, rj):
    pi, pj = (ai > 0) | (ri > 0), (aj > 0) | (rj > 0)
    return np.stack([pi & pj, (ai == 4) & (aj == 4), (ri > 0) & (rj > 0)], -1).astype(float)


class RDPOPolicy:
    def __init__(self, prob: PortfolioProblem, alpha=0.05, edge_features=True, relevance=True, n_h=None):
        self.P, self.alpha, self.edge_features = prob, alpha, edge_features
        n, E = prob.n, len(prob.edges)
        self.G = n_h if n_h is not None else prob.NH[prob.target]
        self.off_edge = n * NPSI
        self.off_h = self.off_edge + (E * NCHI if edge_features else 0)
        self.d = self.off_h + self.G
        self.theta = np.zeros(self.d)
        # Target-relevant countries: the payoff of a pair target depends only on the pair, so for a source
        # touching no relevant country the exact value function is unchanged and the exact Doob policy is p
        # itself. Such sources are frozen at q = p (likelihood ratio exactly 1); features are restricted to
        # relevant countries and to edges between them.
        pair_target = prob.target in ("joint_attachment", "joint_exhaustion")
        rel = set(prob.pair) if (pair_target and relevance) else set(range(n))
        self.relevant = rel
        self.plan = []
        for src in prob.src:
            sites = list(dict.fromkeys(src["sites"]))
            fsites = [i for i in sites if i in rel]
            frozen = not fsites
            edges = [e for e, (i, j) in enumerate(prob.edges) if i in rel and j in rel and (i in fsites or j in fsites)] \
                if (edge_features and not frozen) else []
            idx = [self._loc(i) for i in fsites] + [self._edge(e) for e in edges] + \
                  ([] if frozen else [np.arange(self.off_h, self.d)])
            self.plan.append({"sites": sites, "fsites": fsites, "edges": edges, "frozen": frozen,
                              "idx": np.concatenate(idx) if idx else np.zeros(0, dtype=int)})

    def _loc(self, i):
        return np.arange(i * NPSI, (i + 1) * NPSI)

    def _edge(self, e):
        return np.arange(self.off_edge + e * NCHI, self.off_edge + (e + 1) * NCHI)

    @property
    def n_params(self):
        return self.d

    # --- one step ----------------------------------------------------------------------------
    def delta(self, k, A, R, Ltot, Lp):
        """ΔΦ_k for all candidate increments: returns (dPhi (N,V,|idx|), new site states, new L)."""
        P, src, plan = self.P, self.P.src[k], self.plan[k]
        N, V = A.shape[0], len(src["p"])
        newA = {i: np.repeat(A[:, i][:, None], V, 1) for i in plan["sites"]}
        newR = {i: np.repeat(R[:, i][:, None], V, 1) for i in plan["sites"]}
        for i, inc in src["da"]:
            newA[i] = np.minimum(newA[i] + inc[None, :], 4)
        for i, inc in src["dr"]:
            newR[i] = np.minimum(newR[i] + inc[None, :], 1)
        blocks = []
        dL = np.zeros((N, V))
        for i in plan["sites"]:
            if i in plan["fsites"]:
                blocks.append(psi(newA[i], newR[i]) - psi(A[:, i], R[:, i])[:, None, :])
            dL += P.site_loss(newA[i], newR[i]) - P.site_loss(A[:, i], R[:, i])[:, None]
        for e in plan["edges"]:
            i, j = P.edges[e]
            get = lambda d, X, s: d[s] if s in d else np.repeat(X[:, s][:, None], V, 1)
            new = chi(get(newA, A, i), get(newR, R, i), get(newA, A, j), get(newR, R, j))
            blocks.append(new - chi(A[:, i], R[:, i], A[:, j], R[:, j])[:, None, :])
        Lt_new = Ltot[:, None] + dL
        pi, pj = P.pair
        Li_new = P.site_loss(newA[pi], newR[pi]) if pi in newA else np.repeat(Lp[:, 0][:, None], V, 1)
        Lj_new = P.site_loss(newA[pj], newR[pj]) if pj in newA else np.repeat(Lp[:, 1][:, None], V, 1)
        if not plan["frozen"]:
            blocks.append(P.h(Lt_new, Li_new, Lj_new) - P.h(Ltot, Lp[:, 0], Lp[:, 1])[:, None, :])
        dPhi = np.concatenate(blocks, -1) if blocks else np.zeros((N, V, 0))
        return dPhi, newA, newR, Lt_new, Li_new, Lj_new

    def probs(self, k, dPhi):
        lp = self.P.src[k]["logp"][None, :] + dPhi @ self.theta[self.plan[k]["idx"]]
        lp -= lp.max(1, keepdims=True)
        qt = np.exp(lp); qt /= qt.sum(1, keepdims=True)
        return qt, (1 - self.alpha) * qt + self.alpha * self.P.src[k]["p"][None, :]

    # --- sampling ------------------------------------------------------------------------------
    def rollout(self, N, rng, record=False, grad=None, U=None):
        """Sample N terms. Returns dict(f, w, score, acts). If grad=(coef, mode) also accumulates
        Σ_j coef_j ∇ log q(path_j) (mode 'mix') or the CE quantities (mode 'ce') — see replay()."""
        P = self.P
        A = np.zeros((N, P.n), dtype=np.int64); R = np.zeros((N, P.n), dtype=np.int64)
        Ltot = np.zeros(N); Lp = np.zeros((N, 2)); logw = np.zeros(N)
        acts = np.zeros((N, P.K), dtype=np.int64) if record else None
        idx = np.arange(N)
        for k in range(P.K):
            dPhi, newA, newR, Lt, Li, Lj = self.delta(k, A, R, Ltot, Lp)
            _, q = self.probs(k, dPhi)
            u = rng.random((N, 1)) if U is None else U[:, k:k + 1]   # U: (N, K) uniforms (e.g. scrambled Sobol)
            v = np.minimum((q.cumsum(1) < u).sum(1), q.shape[1] - 1)
            logw += P.src[k]["logp"][v] - np.log(q[idx, v])
            for i in newA:
                A[:, i] = newA[i][idx, v]; R[:, i] = newR[i][idx, v]
            Ltot = Lt[idx, v]; Lp = np.stack([Li[idx, v], Lj[idx, v]], 1)
            if record:
                acts[:, k] = v
        f = P.payoff_from(Ltot, Lp[:, 0], Lp[:, 1])
        return {"f": f, "w": np.exp(logw), "score": P.score_from(Ltot, Lp[:, 0], Lp[:, 1]), "acts": acts}

    def replay(self, acts, coef, mode):
        """Replay recorded actions.
        mode 'ce'  -> (−Σ c_j log q̃(path_j), gradient)   [unmixed tilted policy, convex in θ]
        mode 'mix' -> Σ c_j ∇θ log q(path_j)             [mixture policy, for policy gradients]"""
        P = self.P
        N = acts.shape[0]
        A = np.zeros((N, P.n), dtype=np.int64); R = np.zeros((N, P.n), dtype=np.int64)
        Ltot = np.zeros(N); Lp = np.zeros((N, 2)); idx = np.arange(N)
        g = np.zeros(self.d); val = 0.0
        for k in range(P.K):
            v = acts[:, k]
            dPhi, newA, newR, Lt, Li, Lj = self.delta(k, A, R, Ltot, Lp)
            qt, q = self.probs(k, dPhi)
            cols = self.plan[k]["idx"]
            mean_dphi = np.einsum("nv,nvd->nd", qt, dPhi)
            if mode == "ce":
                lp = P.src[k]["logp"][None, :] + dPhi @ self.theta[cols]
                m = lp.max(1, keepdims=True)
                val -= coef @ (lp[idx, v] - (m[:, 0] + np.log(np.exp(lp - m).sum(1))))
                fac = coef
            else:
                fac = coef * (1 - self.alpha) * qt[idx, v] / q[idx, v]
            np.add.at(g, cols, fac @ (dPhi[idx, v] - mean_dphi))
            for i in newA:
                A[:, i] = newA[i][idx, v]; R[:, i] = newR[i][idx, v]
            Ltot = Lt[idx, v]; Lp = np.stack([Li[idx, v], Lj[idx, v]], 1)
        return (val, -g) if mode == "ce" else g
