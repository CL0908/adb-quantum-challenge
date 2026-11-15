"""RDPO v2 engine: table-driven policy (WP1) with optional flexible target-progress features (WP3).

Every country state is coded as s = 2a + r ∈ {0..9}. All features are functions of these codes or of
the target losses, so for a fixed θ every logit contribution is a *table lookup*:

    country block   ψ(s)·θ_i            -> table (n, 10)
    edge block      χ(s_i, s_j)·θ_e      -> table (|E|, 10, 10)
    progress block  h(L) or h(L_i, L_j)  -> table over integer losses

A rollout step therefore costs a handful of integer gathers on (N, V) arrays instead of building the
(N, V, d) feature tensor. Training (replay) still needs the features themselves; they come from the
same tables. With value="linear" the policy is *identical* to RDPOPolicy (tested); value="spline"
replaces the hand-made progress features by a piecewise-linear (hat-function) basis — a one-hidden-
layer ReLU network in the loss with fixed knots — which keeps the cross-entropy fit concave.
"""
import numpy as np

from .policy import RDPOPolicy, psi, chi, NPSI, NCHI
from .problem import PortfolioProblem

CODES = np.arange(10)
A_OF, R_OF = CODES // 2, CODES % 2
PSI = psi(A_OF, R_OF)                                                   # (10, 5)
CHI = chi(A_OF[:, None], R_OF[:, None], A_OF[None, :], R_OF[None, :])   # (10, 10, 3)


def hat_basis(x, knots):
    """Piecewise-linear hat functions on the given knots (partition of unity), shape (..., len(knots))."""
    x = np.clip(x, knots[0], knots[-1])
    out = np.zeros(x.shape + (len(knots),))
    j = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, len(knots) - 2)
    t = (x - knots[j]) / (knots[j + 1] - knots[j])
    np.put_along_axis(out, j[..., None], (1 - t)[..., None], -1)
    np.put_along_axis(out, (j + 1)[..., None], t[..., None], -1)
    return out


class FastPolicy(RDPOPolicy):
    def __init__(self, prob: PortfolioProblem, alpha=0.05, edge_features=True, relevance=True,
                 value="linear", n_knots=10, stages=1):
        self.value = value
        P = prob
        self.pair_target = P.target in ("joint_attachment", "joint_exhaustion")
        # progress-feature table: over Ltot (0..Lmax) or over (L_i, L_j) (0..F)^2
        if self.pair_target:
            Li, Lj = np.meshgrid(np.arange(P.F + 1), np.arange(P.F + 1), indexing="ij")
            H = P.h(np.zeros_like(Li), Li, Lj)
            if value == "spline":
                kn = np.linspace(0, P.F, 5)
                m = np.minimum(Li, Lj).astype(float)
                H = np.concatenate([hat_basis(m, kn), (Li + Lj)[..., None] / (2 * P.F), H[..., -1:]], -1)
        else:
            L = np.arange(P.Lmax + 1)
            H = P.h(L, np.zeros_like(L), np.zeros_like(L))
            if value == "spline":
                lo = 0.0 if P.target == "expected_loss" else 0.25 * P.T
                hi = P.Lmax if P.target == "expected_loss" else min(P.T, P.Lmax)
                kn = np.unique(np.r_[0.0, np.linspace(lo, hi, n_knots)])
                extra = [] if P.target == "expected_loss" else [(L > P.T).astype(float),
                                                                np.maximum(L - P.T, 0) / max(P.Lmax - P.T, 1)]
                H = np.concatenate([hat_basis(L.astype(float), kn)] + [e[:, None] for e in extra], -1)
        self.Htab = H.astype(float)
        super().__init__(prob, alpha, edge_features, relevance, n_h=self.Htab.shape[-1])
        self.loss_tab = (P.a_u * A_OF + P.r_u * R_OF).astype(np.int64)            # (10,)
        # per source: for each touched site, next-code table (10, V)
        self.next = []
        for src in P.src:
            V = len(src["p"]); tabs = {}
            for i in dict.fromkeys(src["sites"]):
                da = [inc for s, inc in src["da"] if s == i]
                dr = [inc for s, inc in src["dr"] if s == i]
                a2 = A_OF[:, None] + (da[0][None, :] if da else 0)
                r2 = R_OF[:, None] + (dr[0][None, :] if dr else 0)
                tabs[i] = (2 * np.minimum(a2, 4) + np.minimum(r2, 1)) * np.ones((1, V), dtype=np.int64)
            self.next.append(tabs)
        # stage-dependent progress parameters: the exact value function V_k depends on how many sources remain,
        # so the progress block θ_h gets one copy per stage (contiguous blocks of source steps)
        self.stages = stages
        self.blk = np.minimum((np.arange(P.K) * stages) // max(P.K, 1), stages - 1)
        if stages > 1:
            G = self.G
            self.d = self.off_h + G * stages
            self.theta = np.zeros(self.d)
            for k, pl in enumerate(self.plan):
                if not pl["frozen"]:
                    pl["idx"] = np.concatenate([pl["idx"][:-G], self.off_h + self.blk[k] * G + np.arange(G)])

    # ---- progress features ----------------------------------------------------------------
    def _H(self, Lt, Li, Lj):
        return self.Htab[Li, Lj] if self.pair_target else self.Htab[Lt]

    # ---- features for training (same contract as RDPOPolicy.delta) ----------------------------
    def delta_codes(self, k, S, Lt, Li, Lj):
        P, plan = self.P, self.plan[k]
        newS = {i: self.next[k][i][S[:, i]] for i in plan["sites"]}          # (N, V) codes
        dL = sum(self.loss_tab[newS[i]] - self.loss_tab[S[:, i]][:, None] for i in plan["sites"])
        blocks = [PSI[newS[i]] - PSI[S[:, i]][:, None, :] for i in plan["fsites"]]
        for e in plan["edges"]:
            i, j = P.edges[e]
            si = newS[i] if i in newS else np.broadcast_to(S[:, i][:, None], dL.shape)
            sj = newS[j] if j in newS else np.broadcast_to(S[:, j][:, None], dL.shape)
            blocks.append(CHI[si, sj] - CHI[S[:, i], S[:, j]][:, None, :])
        Lt_new = Lt[:, None] + dL
        pi, pj = P.pair
        Li_new = self.loss_tab[newS[pi]] if pi in newS else np.broadcast_to(Li[:, None], dL.shape)
        Lj_new = self.loss_tab[newS[pj]] if pj in newS else np.broadcast_to(Lj[:, None], dL.shape)
        if not plan["frozen"]:
            blocks.append(self._H(Lt_new, Li_new, Lj_new) - self._H(Lt, Li, Lj)[:, None, :])
        dPhi = np.concatenate(blocks, -1) if blocks else np.zeros(dL.shape + (0,))
        return dPhi, newS, Lt_new, Li_new, Lj_new

    # ---- logits by table lookup (rollout) ------------------------------------------------------
    def _tables(self):
        P, th = self.P, self.theta
        psi_t = np.stack([PSI @ th[i * NPSI:(i + 1) * NPSI] for i in range(P.n)]) if P.n else None
        chi_t = [CHI @ th[self.off_edge + e * NCHI: self.off_edge + (e + 1) * NCHI] for e in range(len(P.edges))] \
            if self.edge_features else []
        G = self.G
        h_t = [self.Htab @ th[self.off_h + b * G: self.off_h + (b + 1) * G] for b in range(self.stages)]
        return psi_t, chi_t, h_t

    def _init_state(self, N):
        return (np.zeros((N, self.P.n), dtype=np.int64), np.zeros(N, dtype=np.int64),
                np.zeros(N, dtype=np.int64), np.zeros(N, dtype=np.int64))

    def rollout(self, N, rng, record=False, grad=None, U=None):
        P = self.P
        psi_t, chi_t, h_t = self._tables()
        S, Lt, Li, Lj = self._init_state(N)
        logw = np.zeros(N); idx = np.arange(N)
        acts = np.zeros((N, P.K), dtype=np.int64) if record else None
        pi, pj = P.pair
        a = self.alpha
        for k in range(P.K):
            plan, src = self.plan[k], P.src[k]
            newS = {i: self.next[k][i][S[:, i]] for i in plan["sites"]}
            dL = sum(self.loss_tab[newS[i]] - self.loss_tab[S[:, i]][:, None] for i in plan["sites"])
            Lt_new = Lt[:, None] + dL
            Li_new = self.loss_tab[newS[pi]] if pi in newS else None
            Lj_new = self.loss_tab[newS[pj]] if pj in newS else None
            if plan["frozen"]:
                q = np.broadcast_to(src["p"][None, :], dL.shape)
            else:
                lg = src["logp"][None, :].repeat(N, 0)
                for i in plan["fsites"]:
                    lg += psi_t[i][newS[i]] - psi_t[i][S[:, i]][:, None]
                for e in plan["edges"]:
                    i, j = P.edges[e]
                    si = newS[i] if i in newS else S[:, i][:, None]
                    sj = newS[j] if j in newS else S[:, j][:, None]
                    lg += chi_t[e][si, sj] - chi_t[e][S[:, i], S[:, j]][:, None]
                ht = h_t[self.blk[k]]
                if self.pair_target:
                    Lin = Li_new if Li_new is not None else Li[:, None]
                    Ljn = Lj_new if Lj_new is not None else Lj[:, None]
                    lg += ht[Lin, Ljn] - ht[Li, Lj][:, None]
                else:
                    lg += ht[Lt_new] - ht[Lt][:, None]
                lg -= lg.max(1, keepdims=True)
                qt = np.exp(lg); qt /= qt.sum(1, keepdims=True)
                q = (1 - a) * qt + a * src["p"][None, :]
            u = rng.random((N, 1)) if U is None else U[:, k:k + 1]
            v = np.minimum((q.cumsum(1) < u).sum(1), q.shape[1] - 1)
            if not plan["frozen"]:
                logw += src["logp"][v] - np.log(q[idx, v])
            for i in newS:
                S[:, i] = newS[i][idx, v]
            Lt = Lt_new[idx, v]
            if Li_new is not None:
                Li = Li_new[idx, v]
            if Lj_new is not None:
                Lj = Lj_new[idx, v]
            if record:
                acts[:, k] = v
        f = P.payoff_from(Lt, Li, Lj)
        return {"f": f, "w": np.exp(logw), "score": P.score_from(Lt, Li, Lj), "acts": acts}

    def replay(self, acts, coef, mode):
        P = self.P
        N = acts.shape[0]
        S, Lt, Li, Lj = self._init_state(N)
        idx = np.arange(N); g = np.zeros(self.d); val = 0.0
        for k in range(P.K):
            v = acts[:, k]
            plan = self.plan[k]
            dPhi, newS, Lt_new, Li_new, Lj_new = self.delta_codes(k, S, Lt, Li, Lj)
            if not plan["frozen"]:
                qt, q = self.probs(k, dPhi)
                cols = plan["idx"]
                mean_dphi = np.einsum("nv,nvd->nd", qt, dPhi)
                if mode == "ce":
                    lp = P.src[k]["logp"][None, :] + dPhi @ self.theta[cols]
                    m = lp.max(1, keepdims=True)
                    val -= coef @ (lp[idx, v] - (m[:, 0] + np.log(np.exp(lp - m).sum(1))))
                    fac = coef
                else:
                    fac = coef * (1 - self.alpha) * qt[idx, v] / q[idx, v]
                np.add.at(g, cols, fac @ (dPhi[idx, v] - mean_dphi))
            for i in newS:
                S[:, i] = newS[i][idx, v]
            Lt = Lt_new[idx, v]; Li = np.asarray(Li_new)[idx, v]; Lj = np.asarray(Lj_new)[idx, v]
        return (val, -g) if mode == "ce" else g

    # ---- compressed training (exact): unique local states per step ----------------------------
    def compile(self, acts):
        """Replay recorded actions once and compress each step to its *unique local states*.
        The features ΔΦ_k depend only on the codes of the touched countries, of the other ends of the
        touched edges and on the target-progress state, so many paths share a row. Returns per step
        (cols, dPhi_u (U,V,d), inv (N,), v (N,), logp (V,)); frozen steps are skipped."""
        P = self.P
        N = acts.shape[0]
        S, Lt, Li, Lj = self._init_state(N)
        idx = np.arange(N); out = []
        for k in range(P.K):
            v = acts[:, k]; plan = self.plan[k]
            if not plan["frozen"]:
                ends = sorted({i for e in plan["edges"] for i in P.edges[e]} | set(plan["sites"]))
                # integer-encode the key (most significant column first): np.unique on a 1-d int array gives the same
                # groups and ordering as the lexicographic row-unique, without the slow row sort
                key = np.zeros(N, dtype=np.int64)
                for i in ends:
                    key = key * 10 + S[:, i]
                if self.pair_target:
                    key = (key * (P.F + 1) + Li) * (P.F + 1) + Lj
                else:
                    key = key * (P.Lmax + 1) + Lt
                _, rep, inv = np.unique(key, return_index=True, return_inverse=True)
                inv = inv.reshape(-1)
                dPhi, *_ = self.delta_codes(k, S[rep], Lt[rep], Li[rep], Lj[rep])
                out.append({"cols": plan["idx"], "dPhi": dPhi, "inv": inv, "v": v, "logp": P.src[k]["logp"],
                            "p": P.src[k]["p"], "dPhi2": dPhi.reshape(-1, dPhi.shape[-1])})
            newS = {i: self.next[k][i][S[:, i]] for i in plan["sites"]}
            dL = sum(self.loss_tab[newS[i]][idx, v] - self.loss_tab[S[:, i]] for i in plan["sites"])
            for i in newS:
                S[:, i] = newS[i][idx, v]
            Lt = Lt + dL
            pi, pj = P.pair
            if pi in newS:
                Li = self.loss_tab[S[:, pi]]
            if pj in newS:
                Lj = self.loss_tab[S[:, pj]]
        return out

    @staticmethod
    def aggregate(compiled, coef):
        """C[u, v] = Σ_{paths in state u taking action v} coef."""
        Cs = []
        for st in compiled:
            C = np.zeros(st["dPhi"].shape[:2])
            np.add.at(C, (st["inv"], st["v"]), coef)
            Cs.append(C)
        return Cs

    def ce_from(self, compiled, Cs):
        """Same value/gradient as replay(acts, coef, 'ce') up to a θ-independent constant."""
        g = np.zeros(self.d); val = 0.0
        th = self.theta
        for st, C in zip(compiled, Cs):
            cols = st["cols"]
            lp = st["logp"][None, :] + st["dPhi"] @ th[cols]
            m = lp.max(1, keepdims=True)
            e = np.exp(lp - m); Z = e.sum(1, keepdims=True); qt = e / Z
            lse = (m + np.log(Z))[:, 0]
            Cs_ = C.sum(1)
            val -= (C * lp).sum() - Cs_ @ lse
            W = C - Cs_[:, None] * qt
            g[cols] += W.reshape(-1) @ st["dPhi2"]          # cols are distinct within a step
        return val, -g

    def mix_from(self, compiled, Cs):
        """Same as replay(acts, coef, 'mix')."""
        g = np.zeros(self.d); a = self.alpha
        for st, C in zip(compiled, Cs):
            cols = st["cols"]
            lp = st["logp"][None, :] + st["dPhi"] @ self.theta[cols]
            lp -= lp.max(1, keepdims=True)
            qt = np.exp(lp); qt /= qt.sum(1, keepdims=True)
            q = (1 - a) * qt + a * st["p"][None, :]
            R = C * (1 - a) * qt / q
            mean = np.einsum("uv,uvd->ud", qt, st["dPhi"])
            g[cols] += R.reshape(-1) @ st["dPhi2"] - R.sum(1) @ mean
        return g
