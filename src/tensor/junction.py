"""Exact portfolio-loss law on dependence graphs of small width, by frontier variable elimination.

Model (see src/rdpo/problem.py): the terminal state of country i is a capped pair (a_i, r_i), coded
s_i = 2 a_i + r_i in 0..9. The term law is a product of independent sources, each touching one or two
countries, and capped addition makes the source kernels commute. Hence the joint law is the pairwise
factor graph
        P(s) = [ prod_k  K_k  applied in any order ] |0...0>,
and the total loss L = sum_i loss(s_i) is a sum aggregation over countries.

Algorithm. Countries are processed in an elimination order. A country is *introduced* when first needed
(its solo sources give its starting law), every edge source is applied as a 100x100 kernel on the two
active axes, and a country is *finalised* (summed out of the state) once all of its edge sources have
been applied, at which point its loss is added to a running-loss axis. The message therefore has shape
10^w x (loss bins), w = frontier width (= pathwidth bound of the order; equals the treewidth for the
band, k-tree and ladder families used here).

Running loss is carried up to the target threshold only: bins 0..Tc-1 hold the exact mass at loss l,
bin Tc holds the overflow mass O = Pr(L >= Tc) and a separate array holds the first moment of
(L - Tc) on the overflow. Both are linear in the message, so exceedance and stop-loss are exact with no
cancellation (Tc = floor(T) + 1, so {L > T} = {L >= Tc}).

Numerics. Every operation is a sum of non-negative terms (no subtraction), so relative accuracy is at
machine level for every entry, including the rarest tails; no gauge or log scaling is needed (the
tilted gauge of src/tensor/mps.py exists only to protect SVD truncation, which is absent here).
"""
import time

import numpy as np

from src.rdpo.problem import PortfolioProblem, UNIT
from src.tensor.synthetic import SyntheticModel

A_OF, R_OF = np.arange(10) // 2, np.arange(10) % 2


# --- extra topologies (kept here so synthetic.py is untouched) -----------------------------------
def band_edges(n, k):
    """Each country linked to its next k neighbours (a k-tree-like band; treewidth k for n > k)."""
    return [(i, j) for i in range(n) for j in range(i + 1, min(i + k, n - 1) + 1)]


def ladder_edges(n, diagonals=0):
    """2 x (n/2) ladder: country 2c, 2c+1 form rung c. diagonals=0 plain ladder (tw 2),
    1 one diagonal per square (tw 2, triangulated strip), 2 both diagonals (tw 3)."""
    assert n % 2 == 0
    E = [(2 * c, 2 * c + 1) for c in range(n // 2)]
    for c in range(n // 2 - 1):
        E += [(2 * c, 2 * c + 2), (2 * c + 1, 2 * c + 3)]
        if diagonals >= 1:
            E.append((2 * c, 2 * c + 3))
        if diagonals >= 2:
            E.append((2 * c + 1, 2 * c + 2))
    return E


def dense_topology(kind, n):
    from src.tensor.synthetic import topology
    if kind.startswith("band"):
        return band_edges(n, int(kind[4:]))
    if kind == "ladder":
        return ladder_edges(n, 0)
    if kind == "ladder-diag":
        return ladder_edges(n, 1)
    if kind == "ladder-xdiag":
        return ladder_edges(n, 2)
    return topology(kind, n)


# --- graph widths ------------------------------------------------------------------------------
def treewidth_minfill(n, edges):
    """Min-fill greedy elimination: an upper bound on the treewidth."""
    adj = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)
    w = 0
    while adj:
        def fill(v):
            nb = list(adj[v])
            return sum(1 for a in range(len(nb)) for b in range(a + 1, len(nb)) if nb[b] not in adj[nb[a]])
        v = min(adj, key=lambda x: (fill(x), len(adj[x])))
        nb = adj.pop(v)
        w = max(w, len(nb))
        for a in nb:
            adj[a].discard(v)
            adj[a] |= (nb - {a})
    return w


def frontier_width(n, edges, order):
    """Peak number of simultaneously active countries (minus 1 = pathwidth bound) for an order."""
    pos = {v: t for t, v in enumerate(order)}
    nbr = {i: set() for i in range(n)}
    for i, j in edges:
        nbr[i].add(j); nbr[j].add(i)
    active, peak = set(), 0
    for v in order:
        active.add(v)
        active |= {u for u in nbr[v] if pos[u] > pos[v]}
        peak = max(peak, len(active))
        active.discard(v)
    return peak - 1


def best_order(n, edges):
    """Natural order plus greedy growth orders; returns the order minimising frontier width then cost."""
    nbr = {i: set() for i in range(n)}
    for i, j in edges:
        nbr[i].add(j); nbr[j].add(i)
    cands = [list(range(n))]
    for s in range(n):
        order, seen = [s], {s}
        while len(order) < n:
            front = [u for v in order for u in nbr[v] if u not in seen]
            pool = front if front else [u for u in range(n) if u not in seen]
            # prefer the vertex with the most already-placed neighbours, then fewest new ones
            u = min(pool, key=lambda x: (-len(nbr[x] & seen), len(nbr[x] - seen), x))
            order.append(u); seen.add(u)
        cands.append(order)
    return min(cands, key=lambda o: (frontier_width(n, edges, o), o != list(range(n))))


# --- kernels -------------------------------------------------------------------------------------
def _next_code(src, site, v):
    da = sum(inc[v] for s, inc in src["da"] if s == site)
    dr = sum(inc[v] for s, inc in src["dr"] if s == site)
    return 2 * np.minimum(A_OF + da, 4) + np.minimum(R_OF + dr, 1)


def source_kernel(src):
    """Kernel of one source on its distinct sites: (10,10) for one site, (10,10,10,10) for two."""
    sites = list(dict.fromkeys(src["sites"]))
    if len(sites) == 1:
        K = np.zeros((10, 10))
        for v, pv in enumerate(src["p"]):
            if pv:
                K[np.arange(10), _next_code(src, sites[0], v)] += pv
        return sites, K
    K = np.zeros((10, 10, 10, 10))
    ii, jj = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    for v, pv in enumerate(src["p"]):
        if pv:
            ci, cj = _next_code(src, sites[0], v), _next_code(src, sites[1], v)
            np.add.at(K, (ii, jj, ci[ii], cj[jj]), pv)
    return sites, K


# --- the solver ---------------------------------------------------------------------------------
def _shift(slab, xs, d, Tc):
    """Add loss d to every state of (..., Tc+1) mass slab with overflow moment xs (...)."""
    out = np.zeros_like(slab)
    if d < Tc:
        out[..., d:Tc] = slab[..., :Tc - d]
    O = slab[..., Tc]
    lo = max(0, Tc - d)
    spill = slab[..., lo:Tc]
    w = np.arange(lo, Tc) + d - Tc
    out[..., Tc] = O + spill.sum(-1)
    return out, xs + d * O + spill @ w


def junction_solve(prob: PortfolioProblem, order=None):
    """Exact law of the total loss L (units of 2.5m) below the threshold, the overflow mass and the
    overflow first moment. Returns a dict with exceedance, stop_loss (USD), pmf, mass, width, seconds."""
    t0 = time.perf_counter()
    n, edges = prob.n, [tuple(sorted(e)) for e in prob.edges]
    Tc = int(np.floor(prob.T)) + 1
    srcs = [source_kernel(s) for s in prob.src]
    solo = {i: np.eye(10)[0] for i in range(n)}
    pair_src = {}
    for sites, K in srcs:
        if len(sites) == 1:
            solo[sites[0]] = solo[sites[0]] @ K
        else:
            pair_src.setdefault(tuple(sorted(sites)), []).append((sites, K))
    assert set(pair_src) <= set(edges)
    order = list(order) if order is not None else best_order(n, edges)
    pos = {v: t for t, v in enumerate(order)}
    nbr = {i: [] for i in range(n)}
    for i, j in edges:
        nbr[i].append(j); nbr[j].append(i)
    loss = prob.site_loss(A_OF, R_OF).astype(int)
    A = np.zeros(Tc + 1); A[0] = 1.0
    X = np.zeros(())
    act, peak = [], 0

    def introduce(c, A, X):
        v = solo[c].reshape((10, 1))
        return np.expand_dims(A, -2) * v, X[..., None] * solo[c]

    for v in order:
        if v not in act:
            A, X = introduce(v, A, X); act.append(v)
        for u in sorted(nbr[v], key=lambda x: pos[x]):
            if pos[u] < pos[v]:
                continue
            if u not in act:
                A, X = introduce(u, A, X); act.append(u)
            peak = max(peak, len(act))
            a, b = act.index(v), act.index(u)
            for sites, K in pair_src.get(tuple(sorted((u, v))), []):
                Kv = K if sites[0] == v else K.transpose(1, 0, 3, 2)   # axes -> (s_v, s_u, s_v', s_u')
                A = np.moveaxis(np.tensordot(A, Kv, axes=([a, b], [0, 1])), [-2, -1], [a, b])
                X = np.moveaxis(np.tensordot(X, Kv, axes=([a, b], [0, 1])), [-2, -1], [a, b])
        peak = max(peak, len(act))
        a = act.index(v)
        nA, nX = 0.0, 0.0
        for s in range(10):
            sl, xs = _shift(np.take(A, s, axis=a), np.take(X, s, axis=a), int(loss[s]), Tc)
            nA = nA + sl; nX = nX + xs
        A, X = nA, nX; act.remove(v)
    O, M = float(A[Tc]), float(X)
    return {"Tc": Tc, "pmf": A[:Tc].copy(), "overflow_mass": O, "overflow_moment": M,
            "exceedance": O, "stop_loss": (M + O * (Tc - prob.T)) * UNIT,
            "total_mass": float(A.sum()), "width": peak - 1, "order": order,
            "seconds": time.perf_counter() - t0}


# --- pair law --------------------------------------------------------------------------------------
def pair_law(prob: PortfolioProblem, pair=None):
    """Exact joint law of the terminal codes (s_i, s_j) as a 10x10 array (all other sources integrate out),
    any portfolio size; the pair loss law follows by mapping codes to losses."""
    i, j = pair if pair is not None else prob.pair
    dist = np.zeros((10, 10)); dist[0, 0] = 1.0
    for s in prob.src:
        if i not in s["sites"] and j not in s["sites"]:
            continue
        sites, K = source_kernel(s)
        if len(sites) == 2 and set(sites) != {i, j}:      # touches one tracked site: marginalise the other one
            keep = i if i in sites else j
            K = K[:, 0].sum(-1) if sites[0] == keep else K[0].sum(1)    # (10, 10) kernel of the kept site
            sites = [keep]
        elif len(sites) == 2:
            if (sites[0], sites[1]) == (i, j):
                dist = np.einsum("ab,abcd->cd", dist, K)
            else:
                dist = np.einsum("ab,bacd->dc", dist, K)
            continue
        if sites[0] == i:
            dist = np.einsum("ab,ac->cb", dist, K)
        else:
            dist = np.einsum("ab,bc->ac", dist, K)
    return dist
