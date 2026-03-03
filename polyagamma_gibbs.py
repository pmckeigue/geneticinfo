#!/usr/bin/env python3
"""
CPU-only blocked PG–Gibbs with:
- omega ~ PG(1, |eta|) using polyagamma (CPU)
- Joint Gaussian update of (beta0, z) given omega, using block structure of L
- Enforce constraint sum(z)=0
- Gibbs updates for c and r
- Optional local r-block MH (off by default; keep simple)
- Collapsed slice sampling for theta, using per-block eigen cache

Multi-chain runner:
- Run N chains (default 4), sequentially or in parallel (multiprocessing spawn)
- Saves per-chain pickle results + combined file

Assumes you have:
  pip install polyagamma
and your simulator:
  from genetic_data import create_data
"""

import os
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import pickle
import arviz as az
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
#from tqdm.auto import tqdm
from tqdm import tqdm
from geneticinfo import (
    simulate_casecontrol_related, print_casecontrol_summary,
    lr_discrete_blockdiag, reduce_to_relatives, create_data
)

try:
    from scipy.linalg import cholesky as sp_cholesky
    from scipy.linalg import solve_triangular as sp_solve_triangular
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from polyagamma import random_polyagamma


# ============================================================
# Linear algebra helpers
# ============================================================
def _chol(A: np.ndarray) -> np.ndarray:
    if _HAVE_SCIPY:
        return sp_cholesky(A, lower=True, check_finite=False, overwrite_a=False)
    return np.linalg.cholesky(A)


def _tri_solve(L: np.ndarray, B: np.ndarray, lower: bool) -> np.ndarray:
    """Triangular solve wrapper; B can be 1D or 2D."""
    if _HAVE_SCIPY:
        return sp_solve_triangular(L, B, lower=lower, check_finite=False, overwrite_b=False)
    # fallback
    if B.ndim == 1:
        return np.linalg.solve(L if lower else L.T, B)
    X = np.empty_like(B, dtype=np.float64)
    for j in range(B.shape[1]):
        X[:, j] = np.linalg.solve(L if lower else L.T, B[:, j])
    return X


def _softplus_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def bernoulli_loglik_logit_np(eta: np.ndarray, y: np.ndarray) -> float:
    eta = np.asarray(eta, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.sum(y * eta - _softplus_np(eta)))


# ============================================================
# Block detection (Union-Find on sparsity pattern)
# ============================================================
class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def infer_blocks_from_L(L_host: np.ndarray, tol_rel: float = 0.0):
    """
    Find connected components of the bipartite graph induced by nonzeros of L.
    Returns row/col permutations plus slices describing the blocks.

    tol_rel=0.0 means treat exact zeros as zeros.
    """
    Lh = np.asarray(L_host, dtype=np.float64)
    M, P = Lh.shape
    max_abs = float(np.max(np.abs(Lh))) if Lh.size else 0.0
    thr = tol_rel * max_abs
    if max_abs == 0.0:
        return np.arange(M), np.arange(P), [slice(0, M)], [slice(0, P)]

    uf = _UnionFind(M + P)

    # union row node i with col node (M+j) for each nonzero L[i,j]
    for i in range(M):
        row = Lh[i, :]
        nz = np.flatnonzero(np.abs(row) > thr)
        for j in nz:
            uf.union(i, M + j)

    comp_rows: Dict[int, List[int]] = {}
    comp_cols: Dict[int, List[int]] = {}
    for i in range(M):
        r = uf.find(i)
        comp_rows.setdefault(r, []).append(i)
    for j in range(P):
        r = uf.find(M + j)
        comp_cols.setdefault(r, []).append(j)

    comp_keys = sorted(set(comp_rows.keys()) | set(comp_cols.keys()))
    comps = []
    for k in comp_keys:
        rows_k = sorted(comp_rows.get(k, []))
        cols_k = sorted(comp_cols.get(k, []))
        if not rows_k and not cols_k:
            continue
        key_order = rows_k[0] if rows_k else (M + cols_k[0])
        comps.append((key_order, rows_k, cols_k))
    comps.sort(key=lambda t: t[0])

    perm_rows = np.concatenate([np.array(r, dtype=np.int64) for _, r, _ in comps]) if M else np.array([], dtype=np.int64)
    perm_cols = np.concatenate([np.array(c, dtype=np.int64) for _, _, c in comps]) if P else np.array([], dtype=np.int64)

    row_slices = []
    col_slices = []
    r_off = 0
    c_off = 0
    for _, rows_k, cols_k in comps:
        row_slices.append(slice(r_off, r_off + len(rows_k)))
        col_slices.append(slice(c_off, c_off + len(cols_k)))
        r_off += len(rows_k)
        c_off += len(cols_k)

    # If any isolated rows/cols missed (shouldn’t happen often), append them
    if perm_rows.size < M:
        missing = sorted(set(range(M)) - set(perm_rows.tolist()))
        perm_rows = np.concatenate([perm_rows, np.array(missing, dtype=np.int64)])
        row_slices.append(slice(r_off, r_off + len(missing)))
    if perm_cols.size < P:
        missing = sorted(set(range(P)) - set(perm_cols.tolist()))
        perm_cols = np.concatenate([perm_cols, np.array(missing, dtype=np.int64)])
        col_slices.append(slice(c_off, c_off + len(missing)))

    return perm_rows, perm_cols, row_slices, col_slices


# ============================================================
# BlockStructure
# ============================================================
class BlockStructure:
    def __init__(self, L_host: np.ndarray, col_slices: List[slice], row_slices: Optional[List[slice]] = None):
        self.L_host = np.asarray(L_host, dtype=np.float64)
        self.M, self.P = self.L_host.shape
        if row_slices is None:
            if self.M != self.P:
                raise ValueError("row_slices must be provided for rectangular L.")
            row_slices = col_slices

        self.row_slices = list(row_slices)
        self.col_slices = list(col_slices)
        if len(self.row_slices) != len(self.col_slices):
            raise ValueError("row_slices and col_slices must have same length.")

        self.n_blocks = len(self.row_slices)
        self.L_blocks = [self.L_host[rs, cs] for rs, cs in zip(self.row_slices, self.col_slices)]
        self.sizes = [Lb.shape[1] for Lb in self.L_blocks]
        self._ones_blocks = [np.ones((sz,), dtype=np.float64) for sz in self.sizes]

    def L_times_vec(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        y = np.empty((self.M,), dtype=np.float64)
        for b, (rs, cs) in enumerate(zip(self.row_slices, self.col_slices)):
            y[rs] = self.L_blocks[b] @ z[cs]
        return y


# ============================================================
# Model-specific updates: c, r, p
# ============================================================
def update_c_gibbs_cpu(rng: np.random.Generator, mu_pos: float, r: np.ndarray) -> float:
    # Matches your existing logic
    r = np.asarray(r, dtype=np.float64)
    M_proxy = r.shape[0]
    rbar = float(np.mean(r))
    s = float(np.sqrt(2.0 * mu_pos))
    sd_c = s / np.sqrt(M_proxy)
    mean_c = mu_pos * rbar
    return float(mean_c + sd_c * rng.normal())


def update_indicators_gibbs_cpu(rng: np.random.Generator, z: np.ndarray, c: float, mu_pos: float, p_mix: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z_mix = z + float(c)
    s = float(np.sqrt(2.0 * mu_pos))
    p = float(np.clip(p_mix, 1e-12, 1 - 1e-12))
    logp = np.log(p)
    log1mp = np.log1p(-p)
    const = -np.log(s) - 0.5 * np.log(2.0 * np.pi)

    log_plus = logp + const - 0.5 * ((z_mix - mu_pos) / s) ** 2
    log_minus = log1mp + const - 0.5 * ((z_mix + mu_pos) / s) ** 2
    m = np.maximum(log_plus, log_minus)
    prob_plus = np.exp(log_plus - m) / (np.exp(log_plus - m) + np.exp(log_minus - m))

    u = rng.uniform(size=z.shape[0])
    return np.where(u < prob_plus, 1.0, -1.0).astype(np.float64)


# ============================================================
# Collapsed theta target via eigen cache
# ============================================================
def build_theta_eig_cache(blocks: BlockStructure,
                          omega: np.ndarray,
                          beta0: float,
                          kappa: np.ndarray,
                          r: np.ndarray) -> dict:
    """
    For each block b:
      K_b = (sqrt(omega_b) L_b)^T (sqrt(omega_b) L_b)
      b_b = L_b^T (kappa_b - omega_b * beta0) + 0.5 r_b
      store eigvals(K_b), and proj = U^T b_b
    """
    omega = np.asarray(omega, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    eigvals_list = []
    proj_list = []

    for b, cs in enumerate(blocks.col_slices):
        rs = blocks.row_slices[b]
        Lb = blocks.L_blocks[b]
        w_b = omega[rs]
        Wb = Lb * np.sqrt(w_b)[:, None]
        Kb = Wb.T @ Wb

        t_b = kappa[rs] - omega[rs] * float(beta0)
        b_like_b = Lb.T @ t_b
        b_b = b_like_b + 0.5 * r[cs]

        if Kb.shape[0] == 1:
            eigvals = np.array([Kb[0, 0]], dtype=np.float64)
            proj = np.array([b_b[0]], dtype=np.float64)
        else:
            eigvals, U = np.linalg.eigh(Kb)
            proj = U.T @ b_b

        eigvals_list.append(eigvals)
        proj_list.append(proj)

    return {"eigvals": eigvals_list, "proj": proj_list, "M_total": blocks.M}

def empirical_log_lambdaS_selected_from_L(L: np.ndarray, y: np.ndarray, tol: float = 1e-10):
    L = np.asarray(L, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    # detect sib pairs by chol(G)[i,j] ≈ 0.5 below diagonal
    ii, jj = np.where((np.tril(L, k=-1) != 0.0) & (np.abs(np.tril(L, k=-1) - 0.5) < tol))
    pairs = sorted({(int(i), int(j)) if i > j else (int(j), int(i)) for i, j in zip(ii, jj)})
    if not pairs:
        return {"n_pairs": 0}

    both = disc = 0
    for i, j in pairs:
        yi, yj = int(y[i]), int(y[j])
        if yi == 1 and yj == 1:
            both += 1
        elif yi != yj:
            disc += 1

    denom = 2 * both + disc
    if denom == 0:
        return {"n_pairs": len(pairs), "both": both, "disc": disc}

    # ordered P(sib=1 | proband=1) = 2*both / (2*both + disc)
    p_sib_given_case = (2 * both) / denom
    prev_sel = float(y.mean())

    lambdaS_hat = p_sib_given_case / max(prev_sel, 1e-12)
    return {
        "n_pairs": len(pairs),
        "prev_selected": prev_sel,
        "p_sib_given_case_selected": float(p_sib_given_case),
        "lambdaS_hat_selected": float(lambdaS_hat),
        "log_lambdaS_hat_selected": float(np.log(lambdaS_hat)),
    }



def sample_r_block_exact_collapsed(
    rng: np.random.Generator,
    Lc_b: np.ndarray,
    b_like_b: np.ndarray,
    r_b: np.ndarray,
    p_mix: float,
    max_enum: int = 12,
):
    """
    Exact collapsed Gibbs for r in one block by enumerating 2^B states (if B<=max_enum).
    Target within block (up to constants):
      log w(r_b) = 0.5 * || Lc_b^{-1} (b_like_b + 0.5 r_b) ||^2 + log prior(r_b|p)
    """
    B = int(r_b.size)
    if B > max_enum:
        return r_b  # fallback (handle with MH elsewhere)

    p = float(np.clip(p_mix, 1e-12, 1 - 1e-12))
    logp = np.log(p)
    log1mp = np.log1p(-p)

    # Precompute: solve(Lc_b, ·) is used repeatedly
    # Enumerate bit patterns 0..2^B-1 as r in {+1,-1}^B
    n = 1 << B
    logw = np.empty((n,), dtype=np.float64)
    states = np.empty((n, B), dtype=np.float64)

    for s in range(n):
        r_try = np.ones((B,), dtype=np.float64)
        # bit=1 -> -1
        for j in range(B):
            if (s >> j) & 1:
                r_try[j] = -1.0
        states[s] = r_try

        b_try = b_like_b + 0.5 * r_try
        y = np.linalg.solve(Lc_b, b_try)
        quad = float(y @ y)

        # prior: independent: P(r=+1)=p, P(r=-1)=1-p
        lp_prior = float(((r_try > 0).sum()) * logp + ((r_try < 0).sum()) * log1mp)

        logw[s] = 0.5 * quad + lp_prior

    # sample
    m = float(np.max(logw))
    w = np.exp(logw - m)
    w /= w.sum()
    idx = int(rng.choice(n, p=w))
    return states[idx].copy()




def collapsed_logpost_theta_uncentered_blocks_eig(theta: float, cache: dict, prior_loc: float, prior_scale: float) -> float:
    """
    Matches your form:
      tau(theta) = 1/(2*exp(theta))
      lp(theta) = 0.5 * b^T (K+tau I)^-1 b - 0.5 log|K+tau I|
                  - 0.25 M exp(theta) - 0.5 M log(2 exp(theta)) + NormalPrior(theta)
    """
    mu_pos = float(np.exp(theta))
    tau = 1.0 / (2.0 * mu_pos)

    quad = 0.0
    logdet = 0.0
    for vals, proj in zip(cache["eigvals"], cache["proj"]):
        lam_tau = vals + tau
        logdet += float(np.log(lam_tau).sum())
        quad += float(((proj * proj) / lam_tau).sum())

    M_total = float(cache["M_total"])
    # Normal prior on theta
    z = (theta - prior_loc) / prior_scale
    lp_theta = -0.5 * z * z - np.log(prior_scale) - 0.5 * np.log(2.0 * np.pi)
    

    lp = 0.5 * quad - 0.5 * logdet - 0.25 * M_total * mu_pos - 0.5 * M_total * np.log(2.0 * mu_pos) + lp_theta
    return float(lp)


def slice_sample_theta_collapsed_numpy(rng: np.random.Generator,
                                       theta_cur: float,
                                       f_logpost,
                                       w: float = 1.5,
                                       m: int = 20,
                                       max_steps: int = 250) -> float:
    logp_cur = float(f_logpost(theta_cur))
    logy = logp_cur + np.log(rng.uniform())

    t = rng.uniform()
    Lb = theta_cur - t * w
    Rb = Lb + w

    for _ in range(m):
        if f_logpost(Lb) > logy:
            Lb -= w
        else:
            break
    for _ in range(m):
        if f_logpost(Rb) > logy:
            Rb += w
        else:
            break

    for _ in range(max_steps):
        theta_prop = rng.uniform(Lb, Rb)
        if f_logpost(theta_prop) >= logy:
            return float(theta_prop)
        if theta_prop < theta_cur:
            Lb = theta_prop
        else:
            Rb = theta_prop

    return float(theta_cur)


# ============================================================
# JOINT (beta0, z) Gaussian update given omega (CPU, blocks)
# ============================================================
def sample_beta0_z_gaussian_blocks_joint(
    rng: np.random.Generator,
    blocks: BlockStructure,
    omega: np.ndarray,
    kappa: np.ndarray,
    r: np.ndarray,
    mu_pos: float,
    beta0_sd: float,
    project_affects_beta0: bool = True,
    jitter: float = 1e-12,
) -> Tuple[float, np.ndarray]:
    """
    Joint Gaussian draw of (beta0, z) given omega, using an arrowhead Schur complement.

    If project_affects_beta0=True:
      enforce sum(z)=0 by conditioning the JOINT Gaussian on sum(z)=0 (beta0 changes too).
    If False:
      enforce sum(z)=0 only by projecting z (beta0 remains the unconstrained draw).
    """
    omega = np.asarray(omega, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    tau = 1.0 / (2.0 * float(mu_pos))
    P = blocks.P

    # beta0 block:
    a = float(np.sum(omega)) + 1.0 / float(beta0_sd * beta0_sd)
    b0 = float(np.sum(kappa))

    invb = np.empty((P,), dtype=np.float64)   # D^{-1} b_z
    invg = np.empty((P,), dtype=np.float64)   # D^{-1} g
    inv1 = np.empty((P,), dtype=np.float64)   # D^{-1} 1
    noise = np.empty((P,), dtype=np.float64)  # N(0, D^{-1})

    gT_invg = 0.0
    gT_invb = 0.0
    gT_inv1 = 0.0

    eps = rng.normal(size=P)
    eps_off = 0

    for b_idx, cs in enumerate(blocks.col_slices):
        rs = blocks.row_slices[b_idx]
        Lb = blocks.L_blocks[b_idx]
        w_b = omega[rs]

        b_z = Lb.T @ kappa[rs] + 0.5 * r[cs]   # P_b
        g = Lb.T @ w_b                         # P_b
        ones_b = blocks._ones_blocks[b_idx]
        sz = int(Lb.shape[1])

        if sz == 1:
            col = Lb[:, 0]
            A11 = tau + float(np.dot(col * col, w_b)) + jitter

            invb_b = b_z / A11
            invg_b = g / A11
            inv1_b = ones_b / A11

            noise_b = eps[eps_off] / np.sqrt(A11)
            eps_off += 1

            invb[cs] = invb_b
            invg[cs] = invg_b
            inv1[cs] = inv1_b
            noise[cs] = noise_b

            gT_invg += float(g * invg_b)
            gT_invb += float(g * invb_b)
            gT_inv1 += float(g * inv1_b)

        elif sz == 2:
            c1 = Lb[:, 0]
            c2 = Lb[:, 1]
            A11 = tau + float(np.dot(c1 * c1, w_b)) + jitter
            A22 = tau + float(np.dot(c2 * c2, w_b)) + jitter
            A12 = float(np.dot(c1 * c2, w_b))

            l11 = np.sqrt(max(A11, 1e-18))
            l21 = A12 / l11
            t22 = A22 - l21 * l21
            l22 = np.sqrt(max(t22, 1e-18))

            def Ainv(rhs2):
                y1 = rhs2[0] / l11
                y2 = (rhs2[1] - l21 * y1) / l22
                x2 = y2 / l22
                x1 = (y1 - l21 * x2) / l11
                return np.array([x1, x2], dtype=np.float64)

            invb_b = Ainv(b_z)
            invg_b = Ainv(g)
            inv1_b = Ainv(np.array([1.0, 1.0], dtype=np.float64))

            v1 = eps[eps_off]
            v2 = eps[eps_off + 1]
            eps_off += 2
            x2 = v2 / l22
            x1 = (v1 - l21 * x2) / l11
            noise_b = np.array([x1, x2], dtype=np.float64)

            invb[cs] = invb_b
            invg[cs] = invg_b
            inv1[cs] = inv1_b
            noise[cs] = noise_b

            gT_invg += float(g @ invg_b)
            gT_invb += float(g @ invb_b)
            gT_inv1 += float(g @ inv1_b)

        else:
            Wb = Lb * np.sqrt(w_b)[:, None]
            Kb = Wb.T @ Wb
            di = np.diag_indices_from(Kb)
            Kb[di] += tau + jitter

            Lc = _chol(Kb)

            v = eps[eps_off: eps_off + sz]
            eps_off += sz

            Y = _tri_solve(Lc, np.column_stack([b_z, g, ones_b, v]), lower=True)
            X = _tri_solve(Lc.T, Y, lower=False)

            invb_b = X[:, 0]
            invg_b = X[:, 1]
            inv1_b = X[:, 2]
            noise_b = X[:, 3]

            invb[cs] = invb_b
            invg[cs] = invg_b
            inv1[cs] = inv1_b
            noise[cs] = noise_b

            gT_invg += float(g @ invg_b)
            gT_invb += float(g @ invb_b)
            gT_inv1 += float(g @ inv1_b)

    s = float(a - gT_invg)
    s = max(s, 1e-18)

    mean_beta0 = float((b0 - gT_invb) / s)
    beta0_un = float(mean_beta0 + rng.normal() / np.sqrt(s))

    z_un = invb - invg * beta0_un + noise

    # Enforce sum(z)=0
    if project_affects_beta0:
        # Condition the JOINT Gaussian on sum(z)=0
        x0 = -float(gT_inv1 / s)
        xz = inv1 + invg * float(gT_inv1 / s)
        denom = float(np.sum(xz))
        denom = max(denom, 1e-300)
        alpha = float(np.sum(z_un)) / denom

        beta0_new = beta0_un - x0 * alpha
        z_new = z_un - xz * alpha
    else:
        # Project only z (beta0 unchanged)
        denom = float(np.sum(inv1))
        denom = max(denom, 1e-300)
        alpha = float(np.sum(z_un)) / denom
        beta0_new = beta0_un
        z_new = z_un - inv1 * alpha

    # numerical centering
    z_new = np.asarray(z_new, dtype=np.float64)
    z_new = z_new - z_new.mean()
    return float(beta0_new), z_new


# ============================================================
# Single-chain runner
# ============================================================
@dataclass
class ChainConfig:
    n_warmup: int = 1000
    n_samples: int = 5000
    prior_loc: float = 1.0
    prior_scale: float = 0.75
    beta0_sd: float = 5.0
    slice_w: float = 1.5
    slice_m: int = 20
    slice_max_steps: int = 250
    inner_latent_cycles: int = 1
    learn_p: bool = True
    p_prior_conc: float = 200.0
    project_affects_beta0: bool = False
    diag_every: int = 200
    data_flag:int = True # flag for which data creation method to use


def run_one_chain(
    chain_id: int,
    target: float,
    seed: int,
    dataseed: int,
    K: float,
    prune_unrelated: bool,
    cfg: ChainConfig,
) -> dict:
    """
    Creates data (deterministically via dataseed), runs one MCMC chain.
    """
    # Simulate exactly as your original driver: 3rd arg is exp(target)
    if cfg.data_flag:
        M, L, y, dat, logs = create_data(dataseed, K, np.exp(target))
    else:
        y, A, L, info = simulate_casecontrol_related(
            #n_fullsib_pairs=400_000, n_halfsib_pairs=40_000, n_unrelated=4000,
            n_fullsib_pairs=400_000, n_halfsib_pairs=40_000, n_unrelated=4000,
            seed=dataseed, mu=target
        )


    p_obs = float(y.mean())
    M = L.shape[0]





    L = np.asarray(L, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Optional pruning of unrelated individuals (keeps anyone with any off-diagonal nonzero)
    if prune_unrelated:
        off = L.copy()
        np.fill_diagonal(off, 0.0)
        keep = np.any(off != 0.0, axis=1)
        inds = np.flatnonzero(keep).astype(int)
        y = y[inds]
        L = L[np.ix_(inds, inds)]

    # Infer blocks and permute
    perm_rows, perm_cols, row_slices, col_slices = infer_blocks_from_L(L, tol_rel=0.0)
    L_perm = L[perm_rows, :][:, perm_cols]
    y_perm = y[perm_rows]

    # Check exact block diagonal
    nnz_total = int(np.count_nonzero(L_perm))
    nnz_in_blocks = int(sum(np.count_nonzero(L_perm[rs, cs]) for rs, cs in zip(row_slices, col_slices)))
    nnz_off = nnz_total - nnz_in_blocks
    if nnz_off > 0:
        raise RuntimeError(f"[chain {chain_id}] L not exactly block diagonal after permute (off-block nnz={nnz_off}).")

    inv_perm_cols = np.empty_like(perm_cols)
    inv_perm_cols[perm_cols] = np.arange(perm_cols.size)

    blocks = BlockStructure(L_perm, col_slices, row_slices=row_slices)
    M_eff, P_eff = L_perm.shape
    kappa = y_perm - 0.5

    rng = np.random.default_rng(int(seed))

    # priors / init
    p0 = float(np.mean(y_perm))
    #p0 = p_obs
    a0 = p0 * cfg.p_prior_conc
    b0 = (1.0 - p0) * cfg.p_prior_conc

    theta = float(cfg.prior_loc)
    mu_pos = float(np.exp(theta))
    beta0 = float(np.log(p0 + 1e-12) - np.log1p(-(p0 + 1e-12)))

    r = np.where(rng.uniform(size=P_eff) < p0, 1.0, -1.0).astype(np.float64)
    z_unc = mu_pos * r + np.sqrt(2.0 * mu_pos) * rng.normal(size=P_eff)
    z = (z_unc - z_unc.mean()).astype(np.float64)

    c = float(update_c_gibbs_cpu(rng, mu_pos, r))
    p_mix = float(p0)

    mu_post = []
    theta_post = []
    beta0_post = []
    ll_post = []

    last_omega = None

    total_iters = int(cfg.n_warmup + cfg.n_samples)
    #pbar = tqdm(total=total_iters, desc=f"chain {chain_id}", leave=False)
    pbar = tqdm(
        total=total_iters,
        desc=f"chain {chain_id}",
        position=int(chain_id),     # <-- key change: separate line per chain
        leave=True,
        dynamic_ncols=True,
    )

    for it in range(total_iters):

        for _ in range(int(cfg.inner_latent_cycles)):
            eta = beta0 + blocks.L_times_vec(z)
            last_omega = random_polyagamma(1.0, np.abs(eta))



            ###############
            # block/collapsed r update start
            ###############
            mu_pos = float(np.exp(theta))
            tau = 1.0 / (2.0 * mu_pos)
            
            # build A-block Choleskies once (independent of r)
            Lc_blocks = []
            b_like_blocks = []
            
            for b_idx, cs in enumerate(blocks.col_slices):
                rs = blocks.row_slices[b_idx]
                Lb = blocks.L_blocks[b_idx]
                w_b = last_omega[rs]
                Wb = Lb * np.sqrt(w_b)[:, None]
                A_b = Wb.T @ Wb
                A_b[np.diag_indices_from(A_b)] += tau
                Lc_b = _chol(A_b)
                Lc_blocks.append(Lc_b)
            
                t_b = kappa[rs] - last_omega[rs] * float(beta0)
                b_like_b = Lb.T @ t_b
                b_like_blocks.append(b_like_b)
            
            # exact collapsed r update per block
            r_new = r.copy()
            for b_idx, cs in enumerate(blocks.col_slices):
                r_b = r_new[cs]
                r_b_new = sample_r_block_exact_collapsed(
                    rng, Lc_blocks[b_idx], b_like_blocks[b_idx], r_b, p_mix, max_enum=12
                )
                r_new[cs] = r_b_new
            r = r_new
            ###############
            # block/collapsed r update end
            ###############


            mu_pos = float(np.exp(theta))
            beta0, z = sample_beta0_z_gaussian_blocks_joint(
                rng=rng,
                blocks=blocks,
                omega=np.asarray(last_omega),
                kappa=kappa,
                r=r,
                mu_pos=mu_pos,
                beta0_sd=cfg.beta0_sd,
                project_affects_beta0=cfg.project_affects_beta0,
            )

        # c, r, p
        mu_pos = float(np.exp(theta))
        c = float(update_c_gibbs_cpu(rng, mu_pos, r))
        r = update_indicators_gibbs_cpu(rng, z, c, mu_pos, p_mix)




        if cfg.learn_p:
            n_plus = float((r > 0).sum())
            n_minus = float((r < 0).sum())
            n_total = n_plus + n_minus
            n_plus = 10*n_plus/n_total
            n_minus = 10*n_minus/n_total
            p_mix = float(rng.beta(a0 + n_plus, b0 + n_minus))


        ###############
        # single z update start
        ############### 
        beta0, z = sample_beta0_z_gaussian_blocks_joint(
            rng=rng,
            blocks=blocks,
            omega=np.asarray(last_omega),
            kappa=kappa,
            r=r,
            mu_pos=mu_pos,
            beta0_sd=cfg.beta0_sd,
            project_affects_beta0=cfg.project_affects_beta0,
        )
        ###############
        # single z update end
        ###############



        # theta collapsed slice (eig cache)
        cache = build_theta_eig_cache(blocks, np.asarray(last_omega), beta0, kappa, r)
        f_th = lambda th: collapsed_logpost_theta_uncentered_blocks_eig(
            float(th), cache, float(cfg.prior_loc), float(cfg.prior_scale)
        )
        theta = float(slice_sample_theta_collapsed_numpy(
            rng, theta, f_th, w=cfg.slice_w, m=cfg.slice_m, max_steps=cfg.slice_max_steps
        ))

        # store
        if it + 1 > cfg.n_warmup:
            mu_val = float(np.exp(theta))
            mu_post.append(mu_val)
            theta_post.append(theta)
            beta0_post.append(beta0)
            eta_ll = beta0 + blocks.L_times_vec(z)
            ll_post.append(bernoulli_loglik_logit_np(eta_ll, y_perm))

        if cfg.diag_every and ((it + 1) % int(cfg.diag_every) == 0) and len(mu_post) > 2:
            pbar.set_postfix({"mu_med": f"{np.median(mu_post):.3f}", "p": f"{p_mix:.3f}"})

        pbar.update(1)

    pbar.close()

    # map z,r back to original (unpermuted) column order within this chain’s data
    z_back = np.asarray(z)[inv_perm_cols]
    r_back = np.asarray(r)[inv_perm_cols]

    return {
        "chain_id": int(chain_id),
        "seed": int(seed),
        "target": float(target),
        "prune_unrelated": bool(prune_unrelated),
        "cfg": cfg.__dict__,
        "mu": np.asarray(mu_post, dtype=np.float64),
        "theta": np.asarray(theta_post, dtype=np.float64),
        "beta0": np.asarray(beta0_post, dtype=np.float64),
        "ll": np.asarray(ll_post, dtype=np.float64),
        "final_state": {"theta": float(theta), "beta0": float(beta0), "c": float(c), "z": z_back, "r": r_back},
        "perm_rows": perm_rows,
        "perm_cols": perm_cols,
    }


# ============================================================
# Multi-chain harness
# ============================================================
def _worker_run_one_chain(args_tuple):
    return run_one_chain(*args_tuple)


def run_chains(
    target: float,
    seed: int,
    chains: int,
    dataseed: int,
    K: float,
    prune_unrelated: bool,
    cfg: ChainConfig,
    parallel: bool,
    outdir: str,
):
    os.makedirs(outdir, exist_ok=True)

    jobs = []
    for c in range(int(chains)):
        # Chain-specific seed; dataset seed fixed via dataseed
        jobs.append((c, float(target), int(seed + c), int(dataseed), float(K), bool(prune_unrelated), cfg))

    if parallel and chains > 1:
        #import multiprocessing as mp
        #ctx = mp.get_context("spawn")
        #with ctx.Pool(processes=int(chains)) as pool:
        #    outs = pool.map(_worker_run_one_chain, jobs)
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        
        # One shared lock across processes so tqdm output does not interleave
        lock = ctx.RLock()
        
        with ctx.Pool(
            processes=int(chains),
            initializer=tqdm.set_lock,
            initargs=(lock,),
        ) as pool:
            outs = pool.map(_worker_run_one_chain, jobs)

    else:
        outs = [run_one_chain(*job) for job in jobs]

    # Save per-chain
    for out in outs:
        path = os.path.join(outdir, f"chain_{out['chain_id']}_dataseed_{dataseed}_target_{target}_dataflag_{cfg.data_flag}.pkl")
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save combined
    combined_path = os.path.join(outdir, f"all_chains_dataseed_{dataseed}_target_{target}_dataflag_{cfg.data_flag}.pkl")
    with open(combined_path, "wb") as f:
        pickle.dump(outs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Print quick summary
    mus = [o["mu"] for o in outs]
    lens = [len(m) for m in mus]
    print(f"\nSaved {len(outs)} chains to: {outdir}")
    print("Per-chain mu summaries:")
    result_info = {}
    for o in outs:
        mu = o["mu"]
        print(f"  chain {o['chain_id']}: n={len(mu)} median={np.median(mu):.3f} mean={mu.mean():.3f} sd={mu.std():.3f}")
        if "mu" in result_info:
            result_info["mu"] = np.concatenate([result_info["mu"], mu.reshape(1, -1)])
        else:
            result_info["mu"] = mu.reshape(1, -1)
    data = az.convert_to_inference_data(result_info)
    az.style.use(["arviz-white", "arviz-brownish"])
    fig, ax = plt.subplots(1, 2, squeeze=False)
    az.plot_trace(data, var_names=("mu"), kind="rank_vlines", axes=ax)
    fig.savefig(f"polyagamma_gibbs_dataseed_{dataseed}_target_{target}_dataflag_{cfg.data_flag}_plot_trace.png")
    fig, ax = plt.subplots(1, 1)
    az.plot_ess(data, var_names=["mu"], kind="evolution", ax=ax)
    fig.savefig(f"polyagamma_gibbs_dataseed_{dataseed}_target_{target}_dataflag_{cfg.data_flag}_ess.png")

    return outs






# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, required=True, help="Your log-scale target parameter (used as exp(target) in create_data).")
    ap.add_argument("--seed", type=int, default=1, help="Base seed; chain i uses seed+ i.")
    ap.add_argument("--chains", type=int, default=4, help="Number of chains to run (default 4).")
    ap.add_argument("--parallel", action="store_true", help="Run chains in parallel via multiprocessing spawn.")
    ap.add_argument("--outdir", type=str, default="mcmc_out", help="Output directory.")
    ap.add_argument("--dataseed", type=int, default=0, help="Dataset seed (fixed across chains).")
    ap.add_argument("--K", type=float, default=0.01, help="Simulator K.")
    ap.add_argument("--prune-unrelated", action="store_true", help="Drop individuals with no off-diagonal nonzeros in L.")

    # MCMC settings
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--inner-cycles", type=int, default=1)
    ap.add_argument("--prior-loc", type=float, default=1.0)
    ap.add_argument("--prior-scale", type=float, default=0.75)
    ap.add_argument("--beta0-sd", type=float, default=5.0)
    ap.add_argument("--slice-w", type=float, default=1.5)
    ap.add_argument("--slice-m", type=int, default=20)
    ap.add_argument("--slice-max-steps", type=int, default=250)
    ap.add_argument("--p-prior-conc", type=float, default=100.0)
    ap.add_argument("--no-learn-p", action="store_true")
    ap.add_argument("--data-flag", action="store_true")
    ap.add_argument("--project-affects-beta0", action="store_true",
                    help="If set, sum(z)=0 constraint conditions joint (beta0,z) draw (beta0 changes). "
                         "If not set, only z is projected (beta0 unchanged).")
    ap.add_argument("--diag-every", type=int, default=200)

    args = ap.parse_args()

    cfg = ChainConfig(
        n_warmup=int(args.warmup),
        n_samples=int(args.samples),
        prior_loc=float(args.prior_loc),
        prior_scale=float(args.prior_scale),
        beta0_sd=float(args.beta0_sd),
        slice_w=float(args.slice_w),
        slice_m=int(args.slice_m),
        slice_max_steps=int(args.slice_max_steps),
        inner_latent_cycles=int(args.inner_cycles),
        learn_p=(not args.no_learn_p),
        p_prior_conc=float(args.p_prior_conc),
        project_affects_beta0=True,#bool(args.project_affects_beta0),
        diag_every=int(args.diag_every),
        data_flag=args.data_flag
    )

    run_chains(
        target=float(args.target),
        seed=int(args.seed),
        chains=int(args.chains),
        dataseed=int(args.dataseed),
        K=float(args.K),
        prune_unrelated=bool(args.prune_unrelated),
        cfg=cfg,
        parallel=bool(args.parallel),
        outdir=str(args.outdir),
    )





if __name__ == "__main__":
    main()

