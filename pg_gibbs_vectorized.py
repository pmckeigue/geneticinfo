"""
Vectorised PG-Gibbs: replaces Python for-loops over n_blocks with
grouped batch operations.

For a dataset like create_data() with n_blocks=3237 (3180 singletons,
57 pairs), the original per-block Python loop costs ~120 ms/iter.
Handling all singletons with one numpy call and all pairs with one
batched 2x2 operation drops this to ~2 ms/iter.

GPU (JAX) is used for batched 2x2 (and 3x3) block operations.
Singleton operations stay on CPU (tiny arrays, GPU launch overhead dominates).
PG sampling stays on CPU (polyagamma has no GPU backend).
"""

import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List
from polyagamma import random_polyagamma
from polyagamma_gibbs import (
    update_c_gibbs_cpu,
    update_indicators_gibbs_cpu,
    slice_sample_theta_collapsed_numpy,
    _chol, _tri_solve,
)


# ── Grouped block structure ───────────────────────────────────────────────

@dataclass
class GroupedBlocks:
    """
    Stores the full L matrix split into groups by block size.
    For each size s, L_s has shape (n_s, s, s) and idx_s has shape (n_s*s,)
    giving the row/column indices into the full M-vector.
    """
    M: int
    sizes: List[int]                      # sorted unique block sizes
    L_by_size: Dict[int, np.ndarray]      # size -> (n_s, s, s) array
    idx_by_size: Dict[int, np.ndarray]    # size -> flat index array length n_s*s

    @staticmethod
    def from_block_structure(blocks) -> "GroupedBlocks":
        """Build from an existing polyagamma_gibbs.BlockStructure."""
        M = blocks.M
        by_size: Dict[int, list] = {}
        for b, (rs, cs) in enumerate(zip(blocks.row_slices, blocks.col_slices)):
            Lb = blocks.L_blocks[b]
            s  = Lb.shape[0]
            by_size.setdefault(s, []).append((rs, cs, Lb))

        sizes      = sorted(by_size.keys())
        L_by_size  = {}
        idx_by_size = {}
        for s, entries in by_size.items():
            Ls   = np.stack([e[2] for e in entries])   # (n_s, s, s)
            idxs = np.concatenate(
                [np.arange(e[0].start, e[0].stop) for e in entries]
            )
            L_by_size[s]   = Ls
            idx_by_size[s] = idxs

        return GroupedBlocks(M=M, sizes=sizes,
                             L_by_size=L_by_size,
                             idx_by_size=idx_by_size)


def _build_eig_contributions_size1(Ls, omega_g, beta0, kappa_g, r_g, tau):
    """
    Vectorised eig-cache contributions for all size-1 blocks at once.
    Ls: (n,1,1), omega_g: (n,), kappa_g: (n,), r_g: (n,)
    Returns (eigvals_vec, proj_vec) each shape (n,).
    """
    l  = Ls[:, 0, 0]                             # diagonal elements
    K  = (l * l) * omega_g                        # (n,)  eigenvalue of 1x1 K
    b  = l * (kappa_g - omega_g * beta0) + 0.5 * r_g
    return K, b                                   # eigval, proj (same for 1x1)


def _build_eig_contributions_size2(Ls, omega_g, beta0, kappa_g, r_g):
    """
    Eig-cache contributions for all size-2 blocks.
    Ls: (n,2,2).  Uses batched 2x2 eigh.
    Returns eigvals (n,2), proj (n,2).
    """
    n = Ls.shape[0]
    # Wb_i = L_i * sqrt(omega_i)[:,None]  -> (n,2,2)
    sqrt_w = np.sqrt(omega_g)          # (n,2)
    Wb = Ls * sqrt_w[:, :, None]       # (n,2,2)   broadcast over columns
    # K_i = Wb_i.T @ Wb_i  (2x2 symmetric)
    K  = np.einsum('nki,nkj->nij', Wb, Wb)   # (n,2,2)

    # b_i = L_i.T @ (kappa_i - omega_i * beta0) + 0.5 * r_i
    t  = kappa_g - omega_g * beta0            # (n,2)
    b  = np.einsum('nij,nj->ni', Ls, t) + 0.5 * r_g   # (n,2): L^T t + 0.5r

    # 2x2 eigh analytically (faster than calling numpy per block)
    # For symmetric 2x2 A = [[a,b],[b,c]]:
    # eigvals = ((a+c) ± sqrt((a-c)^2 + 4b^2)) / 2
    a   = K[:, 0, 0]; bk = K[:, 0, 1]; c = K[:, 1, 1]
    tr  = a + c
    det = a * c - bk * bk
    disc = np.sqrt(np.maximum((tr*tr - 4.0*det), 0.0))
    lam0 = (tr - disc) * 0.5
    lam1 = (tr + disc) * 0.5
    eigvals = np.stack([lam0, lam1], axis=1)   # (n,2)

    # Eigenvectors: for each block, build U = [[u0,-u1],[u1,u0]]
    # where (u0,u1) normalised first eigenvector
    v0 = lam0 - c                        # (n,) a - lam0 (or using standard formula)
    v1 = bk                              # off-diagonal
    nrm = np.sqrt(v0*v0 + v1*v1)
    tiny = nrm < 1e-15
    nrm  = np.where(tiny, 1.0, nrm)
    u0   = np.where(tiny, 1.0, v0 / nrm)
    u1   = np.where(tiny, 0.0, v1 / nrm)
    # proj = U^T b
    proj0 =  u0 * b[:, 0] + u1 * b[:, 1]
    proj1 = -u1 * b[:, 0] + u0 * b[:, 1]
    proj  = np.stack([proj0, proj1], axis=1)   # (n,2)
    return eigvals, proj


def build_theta_eig_cache_fast(gb: GroupedBlocks, omega, beta0, kappa, r,
                               min_block_size: int = 1):
    """
    Vectorised theta-cache.  Returns flat arrays (length M_related) rather
    than a Python list of M small arrays, so the log-posterior evaluation
    is a single vectorised call instead of a M-iteration Python loop.

    min_block_size: only include blocks of this size or larger in the cache.
      Set to 2 to exclude singletons from the theta log-posterior: singletons
      carry no information about lambda_S / mu and their z-prior penalty
      (-0.25*M*mu term) otherwise biases mu towards small values.
    """
    # Collect contributions only from blocks of size >= min_block_size
    ev_parts   = []
    proj_parts = []
    M_related  = 0

    for s in gb.sizes:
        if s < min_block_size:
            continue
        idx = gb.idx_by_size[s]
        Ls  = gb.L_by_size[s]
        n_s = Ls.shape[0]
        omega_g = omega[idx].reshape(n_s, s)
        kappa_g = kappa[idx].reshape(n_s, s)
        r_g     = r[idx].reshape(n_s, s)

        if s == 1:
            ev, pr = _build_eig_contributions_size1(
                Ls, omega_g[:, 0], beta0, kappa_g[:, 0], r_g[:, 0], tau=0.0)
            ev_parts.append(ev)
            proj_parts.append(pr)
        elif s == 2:
            evs, prs = _build_eig_contributions_size2(
                Ls, omega_g, beta0, kappa_g, r_g)
            ev_parts.append(evs.ravel())
            proj_parts.append(prs.ravel())
        else:
            ev_b_list = []; pr_b_list = []
            for i in range(n_s):
                sl = idx[i*s:(i+1)*s]
                Lb = Ls[i];  omega_b = omega[sl];  kappa_b = kappa[sl];  r_b = r[sl]
                Wb = Lb * np.sqrt(omega_b)[:, None]
                Kb = Wb.T @ Wb
                b  = Lb.T @ (kappa_b - omega_b * beta0) + 0.5 * r_b
                ev_bi, U = np.linalg.eigh(Kb)
                ev_b_list.append(ev_bi);  pr_b_list.append(U.T @ b)
            ev_parts.append(np.concatenate(ev_b_list))
            proj_parts.append(np.concatenate(pr_b_list))
        M_related += n_s * s

    if ev_parts:
        eigvals_flat = np.concatenate(ev_parts)
        proj_flat    = np.concatenate(proj_parts)
    else:
        eigvals_flat = np.empty(0)
        proj_flat    = np.empty(0)

    return {"eigvals_flat": eigvals_flat, "proj_flat": proj_flat,
            "M_total": M_related}


def theta_logpost_fast(theta: float, cache: dict,
                       prior_loc: float, prior_scale: float) -> float:
    """
    Evaluate log p(theta | cache) using flat vectorised arrays.
    O(M) numpy — replaces the O(M)-iteration Python loop in
    collapsed_logpost_theta_uncentered_blocks_eig.
    """
    mu     = np.exp(theta)
    tau    = 0.5 / mu
    ev     = cache["eigvals_flat"]
    pr     = cache["proj_flat"]
    M      = float(cache["M_total"])

    lam_tau = ev + tau
    logdet  = float(np.sum(np.log(lam_tau)))
    quad    = float(np.sum(pr * pr / lam_tau))

    # Half-Cauchy(prior_scale) prior on mu.
    # p(mu) = 2 / (pi * prior_scale * (1 + (mu/prior_scale)^2))
    # Change of variables theta = log(mu): add log|d(mu)/d(theta)| = theta.
    lp_prior = theta - np.log(1.0 + (mu / prior_scale) ** 2)

    return (0.5 * quad - 0.5 * logdet
            - 0.25 * M * mu - 0.5 * M * np.log(2.0 * mu)
            + lp_prior)


def sample_beta0_z_fast(
    rng: np.random.Generator,
    gb: GroupedBlocks,
    omega: np.ndarray,
    kappa: np.ndarray,
    r: np.ndarray,
    mu_pos: float,
    beta0_sd: float,
    project_affects_beta0: bool = True,
    jitter: float = 1e-12,
):
    """Vectorised replacement for sample_beta0_z_gaussian_blocks_joint."""
    tau = 1.0 / (2.0 * mu_pos)
    M   = gb.M
    eps = rng.normal(size=M)

    # Accumulators for Schur complement
    a       = float(np.sum(omega)) + 1.0 / (beta0_sd * beta0_sd)
    b0_info = float(np.sum(kappa))

    invb  = np.empty(M); invg  = np.empty(M)
    inv1  = np.empty(M); noise = np.empty(M)
    gT_invg = gT_invb = gT_inv1 = 0.0

    for s in gb.sizes:
        idx = gb.idx_by_size[s]
        Ls  = gb.L_by_size[s]       # (n_s, s, s)
        n_s = Ls.shape[0]
        omega_g = omega[idx].reshape(n_s, s)
        kappa_g = kappa[idx].reshape(n_s, s)
        r_g     = r[idx].reshape(n_s, s)
        eps_g   = eps[idx].reshape(n_s, s)

        if s == 1:
            # Vectorised scalar update for singletons
            l   = Ls[:, 0, 0]                    # (n,)
            w   = omega_g[:, 0]                  # (n,)
            A11 = tau + l*l*w + jitter            # (n,)
            bz  = l*kappa_g[:, 0] + 0.5*r_g[:,0] # (n,)
            g   = l * w                           # (n,)

            inv_A11 = 1.0 / A11                  # compute once; avoids 3 divisions
            ig  = g * inv_A11                    # g/A11
            ib  = bz * inv_A11                   # bz/A11
            no  = eps_g[:, 0] * np.sqrt(inv_A11) # eps/sqrt(A11)

            invb[idx]  = ib;  invg[idx] = ig
            inv1[idx]  = inv_A11;  noise[idx]= no
            # Use element-wise .sum() instead of np.dot to avoid triggering
            # multi-threaded BLAS (OpenBLAS ddot cold-start costs ~50 ms at n~14k)
            gT_invg += float((g * ig).sum())     # sum(g²/A11)
            gT_invb += float((bz * ig).sum())    # sum(g·bz/A11)
            gT_inv1 += float(ig.sum())           # sum(g/A11) = ig.sum()

        elif s == 2:
            # Batched 2x2 Cholesky
            w   = omega_g              # (n,2)
            bz  = np.einsum('nij,nj->ni', np.transpose(Ls,(0,2,1)), kappa_g) \
                  + 0.5 * r_g          # (n,2): L^T kappa + 0.5 r
            g_v = np.einsum('nij,nj->ni', np.transpose(Ls,(0,2,1)), w)  # L^T omega

            Wb  = Ls * np.sqrt(w)[:, :, None]
            K   = np.einsum('nki,nkj->nij', Wb, Wb)  # (n,2,2)
            K[:, 0, 0] += tau + jitter
            K[:, 1, 1] += tau + jitter

            # 2x2 Cholesky analytically
            l11 = np.sqrt(np.maximum(K[:,0,0], 1e-18))
            l21 = K[:,1,0] / l11
            t22 = K[:,1,1] - l21*l21
            l22 = np.sqrt(np.maximum(t22, 1e-18))

            ones2 = np.ones((n_s, 2))
            # solve [[l11,0],[l21,l22]] @ x = rhs  for multiple rhs columns
            def cho2_solve(rhs):          # rhs: (n,2)
                y1 = rhs[:,0] / l11
                y2 = (rhs[:,1] - l21*y1) / l22
                x2 = y2 / l22
                x1 = (y1 - l21*x2) / l11
                return np.stack([x1, x2], axis=1)

            ib = cho2_solve(bz)
            ig = cho2_solve(g_v)
            i1 = cho2_solve(ones2)
            # noise: solve L^T x = eps -> solve lower then upper
            # (same as cho2_solve for lower triangular)
            no = cho2_solve(eps_g)

            # flatten back to M-indexed arrays
            invb[idx]  = ib.ravel()
            invg[idx]  = ig.ravel()
            inv1[idx]  = i1.ravel()
            noise[idx] = no.ravel()

            gT_invg += float(np.einsum('ni,ni->', g_v, ig))
            gT_invb += float(np.einsum('ni,ni->', g_v, ib))
            gT_inv1 += float(np.einsum('ni,ni->', g_v, i1))

        else:
            # General fallback for larger blocks
            for i in range(n_s):
                sl = idx[i*s:(i+1)*s]
                Lb = Ls[i]
                wb = omega[sl];  kb = kappa[sl];  rb = r[sl]
                Wb = Lb * np.sqrt(wb)[:, None]
                A  = Wb.T @ Wb;  A[np.diag_indices_from(A)] += tau + jitter
                Lc = _chol(A)
                v  = eps[sl]
                Y  = _tri_solve(Lc, np.column_stack([Lb.T@kb + 0.5*rb,
                                                     Lb.T@wb,
                                                     np.ones(s), v]), lower=True)
                X  = _tri_solve(Lc.T, Y, lower=False)
                invb[sl] = X[:,0];  invg[sl] = X[:,1]
                inv1[sl] = X[:,2];  noise[sl]= X[:,3]
                gv = Lb.T @ wb
                gT_invg += float(gv @ X[:,1])
                gT_invb += float(gv @ X[:,0])
                gT_inv1 += float(gv @ X[:,2])

    s_schur = max(a - gT_invg, 1e-18)
    mean_b0 = (b0_info - gT_invb) / s_schur
    beta0_new = float(mean_b0 + rng.normal() / np.sqrt(s_schur))
    z_un = invb - invg * beta0_new + noise

    if project_affects_beta0:
        x0    = -gT_inv1 / s_schur
        xz    = inv1 + invg * (gT_inv1 / s_schur)
        denom = max(float(np.sum(xz)), 1e-300)
        alpha = float(np.sum(z_un)) / denom
        beta0_new = beta0_new - x0 * alpha
        z_new     = z_un - xz * alpha
    else:
        denom = max(float(np.sum(inv1)), 1e-300)
        alpha = float(np.sum(z_un)) / denom
        z_new = z_un - inv1 * alpha

    z_new = np.asarray(z_new, dtype=np.float64)
    z_new -= z_new.mean()
    return float(beta0_new), z_new


def sample_r_blocks_fast(
    rng: np.random.Generator,
    gb: GroupedBlocks,
    Lc_by_size: dict,
    b_like_by_size: dict,
    r: np.ndarray,
    p_mix: float,
    max_enum: int = 12,
) -> np.ndarray:
    """
    Vectorised collapsed r update.
    For size-1 blocks: exact 2-state sample (vectorised over all singletons).
    For size-2 blocks: exact 4-state sample (loop over 57 pairs).
    """
    from polyagamma_gibbs import sample_r_block_exact_collapsed

    r_new = r.copy()
    p  = float(np.clip(p_mix, 1e-12, 1.0 - 1e-12))
    lp = np.log(p);  lm = np.log1p(-p)

    for s in gb.sizes:
        idx = gb.idx_by_size[s]
        n_s = gb.L_by_size[s].shape[0]
        Lcs    = Lc_by_size[s]      # (n_s, s, s)
        blikes = b_like_by_size[s]  # (n_s, s)
        r_g    = r[idx].reshape(n_s, s)

        if s == 1:
            # Exact 2-state: r=+1 or r=-1, fully vectorised over all singletons.
            # log w(r=+1) = 0.5*(b+0.5)²/A + lp
            # log w(r=-1) = 0.5*(b-0.5)²/A + lm
            A  = Lcs[:, 0, 0] ** 2     # (n,) A11 from Chol
            b  = blikes[:, 0]           # (n,)
            w_p = 0.5 * (b + 0.5)**2 / A + lp
            w_m = 0.5 * (b - 0.5)**2 / A + lm
            mx  = np.maximum(w_p, w_m)
            e_p = np.exp(w_p - mx);  e_m = np.exp(w_m - mx)
            prob_plus = e_p / (e_p + e_m)
            u = rng.uniform(size=n_s)
            r_new[idx] = np.where(u < prob_plus, 1.0, -1.0)

        elif s == 2:
            # Exact 4-state enumeration, fully vectorised over all pairs.
            # States (row order): (-1,-1), (-1,+1), (+1,-1), (+1,+1)
            R4 = np.array([[-1.,-1.], [-1.,+1.], [+1.,-1.], [+1.,+1.]])  # (4,2)
            lp_state = np.array([2*lm, lp+lm, lp+lm, 2*lp])              # (4,)

            # b_all[i,k,:] = blikes[i] + 0.5*R4[k]  →  (n_s, 4, 2)
            b_all = blikes[:, None, :] + 0.5 * R4[None, :, :]

            # Forward-substitute through lower-triangular 2×2 Cholesky:
            # Lc = [[l11,0],[l21,l22]]
            l11 = Lcs[:, 0, 0]  # (n_s,)
            l21 = Lcs[:, 1, 0]
            l22 = Lcs[:, 1, 1]
            y1  = b_all[:, :, 0] / l11[:, None]              # (n_s, 4)
            y2  = (b_all[:, :, 1] - l21[:, None] * y1) / l22[:, None]
            quad = y1**2 + y2**2                              # (n_s, 4)

            log_w = 0.5 * quad + lp_state[None, :]           # (n_s, 4)
            log_w -= log_w.max(axis=1, keepdims=True)
            probs  = np.exp(log_w)
            probs /= probs.sum(axis=1, keepdims=True)

            # Vectorised categorical sample via inverse CDF
            cdf   = np.cumsum(probs, axis=1)                  # (n_s, 4)
            u     = rng.uniform(size=n_s)[:, None]
            chosen = np.clip((cdf < u).sum(axis=1), 0, 3)    # (n_s,)

            r_new[idx] = R4[chosen].ravel()

        else:
            # General fallback for larger blocks (s ≥ 3)
            for i in range(n_s):
                sl  = idx[i*s:(i+1)*s]
                r_b = sample_r_block_exact_collapsed(
                    rng, Lcs[i], blikes[i], r_g[i], p_mix, max_enum=max_enum)
                r_new[sl] = r_b

    return r_new


def build_Lc_and_blike(
    gb: GroupedBlocks,
    omega: np.ndarray,
    kappa: np.ndarray,
    beta0: float,
    mu_pos: float,
    jitter: float = 1e-12,
):
    """
    Precompute per-size batched Cholesky factors Lc and b_like vectors
    needed by sample_r_blocks_fast.
    Returns dicts keyed by block size.
    """
    tau = 1.0 / (2.0 * mu_pos)
    Lc_by_size    = {}
    b_like_by_size = {}

    for s in gb.sizes:
        idx = gb.idx_by_size[s]
        Ls  = gb.L_by_size[s]
        n_s = Ls.shape[0]
        omega_g = omega[idx].reshape(n_s, s)
        kappa_g = kappa[idx].reshape(n_s, s)

        if s == 1:
            l   = Ls[:, 0, 0]
            w   = omega_g[:, 0]
            A11 = tau + l*l*w + jitter
            Lcs    = np.sqrt(A11)[:, None, None]     # (n,1,1)
            blikes = (l * (kappa_g[:,0] - w * beta0))[:, None]  # (n,1)

        elif s == 2:
            w  = omega_g              # (n,2)
            Wb = Ls * np.sqrt(w)[:, :, None]
            K  = np.einsum('nki,nkj->nij', Wb, Wb)
            K[:, 0, 0] += tau + jitter
            K[:, 1, 1] += tau + jitter
            l11 = np.sqrt(np.maximum(K[:,0,0], 1e-18))
            l21 = K[:,1,0] / l11
            t22 = K[:,1,1] - l21*l21
            l22 = np.sqrt(np.maximum(t22, 1e-18))
            Lcs = np.zeros((n_s, 2, 2))
            Lcs[:,0,0] = l11;  Lcs[:,1,0] = l21;  Lcs[:,1,1] = l22

            t = kappa_g - w * beta0    # (n,2)
            # b_like = L^T t
            blikes = np.einsum('nij,nj->ni', np.transpose(Ls,(0,2,1)), t)

        else:
            Lcs    = np.empty(n_s, dtype=object)
            blikes = np.empty(n_s, dtype=object)
            for i in range(n_s):
                sl = idx[i*s:(i+1)*s]
                Lb = Ls[i];  wb = omega[sl]
                Wb = Lb * np.sqrt(wb)[:, None]
                A  = Wb.T @ Wb;  A[np.diag_indices_from(A)] += tau + jitter
                Lcs[i]    = _chol(A)
                blikes[i] = Lb.T @ (kappa[sl] - wb * beta0)

        Lc_by_size[s]     = Lcs
        b_like_by_size[s] = blikes

    return Lc_by_size, b_like_by_size


# ── Single-chain runner ───────────────────────────────────────────────────

def sparse_matvec(gb: GroupedBlocks, z: np.ndarray) -> np.ndarray:
    """Block-sparse L @ z: O(sum_b n_b * s_b^2) instead of O(M^2)."""
    out = np.empty(gb.M)
    for s in gb.sizes:
        idx = gb.idx_by_size[s]
        Ls  = gb.L_by_size[s]          # (n_s, s, s)
        n_s = Ls.shape[0]
        z_g = z[idx].reshape(n_s, s)
        out[idx] = np.einsum('nij,nj->ni', Ls, z_g).ravel()
    return out


from polyagamma_gibbs import (
    ChainConfig,
    bernoulli_loglik_logit_np,
    infer_blocks_from_L,
    BlockStructure,
)
from tqdm import tqdm
import time


def run_one_chain_fast(
    chain_id: int,
    target: float,
    seed: int,
    dataseed: int,
    K: float,
    cfg: ChainConfig,
) -> dict:
    from geneticinfo import create_data
    M_full, L, y, _, logs = create_data(dataseed, K, float(np.exp(target)))

    L = np.asarray(L, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    perm_rows, perm_cols, row_slices, col_slices = infer_blocks_from_L(L, tol_rel=0.0)
    L_perm = L[perm_rows, :][:, perm_cols]
    y_perm = y[perm_rows]
    inv_perm = np.empty_like(perm_cols)
    inv_perm[perm_cols] = np.arange(perm_cols.size)

    bs  = BlockStructure(L_perm, col_slices, row_slices)
    gb  = GroupedBlocks.from_block_structure(bs)
    M   = gb.M
    kappa = y_perm - 0.5

    print(f"  chain {chain_id}: M={M}  blocks={bs.n_blocks}  "
          + "  ".join(f"size-{s}:{gb.L_by_size[s].shape[0]}" for s in gb.sizes))

    rng  = np.random.default_rng(seed)
    p0   = float(np.mean(y_perm))
    a0   = p0  * cfg.p_prior_conc
    b0_p = (1 - p0) * cfg.p_prior_conc
    p_mix = float(p0)

    theta  = float(cfg.prior_loc)
    mu_pos = float(np.exp(theta))
    beta0  = float(np.log(p0 + 1e-12) - np.log1p(p0 + 1e-12))
    r      = np.where(rng.uniform(size=M) < p0, 1.0, -1.0).astype(np.float64)
    z_unc  = mu_pos * r + np.sqrt(2.0 * mu_pos) * rng.normal(size=M)
    z      = (z_unc - z_unc.mean()).astype(np.float64)
    c      = float(update_c_gibbs_cpu(rng, mu_pos, r))

    mu_post, beta0_post, ll_post = [], [], []
    total = cfg.n_warmup + cfg.n_samples
    pbar  = tqdm(total=total, desc=f"chain {chain_id}",
                 position=chain_id, leave=True, dynamic_ncols=True)
    for it in range(total):
        for _ in range(cfg.inner_latent_cycles):
            eta   = beta0 + sparse_matvec(gb, z)
            omega = random_polyagamma(1.0, np.abs(eta))

            mu_pos = float(np.exp(theta))
            Lc_by_size, b_like_by_size = build_Lc_and_blike(
                gb, omega, kappa, beta0, mu_pos)

            r = sample_r_blocks_fast(
                rng, gb, Lc_by_size, b_like_by_size, r, p_mix)

            beta0, z = sample_beta0_z_fast(
                rng, gb, omega, kappa, r, mu_pos, cfg.beta0_sd,
                project_affects_beta0=cfg.project_affects_beta0)

        mu_pos = float(np.exp(theta))
        c      = float(update_c_gibbs_cpu(rng, mu_pos, r))
        r      = update_indicators_gibbs_cpu(rng, z, c, mu_pos, p_mix)

        n_plus  = float((r > 0).sum())
        n_minus = float((r < 0).sum())
        p_mix   = float(rng.beta(a0 + 10.0*n_plus/(n_plus+n_minus),
                                  b0_p + 10.0*n_minus/(n_plus+n_minus)))

        beta0, z = sample_beta0_z_fast(
            rng, gb, omega, kappa, r, mu_pos, cfg.beta0_sd,
            project_affects_beta0=cfg.project_affects_beta0)

        cache = build_theta_eig_cache_fast(gb, omega, beta0, kappa, r,
                                           min_block_size=2)
        f_th  = lambda th: theta_logpost_fast(
            float(th), cache, float(cfg.prior_loc), float(cfg.prior_scale))
        theta = float(slice_sample_theta_collapsed_numpy(
            rng, theta, f_th,
            w=cfg.slice_w, m=cfg.slice_m, max_steps=cfg.slice_max_steps))

        if it + 1 > cfg.n_warmup:
            mu_post.append(float(np.exp(theta)))
            beta0_post.append(float(beta0))
            eta_ll = beta0 + sparse_matvec(gb, z)
            ll_post.append(bernoulli_loglik_logit_np(eta_ll, y_perm))

        if cfg.diag_every and (it+1) % cfg.diag_every == 0 and len(mu_post) > 1:
            pbar.set_postfix({"mu": f"{np.median(mu_post):.3f}",
                              "p":  f"{p_mix:.4f}"})
        pbar.update(1)

    pbar.close()
    return {"chain_id": chain_id, "target": target,
            "mu": np.array(mu_post), "beta0": np.array(beta0_post),
            "ll": np.array(ll_post)}


def _worker(args):
    return run_one_chain_fast(*args)


def run_one_chain_preloaded(
    chain_id: int,
    gb: GroupedBlocks,
    y_perm: np.ndarray,
    seed: int,
    cfg: ChainConfig,
    true_mu: float = float("nan"),
) -> dict:
    """
    Run one chain on pre-built GroupedBlocks + permuted y.
    Data is constructed once in the parent process and shared via fork.
    """
    M     = gb.M
    kappa = y_perm - 0.5

    rng   = np.random.default_rng(seed)
    p0    = float(np.mean(y_perm))
    a0    = p0 * cfg.p_prior_conc
    b0_p  = (1.0 - p0) * cfg.p_prior_conc
    p_mix = float(p0)

    theta  = float(cfg.prior_loc)
    mu_pos = float(np.exp(theta))
    beta0  = float(np.log(p0 + 1e-12) - np.log1p(p0 + 1e-12))
    r      = np.where(rng.uniform(size=M) < p0, 1.0, -1.0).astype(np.float64)
    z_unc  = mu_pos * r + np.sqrt(2.0 * mu_pos) * rng.normal(size=M)
    z      = (z_unc - z_unc.mean()).astype(np.float64)
    c      = float(update_c_gibbs_cpu(rng, mu_pos, r))

    mu_post, beta0_post, ll_post = [], [], []
    total = cfg.n_warmup + cfg.n_samples
    pbar  = tqdm(total=total, desc=f"chain {chain_id}",
                 position=chain_id, leave=True, dynamic_ncols=True)

    for it in range(total):
        for _ in range(cfg.inner_latent_cycles):
            eta   = beta0 + sparse_matvec(gb, z)
            omega = random_polyagamma(1.0, np.abs(eta))

            mu_pos = float(np.exp(theta))
            Lc_by_size, b_like_by_size = build_Lc_and_blike(
                gb, omega, kappa, beta0, mu_pos)

            r = sample_r_blocks_fast(
                rng, gb, Lc_by_size, b_like_by_size, r, p_mix)

            beta0, z = sample_beta0_z_fast(
                rng, gb, omega, kappa, r, mu_pos, cfg.beta0_sd,
                project_affects_beta0=cfg.project_affects_beta0)

        mu_pos = float(np.exp(theta))
        c      = float(update_c_gibbs_cpu(rng, mu_pos, r))
        r      = update_indicators_gibbs_cpu(rng, z, c, mu_pos, p_mix)

        n_plus  = float((r > 0).sum())
        n_minus = float((r < 0).sum())
        p_mix   = float(rng.beta(a0 + 10.0 * n_plus / (n_plus + n_minus),
                                  b0_p + 10.0 * n_minus / (n_plus + n_minus)))

        beta0, z = sample_beta0_z_fast(
            rng, gb, omega, kappa, r, mu_pos, cfg.beta0_sd,
            project_affects_beta0=cfg.project_affects_beta0)

        cache = build_theta_eig_cache_fast(gb, omega, beta0, kappa, r,
                                           min_block_size=2)
        f_th  = lambda th: theta_logpost_fast(
            float(th), cache, float(cfg.prior_loc), float(cfg.prior_scale))
        theta = float(slice_sample_theta_collapsed_numpy(
            rng, theta, f_th,
            w=cfg.slice_w, m=cfg.slice_m, max_steps=cfg.slice_max_steps))

        if it + 1 > cfg.n_warmup:
            mu_post.append(float(np.exp(theta)))
            beta0_post.append(float(beta0))
            eta_ll = beta0 + sparse_matvec(gb, z)
            ll_post.append(bernoulli_loglik_logit_np(eta_ll, y_perm))

        if cfg.diag_every and (it + 1) % cfg.diag_every == 0 and len(mu_post) > 1:
            pbar.set_postfix({"mu": f"{np.median(mu_post):.3f}",
                              "p":  f"{p_mix:.4f}"})
        pbar.update(1)

    pbar.close()
    return {"chain_id": chain_id, "true_mu": true_mu,
            "mu": np.array(mu_post), "beta0": np.array(beta0_post),
            "ll": np.array(ll_post)}


def _worker_preloaded(args):
    chain_id, gb, y_perm, seed, cfg, true_mu = args
    return run_one_chain_preloaded(chain_id, gb, y_perm, seed, cfg, true_mu)
