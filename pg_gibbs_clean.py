# pg_block_sampler.py
# Self-contained PG Gibbs sampler with block-diagonal L.
#
# Model (M individuals):
#   y_i ~ Bernoulli(sigmoid( eta_i )),   eta = beta0 * 1 + L z
#   omega_i | eta_i ~ PG(1, eta_i)  (Polya-Gamma augmentation)
#
# Prior:
#   r_i in {-1,+1},    P(r_i=+1)=p_mix
#   z | r, mu ~ N(mu*r, 2*mu*I)
#   beta0 ~ N(0, beta0_sd^2)
#   mu ~ HalfCauchy(scale=mu_prior_scale)
#
# We enforce the linear constraint 1^T z = 0 by conditioning the joint
# Gaussian of (beta0,z) on that constraint (optional, recommended).
#
# Dependencies:
#   pip install polyagamma
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from polyagamma import random_polyagamma


# ----------------------------- utilities ---------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable enough for this use; if needed clip x.
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out




def safe_logit(p: float, eps: float = 1e-12) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p) - np.log1p(-p))


def slice_sample_1d(
    rng: np.random.Generator,
    x0: float,
    logpdf,
    w: float = 0.5,
    m: int = 50,
    max_steps_out: int = 1000,
) -> float:
    """
    Standard stepping-out slice sampler for 1D distributions.
    logpdf: callable returning log density (up to additive constant).
    """
    fx0 = float(logpdf(x0))
    # Vertical level: log y = log f(x0) - e, e ~ Exp(1)
    logy = fx0 - rng.exponential()

    # Create initial bracket [L,R]
    u = rng.uniform()
    L = x0 - u * w
    R = L + w

    # Step out
    j = int(rng.integers(0, m))
    k = (m - 1) - j

    steps = 0
    while j > 0 and float(logpdf(L)) > logy and steps < max_steps_out:
        L -= w
        j -= 1
        steps += 1

    steps = 0
    while k > 0 and float(logpdf(R)) > logy and steps < max_steps_out:
        R += w
        k -= 1
        steps += 1

    # Shrinkage
    for _ in range(10_000):
        x1 = float(rng.uniform(L, R))
        fx1 = float(logpdf(x1))
        if fx1 >= logy:
            return x1
        if x1 < x0:
            L = x1
        else:
            R = x1

    raise RuntimeError("slice_sample_1d: shrinkage failed to find a point.")


# ------------------------ block-diagonal L storage ------------------------

@dataclass(frozen=True)
class BlockDiagL:
    """
    Stores a block-diagonal square matrix L (already permuted into block-diag form).

    blocks_by_size[s]: array of shape (n_s, s, s)
    idx_by_size[s]:    integer array of shape (n_s, s) giving positions in [0..M)
    """
    M: int
    sizes: Tuple[int, ...]
    blocks_by_size: Dict[int, np.ndarray]
    idx_by_size: Dict[int, np.ndarray]

    @staticmethod
    def from_slices(L: np.ndarray, block_slices: List[slice]) -> "BlockDiagL":
        """
        Build from a dense L and a list of diagonal block slices.
        Assumes each block uses the same row/col slice.
        """
        M = int(L.shape[0])
        if L.shape[1] != M:
            raise ValueError("L must be square.")

        by_size: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for sl in block_slices:
            s = int(sl.stop - sl.start)
            Lb = np.asarray(L[sl, sl], dtype=np.float64)
            idx = np.arange(sl.start, sl.stop, dtype=np.int64)
            by_size.setdefault(s, []).append((Lb, idx))

        sizes = tuple(sorted(by_size.keys()))
        blocks_by_size: Dict[int, np.ndarray] = {}
        idx_by_size: Dict[int, np.ndarray] = {}
        for s in sizes:
            blocks_by_size[s] = np.stack([x[0] for x in by_size[s]], axis=0)   # (n_s,s,s)
            idx_by_size[s] = np.stack([x[1] for x in by_size[s]], axis=0)      # (n_s,s)

        return BlockDiagL(M=M, sizes=sizes, blocks_by_size=blocks_by_size, idx_by_size=idx_by_size)

    def matvec(self, z: np.ndarray) -> np.ndarray:
        """Compute L @ z using blocks."""
        out = np.empty(self.M, dtype=np.float64)
        for s in self.sizes:
            Ls = self.blocks_by_size[s]         # (n_s,s,s)
            idx = self.idx_by_size[s]           # (n_s,s)
            zg = z[idx]                         # (n_s,s)
            out[idx.ravel()] = np.einsum("nij,nj->ni", Ls, zg).ravel()
        return out

    def Lt_vec(self, v: np.ndarray) -> np.ndarray:
        """Compute L.T @ v using blocks."""
        out = np.empty(self.M, dtype=np.float64)
        for s in self.sizes:
            Ls = self.blocks_by_size[s]         # (n_s,s,s)
            idx = self.idx_by_size[s]           # (n_s,s)
            vg = v[idx]                         # (n_s,s)
            # L^T v : einsum uses transposed indices (n,j,i)
            out[idx.ravel()] = np.einsum("nij,ni->nj", Ls, vg).ravel()
        return out


# ------------------------ core conditional updates ------------------------

def sample_beta0_z_given_omega(
    rng: np.random.Generator,
    Lblk: BlockDiagL,
    omega: np.ndarray,
    kappa: np.ndarray,
    r: np.ndarray,
    mu: float,
    beta0_sd: float,
    constrain_sum_z: bool = True,
    project_affects_beta0: bool = False,
    jitter: float = 1e-12,
) -> Tuple[float, np.ndarray]:
    """
    Sample (beta0, z) from their joint Gaussian conditional given omega, r, mu.

    Uses Schur complement to sample beta0, then z | beta0.
    Optionally conditions on 1^T z = 0 (recommended).
    """
    M = Lblk.M
    tau = 1.0 / (2.0 * float(mu))  # prior precision for z

    # Scalars for beta0 Schur complement
    a = float(np.sum(omega)) + 1.0 / (beta0_sd * beta0_sd)   # beta0 precision
    b0_info = float(np.sum(kappa))                           # beta0 rhs from likelihood

    # We need per-coordinate:
    # A = L^T diag(omega) L + tau I  (block-diagonal)
    # b_z = L^T kappa + 0.5 r
    # g   = L^T omega
    #
    # We'll compute:
    # invb = A^{-1} b_z
    # invg = A^{-1} g
    # inv1 = A^{-1} 1  (for constraint)
    # noise = A^{-1/2} eps (via chol solves)

    eps = rng.normal(size=M).astype(np.float64)

    invb = np.empty(M, dtype=np.float64)
    invg = np.empty(M, dtype=np.float64)
    inv1 = np.empty(M, dtype=np.float64)
    noise = np.empty(M, dtype=np.float64)

    gT_invg = 0.0
    gT_invb = 0.0
    gT_inv1 = 0.0

    ones = np.ones(M, dtype=np.float64)

    for s in Lblk.sizes:
        Ls = Lblk.blocks_by_size[s]          # (n_s,s,s)
        idx = Lblk.idx_by_size[s]            # (n_s,s)
        n_s = int(Ls.shape[0])

        og = omega[idx]                      # (n_s,s)
        kg = kappa[idx]                      # (n_s,s)
        rg = r[idx]                          # (n_s,s)
        eg = eps[idx]                        # (n_s,s)
        oneg = ones[idx]                     # (n_s,s)

        # b_z = L^T kappa + 0.5 r
        bz = np.einsum("nij,ni->nj", Ls, kg) + 0.5 * rg      # (n_s,s)
        # g = L^T omega
        gv = np.einsum("nij,ni->nj", Ls, og)                 # (n_s,s)

        if s == 1:
            l = Ls[:, 0, 0]
            w = og[:, 0]
            A11 = tau + (l * l) * w + jitter
            invA = 1.0 / A11

            ib = bz[:, 0] * invA
            ig = gv[:, 0] * invA
            i1 = oneg[:, 0] * invA
            no = eg[:, 0] * np.sqrt(invA)

            flat = idx.ravel()
            invb[flat] = ib
            invg[flat] = ig
            inv1[flat] = i1
            noise[flat] = no

            gT_invg += float(np.sum(gv[:, 0] * ig))
            gT_invb += float(np.sum(gv[:, 0] * ib))
            gT_inv1 += float(np.sum(gv[:, 0] * i1))

        elif s == 2:
            # Build A = L^T diag(omega) L + tau I for each block
            # via W = L * sqrt(omega) row-wise, then K = W^T W.
            W = Ls * np.sqrt(og)[:, :, None]                  # (n_s,2,2)
            K = np.einsum("nki,nkj->nij", W, W)               # (n_s,2,2)
            K[:, 0, 0] += tau + jitter
            K[:, 1, 1] += tau + jitter

            # Batched 2x2 cholesky (manual, stable enough)
            l11 = np.sqrt(np.maximum(K[:, 0, 0], 1e-30))
            l21 = K[:, 1, 0] / l11
            t22 = K[:, 1, 1] - l21 * l21
            l22 = np.sqrt(np.maximum(t22, 1e-30))

            def cho2_solve(rhs: np.ndarray) -> np.ndarray:
                # Solve A x = rhs for each block using lower-tri chol
                y1 = rhs[:, 0] / l11
                y2 = (rhs[:, 1] - l21 * y1) / l22
                x2 = y2 / l22
                x1 = (y1 - l21 * x2) / l11
                return np.stack([x1, x2], axis=1)

            def cho2_noise(eps: np.ndarray) -> np.ndarray:
                # Solve Lc^T x = eps (upper-triangular solve)
                x2 = eps[:, 1] / l22
                x1 = (eps[:, 0] - l21 * x2) / l11
                return np.stack([x1, x2], axis=1)

            ib = cho2_solve(bz)
            ig = cho2_solve(gv)
            i1 = cho2_solve(oneg)
            no = cho2_noise(eg)

            flat = idx.ravel()
            invb[flat] = ib.ravel()
            invg[flat] = ig.ravel()
            inv1[flat] = i1.ravel()
            noise[flat] = no.ravel()

            gT_invg += float(np.einsum("ni,ni->", gv, ig))
            gT_invb += float(np.einsum("ni,ni->", gv, ib))
            gT_inv1 += float(np.einsum("ni,ni->", gv, i1))

        else:
            # General small-block solve (loop per block)
            for b in range(n_s):
                sl = idx[b]                # shape (s,)
                Lb = Ls[b]                 # (s,s)
                wb = omega[sl]             # (s,)
                bz_b = bz[b]               # (s,)
                gv_b = gv[b]               # (s,)
                one_b = oneg[b]            # (s,)
                e_b = eg[b]                # (s,)

                Wb = Lb * np.sqrt(wb)[:, None]
                A = Wb.T @ Wb
                A[np.diag_indices_from(A)] += tau + jitter
                Lc = np.linalg.cholesky(A)

                def chol_solve(rhs: np.ndarray) -> np.ndarray:
                    y = np.linalg.solve(Lc, rhs)
                    x = np.linalg.solve(Lc.T, y)
                    return x

                ib_b = chol_solve(bz_b)
                ig_b = chol_solve(gv_b)
                i1_b = chol_solve(one_b)
                #no_b = chol_solve(e_b)
                no_b = np.linalg.solve(Lc.T, e_b)

                invb[sl] = ib_b
                invg[sl] = ig_b
                inv1[sl] = i1_b
                noise[sl] = no_b

                gT_invg += float(gv_b @ ig_b)
                gT_invb += float(gv_b @ ib_b)
                gT_inv1 += float(gv_b @ i1_b)

    # Sample beta0 via Schur complement
    s_schur = max(a - gT_invg, 1e-18)
    mean_beta0 = (b0_info - gT_invb) / s_schur
    beta0 = float(mean_beta0 + rng.normal() / np.sqrt(s_schur))

    # Sample unconstrained z
    z_un = invb - invg * beta0 + noise

    #if not constrain_sum_z:
    #    return beta0, z_un.astype(np.float64, copy=False)

    ## Condition the *joint* Gaussian on 1^T z = 0
    ## This is the same correction structure you had, written explicitly.
    #x0 = -gT_inv1 / s_schur
    #xz = inv1 + invg * (gT_inv1 / s_schur)

    #denom = max(float(np.sum(xz)), 1e-300)
    #alpha = float(np.sum(z_un)) / denom

    #beta0 = float(beta0 - x0 * alpha)
    #z = (z_un - xz * alpha).astype(np.float64, copy=False)

    ## Numerical cleanup
    #z = z - float(np.mean(z))
    #return beta0, z

    # Enforce sum(z)=0:
    if project_affects_beta0:
        # joint conditioning (your old "project_affects_beta0=True" branch)
        x0 = -gT_inv1 / s_schur
        xz = inv1 + invg * (gT_inv1 / s_schur)
        denom = max(float(np.sum(xz)), 1e-300)
        alpha = float(np.sum(z_un)) / denom
        beta0 = float(beta0 - x0 * alpha)
        z = z_un - xz * alpha
    else:
        # keep beta0 fixed, project z only (your old "False" branch)
        denom = max(float(np.sum(inv1)), 1e-300)
        alpha = float(np.sum(z_un)) / denom
        z = z_un - inv1 * alpha

    z = np.asarray(z, dtype=np.float64)
    z -= float(np.mean(z))  # numerical cleanup (you did this too)
    return beta0, z


def sample_r_given_z(
    rng: np.random.Generator,
    z: np.ndarray,
    p_mix: float,
) -> np.ndarray:
    """
    Exact conditional for r_i under z|r,mu ~ N(mu*r,2mu) and Bernoulli prior.

    log odds: log(p/(1-p)) + z_i
    """
    logit_p = safe_logit(p_mix)
    prob_plus = sigmoid_stable(z + logit_p)
    u = rng.uniform(size=z.size)
    r = np.where(u < prob_plus, 1.0, -1.0).astype(np.float64)
    return r


def sample_p_mix_given_r(
    rng: np.random.Generator,
    r: np.ndarray,
    a0: float,
    b0: float,
) -> float:
    n_plus = float(np.sum(r > 0))
    n_minus = float(np.sum(r < 0))
    return float(rng.beta(a0 + n_plus, b0 + n_minus))


def logpost_theta_given_z(
    theta: float,
    z: np.ndarray,
    mu_prior_scale: float,
    mu_prior_df: float = 10.0,
) -> float:
    """
    Log posterior for theta=log(mu) given z (and r, but r drops out except const),
    up to an additive constant.  Uses half-Student-t(df, scale) prior on mu.

    From z|r,mu ~ N(mu r, 2 mu I):
      log p(z|mu, r) = - (z^T z)/(4 mu) - (M/2) log(2 mu) - (M/4) mu + const
    Add Jacobian for mu=exp(theta): +theta
    Add half-Student-t log prior on mu: -(df+1)/2 * log(1 + (mu/s)^2/df)
    Half-Cauchy is the special case df=1.
    """
    mu = float(np.exp(theta))
    M = float(z.size)
    z2 = float(z @ z)

    # Likelihood term in mu (constants dropped)
    lp = -(z2 / (4.0 * mu)) - 0.5 * M * np.log(2.0 * mu) - 0.25 * M * mu

    # half-Student-t(df, scale) prior on mu + Jacobian for theta=log(mu)
    df = float(mu_prior_df)
    s  = float(mu_prior_scale)
    lp += theta - 0.5 * (df + 1.0) * np.log(1.0 + (mu / s) ** 2 / df)
    return float(lp)


# ----------------------------- sampler ------------------------------------

@dataclass
class SamplerConfig:
    n_warmup: int = 500
    n_samples: int = 1000

    beta0_sd: float = 5.0
    mu_prior_scale: float = 5.0
    mu_prior_df: float = 10.0   # degrees of freedom for half-Student-t prior on mu

    # Slice sampling for theta=log(mu)
    slice_w: float = 0.5
    slice_m: int = 50

    # Whether to impose 1^T z = 0 each iteration
    constrain_sum_z: bool = True


class PGGibbsBlockSampler:
    def __init__(
        self,
        rng: np.random.Generator,
        Lblk: BlockDiagL,
        y: np.ndarray,
        cfg: SamplerConfig,
        p_mix_prior_conc: float = 2.0,
        p0_init: Optional[float] = None,
    ):
        self.rng = rng
        self.Lblk = Lblk
        self.y = np.asarray(y, dtype=np.float64)
        if self.y.shape != (Lblk.M,):
            raise ValueError("y must have shape (M,) matching L.")

        self.cfg = cfg
        self.kappa = self.y - 0.5

        # p_mix prior
        p0 = float(np.mean(self.y)) if p0_init is None else float(p0_init)
        self.a0 = p0 * p_mix_prior_conc
        self.b0 = (1.0 - p0) * p_mix_prior_conc

        # Initialise
        self.p_mix = float(p0)
        self.theta = float(np.log(1.0))   # mu=1
        self.mu = float(np.exp(self.theta))

        self.beta0 = safe_logit(max(min(p0, 1 - 1e-12), 1e-12))

        M = self.Lblk.M
        self.r = np.where(self.rng.uniform(size=M) < self.p_mix, 1.0, -1.0).astype(np.float64)

        # Prior draw for z | r, mu (then optionally project)
        z_unc = self.mu * self.r + np.sqrt(2.0 * self.mu) * self.rng.normal(size=M)
        self.z = z_unc.astype(np.float64) - float(np.mean(z_unc))

        # omega initial
        eta = self.beta0 + self.Lblk.matvec(self.z)
        self.omega = random_polyagamma(1.0, np.abs(eta))

    def step(self) -> None:
        # 1) Sample omega | beta0, z
        eta = self.beta0 + self.Lblk.matvec(self.z)
        self.omega = random_polyagamma(1.0, np.abs(eta))

        # 2) Sample beta0, z | omega, r, mu
        self.beta0, self.z = sample_beta0_z_given_omega(
            self.rng,
            self.Lblk,
            self.omega,
            self.kappa,
            self.r,
            self.mu,
            beta0_sd=self.cfg.beta0_sd,
            #constrain_sum_z=self.cfg.constrain_sum_z,
            project_affects_beta0=True,
        )

        # 3) Sample r | z, p_mix
        self.r = sample_r_given_z(self.rng, self.z, self.p_mix)

        # 4) Sample p_mix | r
        self.p_mix = sample_p_mix_given_r(self.rng, self.r, self.a0, self.b0)

        # 5) Sample mu via theta=log(mu) | z (slice)
        def lp(th: float) -> float:
            return logpost_theta_given_z(th, self.z, self.cfg.mu_prior_scale, self.cfg.mu_prior_df)

        self.theta, _ = slice_sample_1d(
            self.rng,
            self.theta,
            lp,
            w=self.cfg.slice_w,
            m=self.cfg.slice_m,
        )
        self.mu = float(np.exp(self.theta))

    def run(self) -> Dict[str, np.ndarray]:
        n_w = int(self.cfg.n_warmup)
        n_s = int(self.cfg.n_samples)

        mu_draws = np.empty(n_s, dtype=np.float64)
        beta0_draws = np.empty(n_s, dtype=np.float64)
        p_draws = np.empty(n_s, dtype=np.float64)

        # Warmup
        for _ in range(n_w):
            self.step()

        # Samples
        for t in range(n_s):
            self.step()
            mu_draws[t] = self.mu
            beta0_draws[t] = self.beta0
            p_draws[t] = self.p_mix

        return {"mu": mu_draws, "beta0": beta0_draws, "p_mix": p_draws}


import numpy as np

# ---- bring in the "from scratch" sampler types/functions ----
# from pg_block_sampler import BlockDiagL, SamplerConfig, PGGibbsBlockSampler

def bernoulli_loglik_logit_np(eta: np.ndarray, y: np.ndarray) -> float:
    """
    Sum_i log Bernoulli(y_i | sigmoid(eta_i)) computed stably.
    log p = y*eta - log(1+exp(eta))
    """
    # log(1+exp(eta)) stable:
    # softplus(eta) = log1p(exp(-|eta|)) + max(eta,0)
    softplus = np.log1p(np.exp(-np.abs(eta))) + np.maximum(eta, 0.0)
    return float(np.sum(y * eta - softplus))


def blockdiag_from_groupedblocks(gb) -> "BlockDiagL":
    """
    Convert your GroupedBlocks (flat idx arrays) into BlockDiagL (idx as (n_s,s)).
    Assumes each gb.idx_by_size[s] is concatenated block-by-block in order.
    """
    blocks_by_size = {}
    idx_by_size = {}

    for s in gb.sizes:
        Ls = np.asarray(gb.L_by_size[s], dtype=np.float64)          # (n_s,s,s)
        idx_flat = np.asarray(gb.idx_by_size[s], dtype=np.int64)    # (n_s*s,)
        n_s = Ls.shape[0]
        idx = idx_flat.reshape(n_s, s)                              # (n_s,s)

        blocks_by_size[s] = Ls
        idx_by_size[s] = idx

    return BlockDiagL(
        M=int(gb.M),
        sizes=tuple(gb.sizes),
        blocks_by_size=blocks_by_size,
        idx_by_size=idx_by_size,
    )


def sampler_cfg_from_chain_cfg(cfg) -> "SamplerConfig":
    """
    Map your ChainConfig fields onto the new sampler config.
    """
    return SamplerConfig(
        n_warmup=int(cfg.n_warmup),
        n_samples=int(cfg.n_samples),
        beta0_sd=float(cfg.beta0_sd),
        mu_prior_scale=float(cfg.prior_scale),
        mu_prior_df=float(getattr(cfg, "mu_prior_df", 10.0)),
        slice_w=float(cfg.slice_w),
        slice_m=int(cfg.slice_m),
        #constrain_sum_z=bool(getattr(cfg, "project_affects_beta0", True)),
        constrain_sum_z=True,
    )


def run_one_chain_preloaded_scratch(
    chain_id: int,
    gb,
    y_perm: np.ndarray,
    seed: int,
    cfg,
    true_mu: float = float("nan"),
    p0_override: float | None = None,
) -> dict:
    """
    Same role as your run_one_chain_preloaded, but using the from-scratch sampler.
    """
    rng = np.random.default_rng(seed)

    Lblk = blockdiag_from_groupedblocks(gb)

    y_perm = np.asarray(y_perm, dtype=np.float64)

    scfg = sampler_cfg_from_chain_cfg(cfg)

    # Use p0_override only to initialise p_mix / prior centring if you want.
    # Here we pass it as p0_init.
    p0_init = float(p0_override) if p0_override is not None else None

    sampler = PGGibbsBlockSampler(
        rng=rng,
        Lblk=Lblk,
        y=y_perm,
        cfg=scfg,
        p_mix_prior_conc=float(getattr(cfg, "p_prior_conc", 2.0)),
        p0_init=p0_init,
    )

    # Warmup
    for _ in range(scfg.n_warmup):
        sampler.step()

    # Sample
    mu = np.empty(scfg.n_samples, dtype=np.float64)
    beta0 = np.empty(scfg.n_samples, dtype=np.float64)
    ll = np.empty(scfg.n_samples, dtype=np.float64)

    for t in range(scfg.n_samples):
        sampler.step()
        mu[t] = sampler.mu
        beta0[t] = sampler.beta0
        eta = sampler.beta0 + sampler.Lblk.matvec(sampler.z)
        ll[t] = bernoulli_loglik_logit_np(eta, y_perm)

    return {
        "chain_id": int(chain_id),
        "true_mu": float(true_mu),
        "mu": mu,
        "beta0": beta0,
        "ll": ll,
    }


def _worker_preloaded_scratch(args):
    """
    Accepts the same args-tuple shapes as your original _worker_preloaded.
    Extra entries are ignored (fix_beta0_to_pmix, collapse_beta0) because the
    from-scratch sampler does not use those modes.
    """
    chain_id, gb, y_perm, seed, cfg, true_mu = args[:6]
    p0_override = args[6] if len(args) > 6 else None

    return run_one_chain_preloaded_scratch(
        chain_id=chain_id,
        gb=gb,
        y_perm=y_perm,
        seed=seed,
        cfg=cfg,
        true_mu=true_mu,
        p0_override=p0_override,
    )




from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from polyagamma import random_polyagamma


# ---------------- numerics ----------------

def softplus(x: np.ndarray) -> np.ndarray:
    # stable log(1+exp(x))
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def sigmoid_scalar(phi: float) -> float:
    # stable sigmoid for scalar
    if phi >= 0:
        z = np.exp(-phi)
        return float(1.0 / (1.0 + z))
    else:
        z = np.exp(phi)
        return float(z / (1.0 + z))

def log_sigmoid(phi: float) -> float:
    # log(sigmoid(phi)) stably
    return float(-softplus(np.array([-phi]))[0])

def log1m_sigmoid(phi: float) -> float:
    # log(1-sigmoid(phi)) stably
    return float(-softplus(np.array([phi]))[0])

def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def slice_sample_1d(
    rng: np.random.Generator,
    x0: float,
    logpdf,
    w: float = 0.5,
    m: int = 50,
    max_steps_out: int = 5000,
) -> float:
    fx0 = float(logpdf(x0))
    logy = fx0 - rng.exponential()

    u = rng.uniform()
    L = x0 - u * w
    R = L + w

    j = int(rng.integers(0, m))
    k = (m - 1) - j

    steps = 0
    while j > 0 and float(logpdf(L)) > logy and steps < max_steps_out:
        L -= w
        j -= 1
        steps += 1

    steps = 0
    while k > 0 and float(logpdf(R)) > logy and steps < max_steps_out:
        R += w
        k -= 1
        steps += 1

    n_shrink = 0
    for _ in range(100_000):
        x1 = float(rng.uniform(L, R))
        fx1 = float(logpdf(x1))
        if fx1 >= logy:
            return x1, n_shrink
        n_shrink += 1
        if x1 < x0:
            L = x1
        else:
            R = x1

    raise RuntimeError("slice_sample_1d: shrinkage failed.")


# ---------------- block-diagonal L ----------------

@dataclass(frozen=True)
class BlockDiagL:
    """
    blocks_by_size[s]: (n_s, s, s)
    idx_by_size[s]:    (n_s, s) integer indices into the flat M-vector
    """
    M: int
    sizes: Tuple[int, ...]
    blocks_by_size: Dict[int, np.ndarray]
    idx_by_size: Dict[int, np.ndarray]

    def matvec(self, z: np.ndarray) -> np.ndarray:
        out = np.empty(self.M, dtype=np.float64)
        for s in self.sizes:
            Ls = self.blocks_by_size[s]
            idx = self.idx_by_size[s]
            zg = z[idx]  # (n_s, s)
            out[idx.ravel()] = np.einsum("nij,nj->ni", Ls, zg).ravel()
        return out

    def Lt_vec(self, v: np.ndarray) -> np.ndarray:
        out = np.empty(self.M, dtype=np.float64)
        for s in self.sizes:
            Ls = self.blocks_by_size[s]
            idx = self.idx_by_size[s]
            vg = v[idx]  # (n_s, s)
            out[idx.ravel()] = np.einsum("nij,ni->nj", Ls, vg).ravel()
        return out


# ---------------- Gaussian draw for Z_mix ----------------

def sample_Zmix_given_omega(
    rng: np.random.Generator,
    Lblk: BlockDiagL,
    omega: np.ndarray,
    kappa: np.ndarray,
    phi: float,
    mu: float,
    r: np.ndarray,
    v_L1: np.ndarray,      # v = L @ 1  (length M)
    jitter: float = 1e-12,
) -> np.ndarray:
    """
    Sample Z_mix from:
      p(Z_mix | omega, y, phi, mu, r)  where eta = phi + L@(Z_mix - shift) and shift=(2p-1)mu.
    We work with the PG-augmented Gaussian conditional.

    Conditional precision: A = L^T diag(omega) L + tau I,  tau = 1/(2mu)
    RHS: b = L^T (kappa - omega * c) + 0.5 r
    where c = phi - shift * v_L1   (length M; the "offset" part in eta)
    """
    M = Lblk.M
    tau = 1.0 / (2.0 * float(mu))

    # p = sigmoid(phi), mbar = 2p-1 = tanh(phi/2)
    mbar = float(np.tanh(0.5 * phi))
    shift = float(mu * mbar)

    c = phi - shift * v_L1                     # (M,)
    b_like = kappa - omega * c                 # (M,)

    eps = rng.normal(size=M).astype(np.float64)

    Zmix = np.empty(M, dtype=np.float64)

    for s in Lblk.sizes:
        Ls = Lblk.blocks_by_size[s]           # (n_s,s,s)
        idx = Lblk.idx_by_size[s]             # (n_s,s)
        n_s = int(Ls.shape[0])

        og = omega[idx]                       # (n_s,s)
        blg = b_like[idx]                     # (n_s,s)
        rg = r[idx]                           # (n_s,s)
        eg = eps[idx]                         # (n_s,s)

        # bz = L^T b_like + 0.5 r
        bz = np.einsum("nij,ni->nj", Ls, blg) + 0.5 * rg   # (n_s,s)

        if s == 1:
            l = Ls[:, 0, 0]
            w = og[:, 0]
            A11 = tau + (l * l) * w + jitter

            mean = bz[:, 0] / A11
            noise = eg[:, 0] / np.sqrt(A11)   # = A^{-1/2} eps
            Zmix[idx.ravel()] = mean + noise

        elif s == 2:
            # Build K = L^T diag(w) L via W = sqrt(w)*L (row-scaled)
            W = Ls * np.sqrt(og)[:, :, None]      # (n,2,2)
            K = np.einsum("nki,nkj->nij", W, W)   # (n,2,2)
            K[:, 0, 0] += tau + jitter
            K[:, 1, 1] += tau + jitter

            # chol(K) = [[l11,0],[l21,l22]]
            l11 = np.sqrt(np.maximum(K[:, 0, 0], 1e-30))
            l21 = K[:, 1, 0] / l11
            t22 = K[:, 1, 1] - l21 * l21
            l22 = np.sqrt(np.maximum(t22, 1e-30))

            def cho2_solve(rhs: np.ndarray) -> np.ndarray:
                # Solve K x = rhs via chol lower+upper (two triangular solves)
                y1 = rhs[:, 0] / l11
                y2 = (rhs[:, 1] - l21 * y1) / l22
                x2 = y2 / l22
                x1 = (y1 - l21 * x2) / l11
                return np.stack([x1, x2], axis=1)

            def cho2_noise(eps2: np.ndarray) -> np.ndarray:
                # noise = L^{-T} eps: solve (chol(K).T) x = eps (one upper solve)
                x2 = eps2[:, 1] / l22
                x1 = (eps2[:, 0] - l21 * x2) / l11
                return np.stack([x1, x2], axis=1)

            mean = cho2_solve(bz)
            noise = cho2_noise(eg)
            Zmix[idx.ravel()] = (mean + noise).ravel()

        else:
            # small loop is fine for triplets etc.
            for b in range(n_s):
                sl = idx[b]
                Lb = Ls[b]
                wb = omega[sl]
                bzb = bz[b]
                eb = eg[b]

                Wb = Lb * np.sqrt(wb)[:, None]
                A = Wb.T @ Wb
                A[np.diag_indices_from(A)] += tau + jitter
                Lc = np.linalg.cholesky(A)

                # mean: A^{-1} b
                y = np.linalg.solve(Lc, bzb)
                mean = np.linalg.solve(Lc.T, y)

                # noise: A^{-1/2} eps = Lc^{-T} eps
                noise = np.linalg.solve(Lc.T, eb)

                Zmix[sl] = mean + noise

    return Zmix


# ---------------- config & sampler ----------------

@dataclass
class LRBlockdiagPGConfig:
    n_warmup: int = 1000
    n_samples: int = 5000

    # priors
    K_beta: float = 20.0                # same "K=20" as your numpyro model
    p_obs: float = 0.5                  # prior centre (full-sample observed case fraction)
    mu_prior_scale: float = 1.0         # half-Student-t scale
    mu_prior_df: float = 1.0           # degrees of freedom (df=1 → half-Cauchy)

    # slice sampling settings
    slice_w_phi: float = 1.0
    slice_m_phi: int = 20
    slice_w_theta: float = 1.0
    slice_m_theta: int = 20

    # alpha reparameterisation for phi (decouples phi from theta via K constraint)
    use_alpha_reparam: bool = False

    # phi correction: after the collapsed theta update, shift phi by
    # Δmu·tanh(φ/2)·mean_v (approximate K-preserving correction) then
    # re-sample phi from its exact conditional given the new mu.
    use_phi_correction: bool = False

    # collapsed phi update: integrate out Zmix for the phi slice too,
    # reusing the same eigendecomposition as the theta (mu) update.
    use_collapsed_phi: bool = True

    # auxiliary variable expansion for heavy-tailed prior on mu:
    # replaces HalfCauchy (or low-df Student-t) with a scale mixture of
    # HalfNormals.  Enabled automatically when mu_prior_df < 10.
    # aux_v is the current auxiliary variance (updated each iteration).
    use_cauchy_aux: bool = False
    aux_v: float = 1.0


class LRDiscreteBlockdiagPGGibbs:
    """
    PG-Gibbs targeting the same model as lr_discrete_blockdiag.

    State:
      phi   = logit(p) = beta0
      mu    > 0
      r     in {-1,+1}^M         (indicator; +1 corresponds to numpyro's z=1)
      Zmix  in R^M
      omega in R_+^M
    """
    def __init__(
        self,
        rng: np.random.Generator,
        Lblk: BlockDiagL,
        y: np.ndarray,
        cfg: LRBlockdiagPGConfig,
        phi_init: Optional[float] = None,
        mu_init: float = 1.0,
    ):
        self.rng = rng
        self.Lblk = Lblk
        self.y = np.asarray(y, dtype=np.float64)
        self.cfg = cfg

        if self.y.shape != (Lblk.M,):
            raise ValueError("y must have shape (M,) matching L.")

        self.kappa = self.y - 0.5

        # precompute v = L @ 1 (used for centring shift)
        self.v_L1 = self.Lblk.matvec(np.ones(self.Lblk.M, dtype=np.float64))
        self.mean_v = float(np.mean(self.v_L1))

        # initialise phi near logit(p_obs)
        if phi_init is None:
            p0 = float(np.clip(cfg.p_obs, 1e-9, 1.0 - 1e-9))
            self.phi = float(np.log(p0) - np.log1p(-p0))
        else:
            self.phi = float(phi_init)

        self.mu = float(mu_init)
        self.theta = float(np.log(self.mu))

        # init r ~ Bernoulli(p)
        p = sigmoid_scalar(self.phi)
        u = self.rng.uniform(size=self.Lblk.M)
        self.r = np.where(u < p, 1.0, -1.0).astype(np.float64)

        # init Zmix from prior: N(mu*r, 2mu)
        self.Zmix = (self.mu * self.r + np.sqrt(2.0 * self.mu) * self.rng.normal(size=self.Lblk.M)).astype(np.float64)

        # init omega from current eta
        self.omega = np.ones(self.Lblk.M, dtype=np.float64)
        self.resample_omega()

        # diagnostic accumulators
        self._theta_shrink_total: int = 0
        self._theta_step_total: float = 0.0
        self._step_count: int = 0

    def current_p(self) -> float:
        return sigmoid_scalar(self.phi)

    def current_shift(self) -> float:
        # shift = (2p-1)*mu = tanh(phi/2)*mu
        return float(np.tanh(0.5 * self.phi) * self.mu)

    def current_eta(self) -> np.ndarray:
        shift = self.current_shift()
        x = self.Lblk.matvec(self.Zmix)          # L @ Zmix
        return (self.phi + x - shift * self.v_L1).astype(np.float64)

    def resample_omega(self) -> None:
        eta = self.current_eta()
        self.omega = random_polyagamma(1.0, np.abs(eta)).astype(np.float64)

    def logpost_phi_given_rest(self, phi: float) -> float:
        # phi = logit(p)
        # p = sigmoid(phi), mbar = tanh(phi/2)
        p = sigmoid_scalar(phi)
        mbar = float(np.tanh(0.5 * phi))
        shift = float(self.mu * mbar)

        # augmented likelihood term: kappa*eta - 0.5*omega*eta^2
        x = self.Lblk.matvec(self.Zmix)
        eta = phi + x - shift * self.v_L1

        ll_aug = float(np.dot(self.kappa, eta) - 0.5 * np.dot(self.omega, eta * eta))

        # priors:
        a = float(self.cfg.K_beta * self.cfg.p_obs)
        b = float(self.cfg.K_beta * (1.0 - self.cfg.p_obs))

        # log p, log(1-p) stably from phi (no clipping)
        logp = log_sigmoid(phi)
        log1mp = log1m_sigmoid(phi)

        n_plus = float(np.sum(self.r > 0))
        n_minus = float(np.sum(self.r < 0))

        lp_beta = (a - 1.0) * logp + (b - 1.0) * log1mp
        lp_r = n_plus * logp + n_minus * log1mp

        # Jacobian dp/dphi = p(1-p): add log p + log(1-p)
        lp_jac = logp + log1mp

        return ll_aug + lp_beta + lp_r + lp_jac

    def logpost_theta_given_rest(self, theta: float) -> float:
        # theta = log(mu)
        if theta < -745.0 or theta > 709.0:
            return -np.inf
        mu = float(np.exp(theta))
        if not np.isfinite(mu) or mu <= 0.0:
            return -np.inf

        # shift depends on mu and phi
        mbar = float(np.tanh(0.5 * self.phi))
        shift = float(mu * mbar)

        x = self.Lblk.matvec(self.Zmix)
        eta = self.phi + x - shift * self.v_L1
        ll_aug = float(np.dot(self.kappa, eta) - 0.5 * np.dot(self.omega, eta * eta))

        # prior on Zmix | r, mu : Normal(mu*r, 2mu)
        diff = self.Zmix - mu * self.r
        M = float(self.Lblk.M)
        lp_Z = float(-0.5 * M * np.log(mu) - (diff @ diff) / (4.0 * mu))

        # half-Student-t(df, scale) prior on mu + Jacobian for theta=log(mu)
        s  = float(self.cfg.mu_prior_scale)
        df = float(self.cfg.mu_prior_df)
        lp_mu = float(theta - 0.5 * (df + 1.0) * np.log(1.0 + (mu / s) ** 2 / df))

        return ll_aug + lp_Z + lp_mu



    def step(self) -> None:
        # 1) omega | eta
        self.resample_omega()

        # 2) Zmix | omega, y, phi, mu, r  (Gaussian)
        self.Zmix = sample_Zmix_given_omega(
            self.rng, self.Lblk, self.omega, self.kappa,
            self.phi, self.mu, self.r, self.v_L1
        )

        # 3) r | Zmix, phi  (exact)
        prob_plus = sigmoid_stable(self.Zmix + self.phi)
        u = self.rng.uniform(size=self.Lblk.M)
        self.r = np.where(u < prob_plus, 1.0, -1.0).astype(np.float64)

        if self.cfg.use_collapsed_phi:
            # Build cache once — eigendecompositions are reused for both
            # the collapsed phi slice and the collapsed theta (mu) slice.
            mu_cache = build_mu_collapsed_cache(
                self.Lblk, self.omega, self.kappa, self.r, self.phi, self.v_L1,
                mu_prior_scale=self.cfg.mu_prior_scale,
                mu_prior_df=self.cfg.mu_prior_df,
                K_beta=self.cfg.K_beta,
                p_obs=self.cfg.p_obs,
                mu=self.mu,
            )

            # 4) phi | omega, r  (collapsed over Zmix)
            def lp_phi_col(phi: float) -> float:
                return logpost_phi_collapsed(phi, mu_cache)
            self.phi, _ = slice_sample_1d(
                self.rng, self.phi, lp_phi_col,
                w=self.cfg.slice_w_phi, m=self.cfg.slice_m_phi,
            )

            # Update phi-dependent cache fields; reuse eigendecompositions.
            mu_cache = update_cache_for_new_phi(mu_cache, self.phi)

            # 5) theta | omega, r  (collapsed over Zmix)
            _aux_v = self.cfg.aux_v if self.cfg.use_cauchy_aux else 0.0
            def lp_theta_col(th: float) -> float:
                return logpost_theta_mu_collapsed(th, mu_cache, aux_v=_aux_v)
            new_theta, n_shrink = slice_sample_1d(
                self.rng, self.theta, lp_theta_col,
                w=self.cfg.slice_w_theta, m=self.cfg.slice_m_theta,
            )
            self._theta_shrink_total += n_shrink
            self._theta_step_total   += abs(new_theta - self.theta)
            self._step_count         += 1
            self.theta = new_theta
            self.mu    = float(np.exp(self.theta))

        else:
            # 4) phi | rest (slice); optionally via alpha = phi - mu*mbar*mean_v
            if self.cfg.use_alpha_reparam:
                mbar  = float(np.tanh(0.5 * self.phi))
                shift = self.mu * mbar * self.mean_v
                def lp_alpha(alpha: float) -> float:
                    return self.logpost_phi_given_rest(alpha + shift)
                new_alpha, _ = slice_sample_1d(
                    self.rng, self.phi - shift, lp_alpha,
                    w=self.cfg.slice_w_phi, m=self.cfg.slice_m_phi,
                )
                self.phi = new_alpha + shift
            else:
                self.phi, _ = slice_sample_1d(
                    self.rng, self.phi, self.logpost_phi_given_rest,
                    w=self.cfg.slice_w_phi, m=self.cfg.slice_m_phi,
                )

            # 5) mu | omega, r, phi, y  (COLLAPSED over Zmix)
            mu_before    = self.mu
            theta_before = self.theta
            mu_cache = build_mu_collapsed_cache(
                self.Lblk, self.omega, self.kappa, self.r, self.phi, self.v_L1,
                mu_prior_scale=self.cfg.mu_prior_scale,
                mu_prior_df=self.cfg.mu_prior_df,
            )

            # Auxiliary variable update for heavy-tailed prior
            if self.cfg.use_cauchy_aux:
                self.cfg.aux_v = (self.mu ** 2 + 1.0) / (2.0 * self.rng.exponential(1.0))
            _aux_v = self.cfg.aux_v if self.cfg.use_cauchy_aux else 0.0

            def lp_theta(th: float) -> float:
                return logpost_theta_mu_collapsed(th, mu_cache, aux_v=_aux_v)

            new_theta, n_shrink = slice_sample_1d(
                self.rng, self.theta, lp_theta,
                w=self.cfg.slice_w_theta, m=self.cfg.slice_m_theta,
            )
            self._theta_shrink_total += n_shrink
            self._theta_step_total   += abs(new_theta - self.theta)
            self._step_count         += 1
            self.theta = new_theta
            self.mu    = float(np.exp(self.theta))

            # 5b) phi correction: shift phi by the first-order K-preserving amount
            # Δθ·μ_old·tanh(φ/2)·mean_v, then re-sample phi from its exact
            # conditional given the new mu.
            if self.cfg.use_phi_correction:
                delta_theta = new_theta - theta_before
                phi_corrected = (self.phi
                                 + delta_theta * mu_before
                                 * float(np.tanh(0.5 * self.phi)) * self.mean_v)
                if np.isfinite(self.logpost_phi_given_rest(phi_corrected)):
                    self.phi = phi_corrected
                self.phi, _ = slice_sample_1d(
                    self.rng, self.phi, self.logpost_phi_given_rest,
                    w=self.cfg.slice_w_phi, m=500,
                )

        # 6) Refresh Zmix once more under the new mu (optional but usually helps)
        self.Zmix = sample_Zmix_given_omega(
            self.rng, self.Lblk, self.omega, self.kappa,
            self.phi, self.mu, self.r, self.v_L1
        )

    def step_asis(self) -> Tuple[float, int]:
        """
        ASIS ancillary-statistic (AS) update for theta.

        Complements the collapsed SS update in step(). Fixes the NCP
        standardised residual u = (Zmix - mu*r)/sqrt(2*mu) and samples
        theta from p(theta_nc | u, omega, phi, r, y), then reconstructs
        Zmix = mu_new*r + sqrt(2*mu_new)*u.

        Returns (new_theta, n_shrink) for diagnostics.
        """
        sq_mu = np.sqrt(2.0 * self.mu)
        if sq_mu < 1e-300:
            return self.theta, 0

        u = (self.Zmix - self.mu * self.r) / sq_mu

        mbar = float(np.tanh(0.5 * self.phi))
        a = self.Lblk.matvec(self.r) - mbar * self.v_L1
        b = self.Lblk.matvec(u)

        def lp_nc(theta_nc: float) -> float:
            if theta_nc < -745.0 or theta_nc > 709.0:
                return -np.inf
            if abs(theta_nc - self.theta) > 15.0:
                return -np.inf
            mu_nc = float(np.exp(theta_nc))
            eta = self.phi + mu_nc * a + np.sqrt(2.0 * mu_nc) * b
            ll_aug = float(np.dot(self.kappa, eta) - 0.5 * np.dot(self.omega, eta * eta))
            if self.cfg.use_cauchy_aux:
                lp_prior = theta_nc - mu_nc ** 2 / (2.0 * self.cfg.aux_v)
            else:
                s  = float(self.cfg.mu_prior_scale)
                df = float(self.cfg.mu_prior_df)
                lp_prior = theta_nc - 0.5 * (df + 1.0) * np.log(1.0 + (mu_nc / s) ** 2 / df)
            return ll_aug + lp_prior

        new_theta, n_shrink = slice_sample_1d(
            self.rng, self.theta, lp_nc,
            w=self.cfg.slice_w_theta, m=self.cfg.slice_m_theta,
        )
        mu_new = float(np.exp(new_theta))
        self.theta = new_theta
        self.mu = mu_new
        self.Zmix = mu_new * self.r + np.sqrt(2.0 * mu_new) * u
        return new_theta, n_shrink



    #def step(self) -> None:
    #    # 1) omega | eta
    #    self.resample_omega()

    #    # 2) Zmix | omega, y, phi, mu, r
    #    self.Zmix = sample_Zmix_given_omega(
    #        self.rng,
    #        self.Lblk,
    #        self.omega,
    #        self.kappa,
    #        self.phi,
    #        self.mu,
    #        self.r,
    #        self.v_L1,
    #    )

    #    # 3) r | Zmix, phi (exact)
    #    prob_plus = sigmoid_stable(self.Zmix + self.phi)
    #    u = self.rng.uniform(size=self.Lblk.M)
    #    self.r = np.where(u < prob_plus, 1.0, -1.0).astype(np.float64)

    #    # 4) phi (hence p) | rest  (slice)
    #    self.phi = float(slice_sample_1d(
    #        self.rng, self.phi, self.logpost_phi_given_rest,
    #        w=self.cfg.slice_w_phi, m=self.cfg.slice_m_phi
    #    ))

    #    # 5) theta=log(mu) | rest (slice)
    #    self.theta = float(slice_sample_1d(
    #        self.rng, self.theta, self.logpost_theta_given_rest,
    #        w=self.cfg.slice_w_theta, m=self.cfg.slice_m_theta
    #    ))
    #    self.mu = float(np.exp(self.theta))

    def run(self) -> Dict[str, np.ndarray]:
        for _ in range(int(self.cfg.n_warmup)):
            self.step()

        mu = np.empty(int(self.cfg.n_samples), dtype=np.float64)
        p = np.empty(int(self.cfg.n_samples), dtype=np.float64)

        for t in range(int(self.cfg.n_samples)):
            self.step()
            mu[t] = self.mu
            p[t] = self.current_p()

        return {"mu": mu, "p": p}



def blockdiag_from_groupedblocks(gb) -> "BlockDiagL":
    """
    Convert your GroupedBlocks into the BlockDiagL used by the LRDiscreteBlockdiagPGGibbs sampler.

    Assumes gb.idx_by_size[s] is a flat array of length n_s*s, block-concatenated.
    """
    blocks_by_size = {}
    idx_by_size = {}

    for s in gb.sizes:
        Ls = np.asarray(gb.L_by_size[s], dtype=np.float64)           # (n_s, s, s)
        idx_flat = np.asarray(gb.idx_by_size[s], dtype=np.int64)     # (n_s*s,)
        n_s = int(Ls.shape[0])
        idx = idx_flat.reshape(n_s, s)                               # (n_s, s)

        blocks_by_size[int(s)] = Ls
        idx_by_size[int(s)] = idx

    return BlockDiagL(
        M=int(gb.M),
        sizes=tuple(int(s) for s in gb.sizes),
        blocks_by_size=blocks_by_size,
        idx_by_size=idx_by_size,
    )



def run_one_chain_preloaded_lrpg(
    chain_id: int,
    gb,
    y_perm: np.ndarray,
    seed: int,
    cfg,                          # your existing ChainConfig (used only for a few fields)
    true_mu: float = float("nan"),
    p_obs_override: float | None = None,
) -> dict:
    """
    Preloaded chain runner for the lr_discrete_blockdiag-equivalent PG-Gibbs sampler.

    Parameters mirror your old run_one_chain_preloaded:
      (chain_id, gb, y_perm, seed, cfg, true_mu, p_obs_override)

    Notes
    -----
    - We interpret p_obs_override as the "p_obs" used to centre the Beta prior on p.
      If None, we default to mean(y_perm).
    - 'cfg' is your ChainConfig; we read:
        n_warmup, n_samples, prior_scale, slice_w, slice_m
      and map them onto LRBlockdiagPGConfig.
    """
    rng = np.random.default_rng(seed)

    y_perm = np.asarray(y_perm, dtype=np.float64)
    kappa_ok = True  # just a reminder that the sampler internally uses y-0.5

    # Build block-diagonal operator
    Lblk = blockdiag_from_groupedblocks(gb)

    # p_obs used in Beta prior
    if p_obs_override is None:
        p_obs = float(np.mean(y_perm))
    else:
        p_obs = float(p_obs_override)

    # Map ChainConfig -> LRBlockdiagPGConfig
    lr_cfg = LRBlockdiagPGConfig(
        n_warmup=int(cfg.n_warmup),
        n_samples=int(cfg.n_samples),
        K_beta=20.0,                           # matches your numpyro model
        p_obs=p_obs,
        mu_prior_scale=float(cfg.prior_scale),
        mu_prior_df=float(getattr(cfg, "mu_prior_df", 10.0)),
        slice_w_phi=float(cfg.slice_w),
        slice_m_phi=int(cfg.slice_m),
        slice_w_theta=float(cfg.slice_w),
        slice_m_theta=int(cfg.slice_m),
    )

    sampler = LRDiscreteBlockdiagPGGibbs(
        rng=rng,
        Lblk=Lblk,
        y=y_perm,
        cfg=lr_cfg,
        # you can optionally set different inits per chain:
        phi_init=None,
        mu_init=float(np.exp(getattr(cfg, "prior_loc", 0.0))),  # default mu_init = exp(prior_loc)
    )

    # Run and record
    n_w = lr_cfg.n_warmup
    n_s = lr_cfg.n_samples

    # warmup — record mu trace for convergence diagnostics
    mu_warmup = np.empty(n_w, dtype=np.float64)
    for t in range(n_w):
        sampler.step()
        mu_warmup[t] = sampler.mu

    mu = np.empty(n_s, dtype=np.float64)
    beta0 = np.empty(n_s, dtype=np.float64)  # == phi
    p = np.empty(n_s, dtype=np.float64)
    ll = np.empty(n_s, dtype=np.float64)

    for t in range(n_s):
        sampler.step()
        mu[t] = sampler.mu
        beta0[t] = sampler.phi
        p[t] = sampler.current_p()
        eta = sampler.current_eta()
        ll[t] = bernoulli_loglik_logit_np(eta, y_perm)

    return {
        "chain_id": int(chain_id),
        "true_mu": float(true_mu),
        "mu": mu,
        "beta0": beta0,
        "p": p,
        "ll": ll,
        "mu_warmup":         mu_warmup,
        "mean_theta_shrink": sampler._theta_shrink_total / max(sampler._step_count, 1),
        "mean_theta_step":   sampler._theta_step_total   / max(sampler._step_count, 1),
    }


import numpy as np

def bernoulli_loglik_logit_np(eta: np.ndarray, y: np.ndarray) -> float:
    """
    Sum_i log Bernoulli(y_i | sigmoid(eta_i)) computed stably:
      y*eta - log(1+exp(eta))
    """
    eta = np.asarray(eta, dtype=np.float64)
    y   = np.asarray(y, dtype=np.float64)
    softplus = np.log1p(np.exp(-np.abs(eta))) + np.maximum(eta, 0.0)
    return float(np.sum(y * eta - softplus))



def _worker_preloaded_lrpg(args):
    """
    Accepts the same job tuple shapes as your earlier _worker_preloaded:
      (chain_id, gb, y_perm, seed, cfg, true_mu)
      (chain_id, gb, y_perm, seed, cfg, true_mu, p_obs_override)
      (chain_id, gb, y_perm, seed, cfg, true_mu, p_obs_override, ...)
    """
    chain_id, gb, y_perm, seed, cfg, true_mu = args[:6]
    p_obs_override = args[6] if len(args) > 6 else None
    return run_one_chain_preloaded_lrpg(
        chain_id=chain_id,
        gb=gb,
        y_perm=y_perm,
        seed=seed,
        cfg=cfg,
        true_mu=true_mu,
        p_obs_override=p_obs_override,
    )



import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class MuCollapsedCache:
    eigvals_flat: np.ndarray   # concatenated eigenvalues of K = L^T Ω L per block
    p0_flat: np.ndarray        # concatenated U^T b0  (b0 is mu-independent part of RHS)
    p1_flat: np.ndarray        # concatenated U^T b1  (multiplied by mu*mbar inside logpost)
    M: int

    # constants for the mu-dependent "offset-only" term:
    # sum_i [ kappa_i*offset_i - 0.5*omega_i*offset_i^2 ]
    # where offset_i = phi - mu*mbar*v_i.
    c0: float
    c1: float
    c2: float

    mbar: float               # tanh(phi/2)
    mu_prior_scale: float     # half-Student-t scale
    mu_prior_df: float        # degrees of freedom

    # ── extra fields for collapsed phi update ──────────────────────────────
    # h(phi) in eigenbasis = A_flat - phi*B_flat + mu*mbar(phi)*p1_flat
    A_flat: np.ndarray        # U^T*(L^T*kappa + 0.5*r)  [phi-independent]
    B_flat: np.ndarray        # U^T*(L^T*omega)          [phi-independent]
    tau: float                # 1/(2*mu)
    mu: float                 # current mu value

    # scalar sums for the offset-only terms as a function of phi
    sum_kappa:   float        # sum(kappa)
    sum_omega:   float        # sum(omega)
    sum_kappa_v: float        # sum(kappa * v)
    sum_omega_v: float        # sum(omega * v)
    sum_omega_v2: float       # sum(omega * v^2)

    # prior parameters for p = sigmoid(phi)
    K_beta: float             # Beta prior concentration
    p_obs:  float             # prior centre (observed case fraction)
    n_plus: int               # sum(r > 0)
    n_minus: int              # sum(r < 0)


def build_mu_collapsed_cache(
    Lblk,
    omega: np.ndarray,
    kappa: np.ndarray,
    r: np.ndarray,
    phi: float,
    v_L1: np.ndarray,          # v = L @ 1
    mu_prior_scale: float,
    mu_prior_df: float = 10.0,
    K_beta: float = 20.0,
    p_obs: float = 0.5,
    mu: float = 1.0,
) -> MuCollapsedCache:
    """
    Build cache for collapsed mu and phi updates: integrates out Zmix.

    Model pieces:
      eta = offset + L @ Zmix
      offset = phi - (mu*mbar) * v,  mbar = tanh(phi/2)
      Prior: Zmix | r,mu ~ N(mu*r, 2mu I)  -> contributes precision tau I and linear term +0.5 r
      PG: contributes K = L^T Ω L and linear term L^T (kappa - omega*offset)

    The same eigendecomposition of K supports both:
      - collapsed theta update: p0, p1, c0, c1, c2 (phi fixed)
      - collapsed phi update:   A, B, p1, scalar sums (mu/tau fixed)
    """
    omega = np.asarray(omega, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    r     = np.asarray(r, dtype=np.float64)
    v     = np.asarray(v_L1, dtype=np.float64)
    M     = int(omega.size)

    mbar = float(np.tanh(0.5 * phi))
    tau  = 0.5 / float(mu) if mu > 0.0 else np.inf

    # Offset-only terms for theta update (c0 + mu*c1 + mu^2*c2 evaluated at current phi):
    c0 = float(np.sum(kappa * phi - 0.5 * omega * (phi * phi)))
    c1 = float(mbar * np.sum(omega * phi * v - kappa * v))
    c2 = float(-0.5 * (mbar * mbar) * np.sum(omega * (v * v)))

    # Scalar sums for collapsed phi update (phi-independent given omega, v, kappa):
    sum_kappa   = float(np.sum(kappa))
    sum_omega   = float(np.sum(omega))
    sum_kappa_v = float(np.sum(kappa * v))
    sum_omega_v = float(np.sum(omega * v))
    sum_omega_v2 = float(np.sum(omega * v * v))

    eigvals_parts = []
    p0_parts = []
    p1_parts = []
    A_parts  = []
    B_parts  = []

    for s in Lblk.sizes:
        Ls  = Lblk.blocks_by_size[s]      # (n_s,s,s)
        idx = Lblk.idx_by_size[s]         # (n_s,s)
        n_s = int(Ls.shape[0])

        og = omega[idx]                   # (n_s,s)
        kg = kappa[idx]                   # (n_s,s)
        rg = r[idx]                       # (n_s,s)
        vg = v[idx]                       # (n_s,s)

        # mu-independent part of b_like: (kappa - omega*phi)
        b_like0 = kg - og * phi

        # b0 = L^T b_like0 + 0.5 r  (for theta update, depends on current phi)
        b0 = np.einsum("nij,ni->nj", Ls, b_like0) + 0.5 * rg   # (n_s,s)

        # b1 = L^T (omega*v)  (mu*mbar coefficient, shared by both updates)
        b1 = np.einsum("nij,ni->nj", Ls, og * vg)              # (n_s,s)

        # b_A = L^T*kappa + 0.5*r  (phi=0 baseline of h, for phi update)
        b_A = np.einsum("nij,ni->nj", Ls, kg) + 0.5 * rg       # (n_s,s)

        # b_B = L^T*omega  (phi linear coefficient in h, for phi update)
        b_B = np.einsum("nij,ni->nj", Ls, og)                  # (n_s,s)

        # K = L^T diag(omega) L (blockwise)
        W = Ls * np.sqrt(og)[:, :, None]                       # (n_s,s,s)
        K = np.einsum("nki,nkj->nij", W, W)                    # (n_s,s,s)

        for b in range(n_s):
            lam, U = np.linalg.eigh(K[b])
            eigvals_parts.append(lam)
            p0_parts.append(U.T @ b0[b])
            p1_parts.append(U.T @ b1[b])
            A_parts.append(U.T @ b_A[b])
            B_parts.append(U.T @ b_B[b])

    empty = np.empty(0)
    return MuCollapsedCache(
        eigvals_flat=np.concatenate(eigvals_parts) if eigvals_parts else empty,
        p0_flat=np.concatenate(p0_parts) if p0_parts else empty,
        p1_flat=np.concatenate(p1_parts) if p1_parts else empty,
        M=M,
        c0=c0, c1=c1, c2=c2,
        mbar=mbar,
        mu_prior_scale=float(mu_prior_scale),
        mu_prior_df=float(mu_prior_df),
        A_flat=np.concatenate(A_parts) if A_parts else empty,
        B_flat=np.concatenate(B_parts) if B_parts else empty,
        tau=tau,
        mu=float(mu),
        sum_kappa=sum_kappa,
        sum_omega=sum_omega,
        sum_kappa_v=sum_kappa_v,
        sum_omega_v=sum_omega_v,
        sum_omega_v2=sum_omega_v2,
        K_beta=float(K_beta),
        p_obs=float(p_obs),
        n_plus=int(np.sum(r > 0)),
        n_minus=int(np.sum(r < 0)),
    )




def logpost_theta_mu_collapsed(theta: float, cache: MuCollapsedCache,
                               aux_v: float = 0.0) -> float:
    # Guard exp under/overflow
    if theta < -745.0 or theta > 709.0:
        return -np.inf
    mu = float(np.exp(theta))
    if not np.isfinite(mu) or mu <= 0.0:
        return -np.inf

    # tau = 1/(2mu)
    tau = 0.5 / mu

    lam = cache.eigvals_flat
    lam_tau = lam + tau

    # b(mu) in eigenbasis: p0 + (mu*mbar)*p1
    u = mu * cache.mbar
    proj = cache.p0_flat + u * cache.p1_flat

    quad = float(np.sum((proj * proj) / lam_tau))
    logdet = float(np.sum(np.log(lam_tau)))

    # offset-only term (quadratic in mu)
    const = cache.c0 + mu * cache.c1 + (mu * mu) * cache.c2

    M = float(cache.M)

    # Prior normaliser terms from Zmix|mu: (2mu)^(-M/2) * exp(-M*mu/4)
    lp = (const
          + 0.5 * quad
          - 0.5 * logdet
          - 0.25 * M * mu
          - 0.5 * M * np.log(2.0 * mu))

    # Prior on mu + Jacobian for theta = log(mu)
    if aux_v > 0.0:
        # Conditional prior: HalfNormal(0, aux_v)  [used with auxiliary variable]
        lp += theta - mu ** 2 / (2.0 * aux_v)
    else:
        # Half-Student-t(df, scale) prior (default)
        s  = float(cache.mu_prior_scale)
        df = float(cache.mu_prior_df)
        lp += theta - 0.5 * (df + 1.0) * np.log(1.0 + (mu / s) ** 2 / df)

    return float(lp)


def logpost_phi_collapsed(phi: float, cache: MuCollapsedCache) -> float:
    """
    Log-posterior of phi with Zmix integrated out (collapsed).

    Uses the cached eigendecomposition of K = L^T Ω L.  The collapsed
    likelihood is:

        offset_terms(phi) + 0.5 * Σ_j h_j(phi)² / (λ_j + τ)

    where h(phi) = A_flat - phi·B_flat + μ·m̄(phi)·p1_flat in the eigenbasis
    and m̄ = tanh(phi/2).  The log-det term −0.5·Σ log(λ+τ) is phi-independent
    and is omitted (cancels in acceptance ratio / is absorbed in the normaliser).
    """
    mbar = float(np.tanh(0.5 * phi))
    mu   = cache.mu
    tau  = cache.tau
    u    = mu * mbar                         # = μ·m̄

    # Offset-only terms: Σ_i [κ_i·offset_i − 0.5·ω_i·offset_i²]
    # offset_i = phi − u·v_i
    ofs_lin  = phi * cache.sum_kappa - u * cache.sum_kappa_v
    ofs_quad = -0.5 * (phi * phi * cache.sum_omega
                       - 2.0 * phi * u * cache.sum_omega_v
                       + u * u * cache.sum_omega_v2)

    # Eigenbasis projection h(phi) = A_flat − phi·B_flat + u·p1_flat
    h       = cache.A_flat - phi * cache.B_flat + u * cache.p1_flat
    lam_tau = cache.eigvals_flat + tau
    quad    = float(np.sum(h * h / lam_tau))

    # Prior on p = sigmoid(phi): Beta + r likelihood + Jacobian (same as phi|rest)
    a      = float(cache.K_beta * cache.p_obs)
    b      = float(cache.K_beta * (1.0 - cache.p_obs))
    logp   = log_sigmoid(phi)
    log1mp = log1m_sigmoid(phi)

    lp_beta = (a - 1.0) * logp + (b - 1.0) * log1mp
    lp_r    = float(cache.n_plus) * logp + float(cache.n_minus) * log1mp
    lp_jac  = logp + log1mp    # Jacobian dp/dphi = p(1−p)

    return ofs_lin + ofs_quad + 0.5 * quad + lp_beta + lp_r + lp_jac


def update_cache_for_new_phi(cache: MuCollapsedCache, phi_new: float) -> MuCollapsedCache:
    """
    Return an updated MuCollapsedCache after phi changes.

    Reuses the stored eigendecompositions (eigvals_flat, A_flat, B_flat, p1_flat)
    and recomputes only the phi-dependent fields (p0_flat, mbar, c0, c1, c2)
    from the cached scalar sums.  Called once per step when use_collapsed_phi=True,
    between the phi slice and the theta slice.
    """
    mbar_new    = float(np.tanh(0.5 * phi_new))
    p0_flat_new = cache.A_flat - phi_new * cache.B_flat

    c0_new = float(phi_new * cache.sum_kappa
                   - 0.5 * phi_new * phi_new * cache.sum_omega)
    c1_new = float(mbar_new * (phi_new * cache.sum_omega_v - cache.sum_kappa_v))
    c2_new = float(-0.5 * mbar_new * mbar_new * cache.sum_omega_v2)

    return MuCollapsedCache(
        eigvals_flat=cache.eigvals_flat,
        p0_flat=p0_flat_new,
        p1_flat=cache.p1_flat,
        M=cache.M,
        c0=c0_new, c1=c1_new, c2=c2_new,
        mbar=mbar_new,
        mu_prior_scale=cache.mu_prior_scale,
        mu_prior_df=cache.mu_prior_df,
        A_flat=cache.A_flat,
        B_flat=cache.B_flat,
        tau=cache.tau,
        mu=cache.mu,
        sum_kappa=cache.sum_kappa,
        sum_omega=cache.sum_omega,
        sum_kappa_v=cache.sum_kappa_v,
        sum_omega_v=cache.sum_omega_v,
        sum_omega_v2=cache.sum_omega_v2,
        K_beta=cache.K_beta,
        p_obs=cache.p_obs,
        n_plus=cache.n_plus,
        n_minus=cache.n_minus,
    )


# ----------------------------- example ------------------------------------
if __name__ == "__main__":
    # Example stub: you must supply L already permuted into block-diagonal order
    # and the slices describing its diagonal blocks.
    rng = np.random.default_rng(0)

    # Fake tiny example with two blocks: size-1 and size-2
    L = np.zeros((3, 3))
    L[0, 0] = 1.0
    L[1:3, 1:3] = np.array([[1.0, 0.0], [0.2, 0.98]])
    block_slices = [slice(0, 1), slice(1, 3)]
    Lblk = BlockDiagL.from_slices(L, block_slices)

    y = rng.binomial(1, 0.5, size=3).astype(np.float64)

    cfg = SamplerConfig(n_warmup=200, n_samples=500, beta0_sd=5.0, mu_prior_scale=5.0)
    sampler = PGGibbsBlockSampler(rng, Lblk, y, cfg, p_mix_prior_conc=2.0, p0_init=0.5)
    out = sampler.run()
    print("mu median:", out["mu"].shape)
    print("mu median:", np.median(out["mu"]))

