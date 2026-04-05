"""
geninfo.py — Bayesian inference of genetic information for discrimination.

Public API
----------
sample_posterior(L, y, ...)
    Run multi-chain Polya-Gamma Gibbs sampling for mu (genetic information,
    in nats) given the Cholesky factor L of a genetic relationship matrix and
    binary case/control outcomes y.

summarize_and_plot(result, ...)
    Print posterior summary statistics and produce a figure showing the prior,
    posterior, and likelihood for mu.

compress(L, y, posterior_result, outfile)
    Save a privacy-preserving compressed model (block L matrices + posterior
    samples) to a .npz file for offline use.

generate_synthetic(model_file, seed)
    Draw synthetic (L_syn, y_syn) from the posterior predictive stored in a
    compressed model file.

validate(L_real, y_real, model_file, ...)
    Compare real and synthetic datasets on block sizes, concordance,
    prevalence, and posterior recovery.

test_privacy(L, y, model_file, ...)
    Membership inference test: verify that LOO removal of any individual
    changes predictions by less than a threshold.

Model
-----
  y_i ~ Bernoulli(sigmoid(phi + (L z)_i))
  r_i in {-1, +1},  P(r_i=+1) = p  (discrete class indicator)
  z | r, mu ~ N(mu * r, 2*mu * I)
  p ~ Beta(K_beta * p_obs, K_beta * (1 - p_obs))
  mu ~ half-Student-t(df=mu_prior_df, scale=mu_prior_scale)   [df=1 → half-Cauchy]

The key quantity mu equals Lambda (expected log-likelihood ratio in nats)
and also equals log(lambda_S) where lambda_S is the sibling recurrence
risk ratio.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue_module
import sys
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde, ks_2samp
from scipy.sparse.csgraph import connected_components as _scipy_connected_components
from threadpoolctl import threadpool_limits


def _set_blas_threads(n: int) -> None:
    """
    Set BLAS/LAPACK thread count in the current process.

    Tries four methods so this works for both OpenBLAS and MKL, including
    inside forked subprocesses where threadpoolctl's dl_iterate_phdr /
    RTLD_NOLOAD library discovery sometimes fails:

    1. Set BLAS env vars — MKL re-reads MKL_NUM_THREADS when it (re-)creates
       its thread pool after fork.
    2. MKL: load libmkl_rt.so directly via ctypes (without RTLD_NOLOAD, which
       avoids the path-resolution failure that threadpoolctl hits in forked
       processes), then call MKL_Set_Num_Threads.
    3. OpenBLAS: call openblas_set_num_threads via the global symbol table
       (ctypes.CDLL(None)) and also try mkl-service if available.
    4. threadpoolctl.threadpool_limits() as a final fallback, with
       dl_iterate_phdr warnings suppressed.
    """
    import ctypes
    # 1. Env vars
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    # 2. MKL: try loading libmkl_rt directly (no RTLD_NOLOAD, so this works
    #    even when dl_iterate_phdr paths are stale in forked workers).
    #    First try names that rely on LD_LIBRARY_PATH / ldconfig; then scan
    #    /proc/self/maps to find the exact path of the already-loaded library.
    def _try_mkl_set(lib):
        try:
            lib.MKL_Set_Num_Threads(ctypes.c_int(n))
            return True
        except Exception:
            return False

    _mkl_done = False
    for _mkl_name in ("libmkl_rt.so", "libmkl_rt.so.2", "libmkl_rt.so.1",
                      "mkl_rt"):
        try:
            if _try_mkl_set(ctypes.CDLL(_mkl_name)):
                _mkl_done = True
                break
        except Exception:
            pass

    if not _mkl_done:
        # Fall back to scanning /proc/self/maps for the loaded MKL path
        try:
            import re as _re
            with open("/proc/self/maps") as _f:
                for _line in _f:
                    _m = _re.search(r'(/[^\s]*libmkl_rt[^\s]*\.so[0-9.]*)', _line)
                    if _m:
                        _path = _m.group(1)
                        try:
                            if _try_mkl_set(ctypes.CDLL(_path)):
                                break
                        except Exception:
                            pass
        except Exception:
            pass
    # 3a. OpenBLAS via global symbol table
    try:
        ctypes.CDLL(None).openblas_set_num_threads(ctypes.c_int(n))
    except Exception:
        pass
    # 3b. mkl-service Python package (conda)
    try:
        import mkl as _mkl
        _mkl.set_num_threads(n)
    except Exception:
        pass
    # 4. threadpoolctl fallback (covers scipy_openblas, Accelerate, etc.)
    _old = sys.unraisablehook
    sys.unraisablehook = lambda _args: None
    try:
        threadpool_limits(limits=n, user_api="blas")
    finally:
        sys.unraisablehook = _old


def _pool_init_blas(n_threads: int) -> None:
    """Pool initializer: set BLAS threads in each worker right after fork."""
    _set_blas_threads(n_threads)

# Limit BLAS threads to 1 at import time.  numpy imports MKL/OpenBLAS before
# this point, so os.environ.setdefault() would only set env vars — it cannot
# change the already-running BLAS thread pool.  Calling _set_blas_threads(1)
# directly limits the pool immediately.  Workers override this via
# _pool_init_blas(n_blas_threads) in the Pool initialiser.
_set_blas_threads(1)

# Module-level queue used by _worker_with_queue.  Set in the parent process
# before Pool creation so that forked workers inherit it without pickling.
_POOL_PROGRESS_Q = None

from tqdm.auto import tqdm

from polyagamma_gibbs import infer_blocks_from_L, BlockStructure, ChainConfig
from pg_gibbs_vectorized import GroupedBlocks
from pg_gibbs_clean import (
    _worker_preloaded_lrpg,
    LRBlockdiagPGConfig,
    LRDiscreteBlockdiagPGGibbs,
    blockdiag_from_groupedblocks,
    bernoulli_loglik_logit_np,
)


# ─── private helpers ─────────────────────────────────────────────────────────

def _blocks_from_corr_threshold(
    A: np.ndarray,
    corr_threshold: float,
) -> tuple[np.ndarray, list[slice]]:
    """
    Find connected components of individuals with |A[i,j]| > corr_threshold.

    Uses the same adjacency construction as grm_functions.block_diagonal_split
    but returns individual blocks (one per connected component) rather than
    packing them into groups, as required by GroupedBlocks.

    Returns
    -------
    perm : np.ndarray  permutation of row/column indices
    slices : list[slice]  one slice per connected component in permuted order
    """
    n = A.shape[0]
    adjacency = (np.abs(A) > corr_threshold) & (~np.eye(n, dtype=bool))
    _, labels = _scipy_connected_components(adjacency, directed=False)

    components: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        components[lbl].append(idx)
    blocks = sorted(components.values(), key=lambda b: min(b))

    perm = np.array([idx for block in blocks for idx in block], dtype=np.int64)

    slices: list[slice] = []
    offset = 0
    for block in blocks:
        slices.append(slice(offset, offset + len(block)))
        offset += len(block)

    return perm, slices


def _build_block_structure(
    L: np.ndarray,
    y: np.ndarray,
    corr_threshold: float = 0.0,
) -> tuple[GroupedBlocks, np.ndarray, dict]:
    """
    Permute L and y into block-diagonal order and return GroupedBlocks.

    Parameters
    ----------
    L : (M, M) Cholesky factor
    y : (M,) binary outcomes
    corr_threshold : float
        Absolute correlation threshold.  Pairs with |A[i,j]| <= corr_threshold
        are treated as unrelated, allowing approximately block-diagonal GRMs to
        be split into separate families.  When 0.0 (default), only exact zeros
        in L are used to identify blocks (suitable when L is exactly
        block-diagonal).  When > 0, A = L @ L.T is computed and
        connected_components is called on the thresholded adjacency matrix
        (approach from grm_functions.block_diagonal_split).  This requires
        O(M^2) memory for A.

    Returns
    -------
    gb : GroupedBlocks
    y_perm : np.ndarray  (y reordered to match gb)
    block_info : dict with keys M, n_blocks, sizes_summary, corr_threshold
    """
    L_np = np.asarray(L, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)

    if corr_threshold > 0.0:
        A = L_np @ L_np.T
        perm, slices = _blocks_from_corr_threshold(A, corr_threshold)
        L_perm = L_np[np.ix_(perm, perm)]
        y_perm = y_np[perm]
        bs = BlockStructure(L_perm, slices)
    else:
        perm_rows, perm_cols, row_slices, col_slices = infer_blocks_from_L(L_np, tol_rel=0.0)
        L_perm = L_np[perm_rows, :][:, perm_cols]
        y_perm = y_np[perm_rows]
        bs = BlockStructure(L_perm, col_slices, row_slices)

    gb = GroupedBlocks.from_block_structure(bs)

    sizes_summary = {s: int(gb.L_by_size[s].shape[0]) for s in gb.sizes}
    block_info = {
        "M": int(gb.M),
        "n_blocks": int(bs.n_blocks),
        "sizes_summary": sizes_summary,
        "corr_threshold": corr_threshold,
    }
    return gb, y_perm, block_info


def _print_block_info(block_info: dict) -> None:
    bi = block_info
    max_size = max(bi["sizes_summary"].keys()) if bi["sizes_summary"] else 0
    n_rel = sum(s * n for s, n in bi["sizes_summary"].items() if s >= 2)
    size_str = "  ".join(
        f"size-{s}: {n}" for s, n in sorted(bi["sizes_summary"].items())
    )
    thr = bi.get("corr_threshold", 0.0)
    thr_str = f"  corr_threshold={thr}" if thr > 0.0 else ""
    print(f"Block structure: M={bi['M']}  n_blocks={bi['n_blocks']}  "
          f"largest={max_size}  relatives(size≥2)={n_rel}{thr_str}")
    print(f"  {size_str}")


def _make_chain_config(
    n_warmup: int,
    n_samples: int,
    mu_prior_scale: float,
    p_prior_conc: float,
    mu_prior_df: float = 10.0,
    slice_w: float = 1.5,
) -> ChainConfig:
    return ChainConfig(
        n_warmup=n_warmup,
        n_samples=n_samples,
        prior_loc=0.0,
        prior_scale=mu_prior_scale,
        mu_prior_df=mu_prior_df,
        beta0_sd=5.0,            # not used by LRDiscreteBlockdiagPGGibbs
        slice_w=slice_w,
        slice_m=20,
        slice_max_steps=250,
        inner_latent_cycles=2,
        learn_p=True,
        p_prior_conc=p_prior_conc,
        project_affects_beta0=False,
        diag_every=n_warmup + n_samples + 1,  # suppress per-step diagnostics
    )


def _half_student_t_pdf(x: np.ndarray, scale: float = 1.0, df: float = 10.0) -> np.ndarray:
    """Unnormalised half-Student-t pdf (positive half only, not normalised to 1)."""
    x = np.asarray(x, dtype=np.float64)
    return (1.0 + (x / scale) ** 2 / df) ** (-0.5 * (df + 1.0))


# ─── progress-bar worker ─────────────────────────────────────────────────────

def _worker_nobar(args):
    """No-progress-bar worker (BLAS threads already set in parent via fork)."""
    return _worker_preloaded_lrpg(args)


def _worker_with_queue(args):
    """
    PG-Gibbs worker that sends progress updates to _POOL_PROGRESS_Q (a
    module-level multiprocessing.Queue set in the parent before Pool creation
    and inherited by forked workers without pickling).

    args: (chain_id, gb, y_perm, seed, cfg, true_mu, p_obs_override)

    Messages put on queue:
      ("step",  chain_id, delta, mu)  — advance bar by delta steps
      ("phase", chain_id, label)      — change bar description
      ("done",  chain_id)             — chain finished
    """
    chain_id, gb, y_perm, seed, cfg = args[:5]
    true_mu = args[5] if len(args) > 5 else float("nan")
    p_obs_override = args[6] if len(args) > 6 else None
    jags_adapt          = args[7] if len(args) > 7 else False
    n_warmup_phase1     = int(args[8]) if len(args) > 8 else 0
    use_asis            = bool(args[9]) if len(args) > 9 else False
    use_alpha_reparam   = bool(args[10]) if len(args) > 10 else False
    q = _POOL_PROGRESS_Q

    rng = np.random.default_rng(seed)
    y_perm = np.asarray(y_perm, dtype=np.float64)
    Lblk = blockdiag_from_groupedblocks(gb)

    p_obs = float(p_obs_override) if p_obs_override is not None else float(np.mean(y_perm))

    lr_cfg = LRBlockdiagPGConfig(
        n_warmup=int(cfg.n_warmup),
        n_samples=int(cfg.n_samples),
        K_beta=20.0,
        p_obs=p_obs,
        mu_prior_scale=float(cfg.prior_scale),
        mu_prior_df=float(getattr(cfg, "mu_prior_df", 10.0)),
        slice_w_phi=float(cfg.slice_w),
        slice_m_phi=int(cfg.slice_m),
        slice_w_theta=float(cfg.slice_w),
        slice_m_theta=int(cfg.slice_m),
        use_alpha_reparam=use_alpha_reparam,
    )

    sampler = LRDiscreteBlockdiagPGGibbs(
        rng=rng,
        Lblk=Lblk,
        y=y_perm,
        cfg=lr_cfg,
        phi_init=None,
        mu_init=float(np.exp(getattr(cfg, "prior_loc", 0.0))),
    )

    n_w = lr_cfg.n_warmup
    n_s = lr_cfg.n_samples
    UPDATE_EVERY = 50

    # Two-phase warmup when jags_adapt=True and n_warmup_phase1 > 0:
    #   Phase 1 (iterations 0..n_warmup_phase1-1): fixed slice_w, no adaptation.
    #             Brings the chain close to stationarity before adaptation begins.
    #   Phase 2 (iterations n_warmup_phase1..n_w-1): JAGS adaptation.
    #             Adapts slice widths using linearly-weighted mean of |displacement|
    #             (JAGS 4.x Slicer.cc: w = 2 * sumdiff / iter / (iter-1)).
    # When jags_adapt=False, the single loop runs as before with no adaptation.
    _JAGS_MIN_ADAPT = 50
    sumdiff_theta = 0.0
    sumdiff_phi   = 0.0
    n_adapt       = 0

    asis_shrink_total = 0
    asis_step_total   = 0.0
    asis_count        = 0

    mu_warmup = np.empty(n_w, dtype=np.float64)
    ll_warmup = np.empty(n_w, dtype=np.float64)
    last = 0
    for i in range(n_w):
        theta_old = sampler.theta
        phi_old   = sampler.phi
        sampler.step()
        if use_asis:
            theta_pre_asis = sampler.theta
            _, n_shrink = sampler.step_asis()
            asis_shrink_total += n_shrink
            asis_step_total   += abs(sampler.theta - theta_pre_asis)
            asis_count        += 1
        mu_warmup[i] = sampler.mu
        ll_warmup[i] = bernoulli_loglik_logit_np(sampler.current_eta(), y_perm)

        if jags_adapt and i >= n_warmup_phase1:
            sumdiff_theta += n_adapt * abs(sampler.theta - theta_old)
            sumdiff_phi   += n_adapt * abs(sampler.phi   - phi_old)
            n_adapt += 1
            if n_adapt > _JAGS_MIN_ADAPT:
                lr_cfg.slice_w_theta = 2.0 * sumdiff_theta / n_adapt / (n_adapt - 1)
                lr_cfg.slice_w_phi   = 2.0 * sumdiff_phi   / n_adapt / (n_adapt - 1)

        if (i + 1) % UPDATE_EVERY == 0 or i == n_w - 1:
            delta = (i + 1) - last
            last = i + 1
            q.put(("step", chain_id, delta, sampler.mu))

    q.put(("phase", chain_id, "sample"))

    mu = np.empty(n_s, dtype=np.float64)
    beta0 = np.empty(n_s, dtype=np.float64)
    p = np.empty(n_s, dtype=np.float64)
    ll = np.empty(n_s, dtype=np.float64)

    last = 0
    for t in range(n_s):
        sampler.step()
        if use_asis:
            theta_pre_asis = sampler.theta
            _, n_shrink = sampler.step_asis()
            asis_shrink_total += n_shrink
            asis_step_total   += abs(sampler.theta - theta_pre_asis)
            asis_count        += 1
        mu[t] = sampler.mu
        beta0[t] = sampler.phi
        p[t] = sampler.current_p()
        ll[t] = bernoulli_loglik_logit_np(sampler.current_eta(), y_perm)
        if (t + 1) % UPDATE_EVERY == 0 or t == n_s - 1:
            delta = (t + 1) - last
            last = t + 1
            q.put(("step", chain_id, delta, sampler.mu))

    q.put(("done", chain_id))

    n_steps = max(sampler._step_count, 1)
    return {
        "chain_id": int(chain_id),
        "true_mu": float(true_mu),
        "mu": mu,
        "beta0": beta0,
        "p": p,
        "ll": ll,
        "mu_warmup":         mu_warmup,
        "ll_warmup":         ll_warmup,
        "mean_theta_shrink":   sampler._theta_shrink_total / n_steps,
        "mean_theta_step":     sampler._theta_step_total   / n_steps,
        "adapted_slice_w_theta": float(lr_cfg.slice_w_theta),
        "adapted_slice_w_phi":   float(lr_cfg.slice_w_phi),
        "mean_asis_shrink":    asis_shrink_total / max(asis_count, 1),
        "mean_asis_step":      asis_step_total   / max(asis_count, 1),
    }


# ─── public API ──────────────────────────────────────────────────────────────

def build_blocks(
    L: np.ndarray,
    y: np.ndarray,
    *,
    corr_threshold: float = 0.0,
) -> dict:
    """
    Detect and report the block structure of the genetic relationship matrix.

    Call this before sample_posterior() to inspect the block decomposition and
    choose an appropriate corr_threshold before committing to a long MCMC run.
    The returned dict can be passed directly to sample_posterior() via the
    ``blocks`` argument to avoid rebuilding.

    Parameters
    ----------
    L : array of shape (M, M)
        Lower-triangular Cholesky factor of the genetic relationship matrix.
    y : array of shape (M,)
        Binary case/control outcomes (0 or 1).
    corr_threshold : float
        Absolute correlation threshold.  Pairs with |A[i,j]| <= corr_threshold
        are treated as unrelated.  Default 0.0 uses exact zeros in L only
        (suitable when L is exactly block-diagonal).  A small positive value
        (e.g. 0.05) handles GRMs with residual distant-relative correlations;
        requires computing A = L @ L.T (O(M^2) memory).

    Returns
    -------
    dict with keys:
      "gb"         : GroupedBlocks object (passed to sample_posterior)
      "y_perm"     : np.ndarray — y reordered to match gb
      "block_info" : dict — M, n_blocks, sizes_summary, corr_threshold
    """
    L = np.asarray(L, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    gb, y_perm, block_info = _build_block_structure(L, y, corr_threshold=corr_threshold)
    _print_block_info(block_info)
    return {"gb": gb, "y_perm": y_perm, "block_info": block_info}


def sample_posterior(
    L: np.ndarray,
    y: np.ndarray,
    *,
    blocks: dict | None = None,
    n_chains: int = 4,
    n_warmup: int = 1000,
    n_samples: int = 5000,
    mu_prior_scale: float = 1.0,
    p_prior_conc: float = 20.0,
    seed: int = 0,
    progress_bar: bool = True,
    corr_threshold: float = 0.0,
    n_blas_threads: int | None = None,
    mu_prior_df: float = 10.0,
    slice_w: float = 1.5,
    jags_adapt: bool = False,
    n_warmup_phase1: int = 0,
    use_asis: bool = False,
    use_alpha_reparam: bool = False,
) -> dict:
    """
    Sample the posterior distribution of mu (genetic information, nats) using
    a Polya-Gamma Gibbs sampler.

    Parameters
    ----------
    L : array of shape (M, M)
        Lower-triangular Cholesky factor of the genetic relationship matrix.
    y : array of shape (M,)
        Binary case/control outcomes (0 or 1).
    blocks : dict, optional
        Pre-built block structure from build_blocks().  If supplied, L, y and
        corr_threshold are ignored for block detection and the sampler uses
        the pre-built GroupedBlocks directly.  Recommended for large datasets
        where block construction is expensive.
    n_chains : int
        Number of independent MCMC chains (run in parallel via fork).
    n_warmup : int
        Number of warmup (burn-in) iterations discarded per chain.
    n_samples : int
        Number of posterior samples retained per chain.
    mu_prior_scale : float
        Scale of the half-Cauchy prior on mu.
    p_prior_conc : float
        Concentration K for the Beta(K*p_obs, K*(1-p_obs)) prior on the
        mixture probability p.  K=20 gives a weakly informative prior
        centred at the observed case rate.
    seed : int
        Base random seed; chain c uses seed+c.
    progress_bar : bool
        Show a tqdm progress bar for each chain (default True).
    corr_threshold : float
        Absolute correlation threshold for block detection (ignored when
        ``blocks`` is supplied).  Default 0.0 uses only exact zeros in L.
        A small positive value (e.g. 0.05) handles GRMs with residual
        distant-relative correlations; requires computing A = L @ L.T
        (O(M^2) additional memory).
    n_blas_threads : int, optional
        Number of BLAS threads allocated to each chain.  Defaults to 1.
        MKL and OpenBLAS Cholesky factorisation shows negligible speedup
        below roughly 5000×5000 (e.g. only ~11% faster at 8 threads for
        1500×1500 on MKL), so the default of 1 keeps all cores available
        for inter-chain parallelism.  Increase only if you have a single
        very large block and can spare threads from chain parallelism.
    mu_prior_df : float
        Degrees of freedom for the half-Student-t prior on mu.  Default 10.
        df=1 recovers the half-Cauchy.  Larger df gives lighter tails and
        stronger regularisation towards zero.

    Returns
    -------
    dict with keys:
      "mu_chains"   : np.ndarray (n_chains, n_samples) — per-chain samples
      "mu_all"      : np.ndarray (n_chains * n_samples,) — all samples pooled
      "chain_dicts" : list of per-chain result dicts (mu, beta0, p, ll, chain_id)
      "block_info"  : dict — M, n_blocks, sizes_summary, corr_threshold
      "cfg"         : ChainConfig used
      "mu_prior_scale" : float
    """
    if blocks is not None:
        gb = blocks["gb"]
        y_perm = blocks["y_perm"]
        block_info = blocks["block_info"]
        _print_block_info(block_info)
    else:
        L = np.asarray(L, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError(f"L must be a square 2-D array; got shape {L.shape}")
        M = L.shape[0]
        if y.shape != (M,):
            raise ValueError(f"y must have shape ({M},); got {y.shape}")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("y must contain only 0s and 1s")
        gb, y_perm, block_info = _build_block_structure(L, y, corr_threshold=corr_threshold)
        _print_block_info(block_info)

    cfg = _make_chain_config(n_warmup, n_samples, mu_prior_scale, p_prior_conc, mu_prior_df, slice_w)

    n_cpu = os.cpu_count() or 1
    if n_blas_threads is None:
        n_blas_threads = 1  # MKL/OpenBLAS Cholesky scales poorly below ~5000x5000
    print(f"  {n_chains} chain(s) × {n_blas_threads} BLAS thread(s)/chain "
          f"({n_chains * n_blas_threads} of {n_cpu} CPUs used)")
    _set_blas_threads(n_blas_threads)

    p_obs = float(np.mean(y_perm))

    ctx = mp.get_context("fork")
    if progress_bar:
        # Set the module-level queue BEFORE creating the Pool so that forked
        # workers inherit it directly (avoids pickling the Queue through the
        # pool task pipe, which can deadlock).
        global _POOL_PROGRESS_Q
        _POOL_PROGRESS_Q = ctx.Queue()

        jobs = [
            (cid, gb, y_perm, seed + cid, cfg, float("nan"), p_obs, jags_adapt, n_warmup_phase1, use_asis, use_alpha_reparam)
            for cid in range(n_chains)
        ]

        bars = [
            tqdm(total=n_warmup + n_samples, position=cid,
                 desc=f"chain {cid}  warmup", leave=True, dynamic_ncols=True,
                 mininterval=30)
            for cid in range(n_chains)
        ]

        with ctx.Pool(processes=n_chains,
                      initializer=_pool_init_blas,
                      initargs=(n_blas_threads,)) as pool:
            async_result = pool.map_async(_worker_with_queue, jobs)
            finished = 0
            while finished < n_chains:
                try:
                    msg = _POOL_PROGRESS_Q.get(timeout=1.0)
                    kind = msg[0]
                    cid = msg[1]
                    if kind == "step":
                        delta, mu_val = msg[2], msg[3]
                        bars[cid].update(delta)
                        bars[cid].set_postfix(mu=f"{mu_val:.3f}", refresh=False)
                    elif kind == "phase":
                        bars[cid].set_description(f"chain {cid}  {msg[2]}")
                    elif kind == "done":
                        finished += 1
                except _queue_module.Empty:
                    pass
            # Collect results inside the pool context so pool.terminate()
            # (called by __exit__) does not race with workers finalising
            # their return values.
            chain_dicts = async_result.get()
        for bar in bars:
            bar.close()
        print()  # newline after the stacked bars
    else:
        jobs = [
            (cid, gb, y_perm, seed + cid, cfg, float("nan"), p_obs, jags_adapt, n_warmup_phase1, use_asis, use_alpha_reparam)
            for cid in range(n_chains)
        ]
        with ctx.Pool(processes=n_chains,
                      initializer=_pool_init_blas,
                      initargs=(n_blas_threads,)) as pool:
            chain_dicts = pool.map(_worker_nobar, jobs)

    mu_chains = np.stack([c["mu"] for c in chain_dicts])   # (n_chains, n_samples)
    mu_all = mu_chains.ravel()

    return {
        "mu_chains": mu_chains,
        "mu_all": mu_all,
        "chain_dicts": chain_dicts,
        "block_info": block_info,
        "cfg": cfg,
        "mu_prior_scale": mu_prior_scale,
        "mu_prior_df": mu_prior_df,
        "slice_w": slice_w,
        "jags_adapt": jags_adapt,
        "use_asis": use_asis,
        "use_alpha_reparam": use_alpha_reparam,
    }


def summarize_and_plot(
    result: dict,
    *,
    mu_true: float | None = None,
    title: str | None = None,
    outfile: str | None = None,
    prior_scale: float | None = None,
) -> dict:
    """
    Print posterior summary statistics and plot prior, posterior, and
    likelihood for mu.

    Parameters
    ----------
    result : dict
        Output of sample_posterior().
    mu_true : float, optional
        True value of mu (for validation; drawn as a vertical line).
    title : str, optional
        Figure title.  Defaults to a short description.
    outfile : str, optional
        File path to save the figure (PNG recommended).  If None, calls
        plt.show() instead.
    prior_scale : float, optional
        Override the half-Cauchy prior scale used in the plot.  Defaults to
        result["mu_prior_scale"].

    Returns
    -------
    dict with keys: mean, median, sd, ci90_lo, ci90_hi, ess_bulk, r_hat
    """
    mu_chains = result["mu_chains"]          # (n_chains, n_samples)
    mu_all = result["mu_all"]
    scale = prior_scale if prior_scale is not None else result["mu_prior_scale"]
    df    = float(result.get("mu_prior_df", 10.0))

    # ── summary statistics ──────────────────────────────────────────────────
    idata = az.convert_to_inference_data({"mu": mu_chains})
    summ = az.summary(idata, var_names=["mu"])
    ess = float(summ["ess_bulk"].iloc[0])
    rhat = float(summ["r_hat"].iloc[0])

    mean_mu = float(np.mean(mu_all))
    median_mu = float(np.median(mu_all))
    sd_mu = float(np.std(mu_all))
    ci90_lo = float(np.percentile(mu_all, 5))
    ci90_hi = float(np.percentile(mu_all, 95))

    n_chains, n_samples = mu_chains.shape

    print(f"\nPG-Gibbs posterior summary for mu (genetic information, nats)")
    print(f"  Chains: {n_chains}    Samples/chain: {n_samples}")
    bi = result.get("block_info", {})
    if bi:
        print(f"  M={bi['M']}  n_blocks={bi['n_blocks']}  "
              + "  ".join(f"size-{s}:{bi['sizes_summary'][s]}" for s in sorted(bi['sizes_summary'])))
    print(f"  mean   = {mean_mu:.4f}")
    print(f"  median = {median_mu:.4f}")
    print(f"  sd     = {sd_mu:.4f}")
    print(f"  90% CI = [{ci90_lo:.4f}, {ci90_hi:.4f}]")
    print(f"  ESS_bulk = {ess:.0f}    R-hat = {rhat:.4f}")

    # ── plot ────────────────────────────────────────────────────────────────
    mu_max = max(4.0, float(np.percentile(mu_all, 99)) * 1.6)
    mu_grid = np.linspace(1e-4, mu_max, 600)

    prior_d = _half_student_t_pdf(mu_grid, scale, df)
    prior_d /= np.trapezoid(prior_d, mu_grid)   # normalise for display
    post_d = gaussian_kde(mu_all)(mu_grid)

    # Likelihood ≈ posterior / prior (importance-weighted KDE).
    # Weight each posterior sample by 1/prior(mu); the resulting KDE estimates
    # the likelihood up to a normalising constant.
    prior_at_samples = np.maximum(_half_student_t_pdf(mu_all, scale, df), 1e-300)
    w = 1.0 / prior_at_samples
    w /= w.sum()
    lik_d = gaussian_kde(mu_all, weights=w)(mu_grid)
    log_lik = np.log(np.maximum(lik_d, 1e-300))
    # Mask to the 1%–99% quantile range of the posterior samples.
    # Outside this range the 1/prior weighting is unreliable.
    mu_lo, mu_hi = np.percentile(mu_all, [1.0, 99.0])
    reliable = (mu_grid >= mu_lo) & (mu_grid <= mu_hi)
    log_lik_masked = np.where(reliable, log_lik, np.nan)
    # Rescale so the maximum of the masked values is 0.
    log_lik_masked -= np.nanmax(log_lik_masked)

    fig, ax = plt.subplots(figsize=(8, 5))

    prior_label = (f"Prior  [half-Student-t(df={df:.4g}, scale={scale:.2g})]"
                   if df != 1.0 else f"Prior  [half-Cauchy(scale={scale:.2g})]")
    ax.plot(mu_grid, prior_d, color="gray", lw=1.8, ls="--", label=prior_label)
    ax.plot(mu_grid, post_d, color="steelblue", lw=2.2, label="Posterior")

    # Log-likelihood on right axis; limits set from plotted values with 10% margin.
    ax2 = ax.twinx()
    ax2.plot(mu_grid, log_lik_masked, color="firebrick", lw=2.0, ls="-.",
             label="Log-likelihood")
    ax2.set_ylabel("Log-likelihood  (relative to max)", color="firebrick", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="firebrick")
    lk_min = float(np.nanmin(log_lik_masked))
    margin = 0.1 * abs(lk_min)
    # 0 (the peak) sits just below the top; lower limit has a small margin.
    ax2.set_ylim(lk_min - margin, margin)

    if mu_true is not None:
        ax.axvline(mu_true, color="black", ls=":", lw=1.5,
                   label=f"True \u03bc = {mu_true:.3g}")

    ax.set_xlabel("\u03bc  (genetic information, nats)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    plot_title = title or (
        f"Genetic information posterior\n"
        f"{n_chains} chains \u00d7 {n_samples:,} samples"
    )
    ax.set_title(plot_title, fontsize=11)

    # Combined legend from both axes.
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    ax.text(
        0.97, 0.95,
        f"median = {median_mu:.3f}\n"
        f"90% CI = [{ci90_lo:.3f}, {ci90_hi:.3f}]\n"
        f"ESS = {ess:.0f}\n"
        f"$\\hat{{R}}$ = {rhat:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
    )

    ax.set_xlim(0, mu_max)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if outfile is not None:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {outfile}")
    else:
        try:
            from IPython.display import display as _ipy_display
            _ipy_display(fig)
        except ImportError:
            plt.show()
    plt.close(fig)

    return {
        "mean": mean_mu,
        "median": median_mu,
        "sd": sd_mu,
        "ci90_lo": ci90_lo,
        "ci90_hi": ci90_hi,
        "ess_bulk": ess,
        "r_hat": rhat,
    }


# ── compression / synthetic generation ───────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _reconstruct_L_perm(gb) -> np.ndarray:
    """Reconstruct the full block-diagonal L in permuted order from a GroupedBlocks."""
    M = gb.M
    L_perm = np.zeros((M, M), dtype=np.float64)
    for s in gb.sizes:
        idxs = gb.idx_by_size[s]
        n_s  = len(idxs) // s
        for k in range(n_s):
            block_idxs = idxs[k * s:(k + 1) * s]
            L_perm[np.ix_(block_idxs, block_idxs)] = gb.L_by_size[s][k]
    return L_perm


def _gb_from_block_list(L_blocks: list, block_sizes: np.ndarray):
    """
    Build a GroupedBlocks from an ordered list of L sub-matrices.

    Blocks occupy consecutive rows/columns in the order given (block 0 →
    rows 0..s0-1, block 1 → s0..s0+s1-1, etc.).
    """
    from pg_gibbs_vectorized import GroupedBlocks

    by_size: dict = {}
    offset = 0
    for k, s_raw in enumerate(block_sizes):
        s = int(s_raw)
        by_size.setdefault(s, []).append((offset, L_blocks[k]))
        offset += s

    L_by_size  = {}
    idx_by_size = {}
    for s, entries in by_size.items():
        L_by_size[s]   = np.stack([e[1] for e in entries])
        idx_by_size[s] = np.concatenate(
            [np.arange(e[0], e[0] + s) for e in entries]
        )

    sizes = sorted(by_size.keys())
    M = int(sum(int(s) for s in block_sizes))
    return GroupedBlocks(M=M, sizes=sizes,
                         L_by_size=L_by_size, idx_by_size=idx_by_size)


def _load_block_list(model_file: str) -> tuple:
    """Return (L_blocks, block_sizes) from a compressed model file."""
    data = np.load(model_file)
    block_sizes  = data["block_sizes"].astype(int)
    block_L_flat = data["block_L_flat"]
    L_blocks = []
    offset = 0
    for s in block_sizes:
        s = int(s)
        L_blocks.append(block_L_flat[offset:offset + s * s].reshape(s, s))
        offset += s * s
    return L_blocks, block_sizes


def _pairwise_concordance(gb, y_perm: np.ndarray, bins: list) -> dict:
    """
    Compute within-block pairwise concordance P(y_i == y_j) by kinship bin.
    Only blocks of size >= 2 contribute.  Kinship is A_{ij} = (L L^T)_{ij}.
    """
    bin_edges = list(zip([0.0] + list(bins), list(bins) + [np.inf]))
    A_vals, conc_vals = [], []

    for s in gb.sizes:
        if s < 2:
            continue
        Ls   = gb.L_by_size[s]
        idxs = gb.idx_by_size[s]
        n_s  = Ls.shape[0]
        for k in range(n_s):
            L_k     = Ls[k]
            A_k     = L_k @ L_k.T
            block_y = y_perm[idxs[k * s:(k + 1) * s]]
            for i in range(s):
                for j in range(i + 1, s):
                    A_vals.append(A_k[i, j])
                    conc_vals.append(int(block_y[i] == block_y[j]))

    A_arr    = np.array(A_vals)
    conc_arr = np.array(conc_vals)

    concordance, n_pairs = [], []
    for lo, hi in bin_edges:
        mask = (A_arr >= lo) & (A_arr < hi)
        n = int(np.sum(mask))
        n_pairs.append(n)
        concordance.append(float(np.mean(conc_arr[mask])) if n > 0 else float("nan"))

    return {"concordance": concordance, "n_pairs": n_pairs, "bin_edges": bin_edges}


def compress(
    L: np.ndarray,
    y: np.ndarray,
    posterior_result: dict,
    outfile: str = "compressed_model.npz",
    corr_threshold: float = 0.0,
) -> None:
    """
    Compress an (L, y) dataset and posterior samples into a downloadable model.

    What is stored
    --------------
    * Per-block L sub-matrices (no individual IDs)
    * Population-level posterior samples of (mu, beta0, p)
    * Observed prevalence K = mean(y)

    What is NOT stored
    ------------------
    * Individual y values
    * Any individual identifier
    * Any quantity that allows reconstructing individual y beyond what is
      predictable from genetics alone (see test_privacy()).

    Parameters
    ----------
    L : (M, M) Cholesky factor of the genetic relationship matrix
    y : (M,) binary case/control outcomes
    posterior_result : dict returned by sample_posterior()
    outfile : output path (.npz)
    corr_threshold : passed to build_blocks() for block detection
    """
    L = np.asarray(L, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    blocks = build_blocks(L, y, corr_threshold=corr_threshold)
    gb     = blocks["gb"]
    block_info = blocks["block_info"]

    block_sizes_list: list = []
    L_flat_parts: list     = []
    for s in sorted(gb.sizes):
        Ls  = gb.L_by_size[s]
        n_s = Ls.shape[0]
        for k in range(n_s):
            block_sizes_list.append(s)
            L_flat_parts.append(Ls[k].ravel())

    block_sizes  = np.array(block_sizes_list, dtype=np.int32)
    block_L_flat = np.concatenate(L_flat_parts).astype(np.float64)

    chain_dicts   = posterior_result["chain_dicts"]
    mu_samples    = np.concatenate([d["mu"]    for d in chain_dicts])
    beta0_samples = np.concatenate([d["beta0"] for d in chain_dicts])
    p_samples     = np.concatenate([d["p"]     for d in chain_dicts])

    K              = float(np.mean(y))
    mu_prior_scale = float(posterior_result.get("mu_prior_scale", 1.0))

    np.savez_compressed(
        outfile,
        K=np.float64(K),
        block_sizes=block_sizes,
        block_L_flat=block_L_flat,
        mu_samples=mu_samples.astype(np.float64),
        beta0_samples=beta0_samples.astype(np.float64),
        p_samples=p_samples.astype(np.float64),
        mu_prior_scale=np.float64(mu_prior_scale),
    )

    n_blocks = len(block_sizes)
    M        = int(np.sum(block_sizes))
    print(f"Compressed model saved to {outfile!r}")
    print(f"  M={M}  n_blocks={n_blocks}  posterior_samples={len(mu_samples)}")
    print(f"  K={K:.4f}  mu_prior_scale={mu_prior_scale:.3g}")
    print(f"  sizes: "
          + "  ".join(f"size-{s}: {block_info['sizes_summary'][s]}"
                      for s in sorted(block_info["sizes_summary"])))


def generate_synthetic(
    model_file: str,
    seed: int = 0,
) -> tuple:
    """
    Generate synthetic (L_syn, y_syn) by drawing from the posterior predictive.

    The block structure (sizes and L sub-matrices) is identical to the original
    dataset.  Synthetic y values are new draws: no synthetic individual
    corresponds to any real individual.

    Parameters
    ----------
    model_file : path to compressed_model.npz
    seed : int — selects the posterior sample and seeds all random draws

    Returns
    -------
    L_syn : (M, M) block-diagonal Cholesky factor (same blocks as original)
    y_syn : (M,) synthetic binary outcomes
    """
    data = np.load(model_file)
    block_sizes   = data["block_sizes"].astype(int)
    block_L_flat  = data["block_L_flat"]
    mu_samples    = data["mu_samples"]
    beta0_samples = data["beta0_samples"]
    p_samples     = data["p_samples"]

    rng   = np.random.default_rng(seed)
    n_post = len(mu_samples)
    idx   = int(seed % n_post)
    mu    = float(mu_samples[idx])
    beta0 = float(beta0_samples[idx])
    p     = float(p_samples[idx])

    L_blocks = []
    offset = 0
    for s in block_sizes:
        s = int(s)
        L_blocks.append(block_L_flat[offset:offset + s * s].reshape(s, s))
        offset += s * s

    M     = int(sum(block_sizes))
    y_syn = np.empty(M, dtype=np.float64)

    pos = 0
    for L_k in L_blocks:
        s = L_k.shape[0]
        r   = 2 * rng.binomial(1, p, s) - 1
        z   = rng.normal(r * mu, np.sqrt(max(2 * mu, 1e-10)), s)
        eta = beta0 + L_k @ z
        y_syn[pos:pos + s] = rng.binomial(1, _sigmoid(eta))
        pos += s

    L_syn = np.zeros((M, M), dtype=np.float64)
    pos = 0
    for L_k in L_blocks:
        s = L_k.shape[0]
        L_syn[pos:pos + s, pos:pos + s] = L_k
        pos += s

    return L_syn, y_syn


def validate(
    L_real: np.ndarray,
    y_real: np.ndarray,
    model_file: str,
    n_synthetic: int = 5,
    sampler_kwargs: dict | None = None,
) -> None:
    """
    Compare real and synthetic datasets on four criteria.

    1. Block size distribution (histogram + KS test)
    2. Pairwise concordance P(y_i==y_j) by kinship bin (within-family pairs)
    3. Marginal prevalence K
    4. Posterior recovery: run sample_posterior() on one synthetic dataset
       and compare the mu posterior to the stored real-data posterior

    Parameters
    ----------
    L_real, y_real : original data
    model_file     : path to compressed_model.npz
    n_synthetic    : number of synthetic datasets for prevalence statistics
    sampler_kwargs : forwarded to sample_posterior()
    """
    data          = np.load(model_file)
    mu_samples    = data["mu_samples"]
    K_model       = float(data["K"])
    mu_prior_scale = float(data["mu_prior_scale"])

    L_real = np.asarray(L_real, dtype=np.float64)
    y_real = np.asarray(y_real, dtype=np.float64)

    blocks_real = build_blocks(L_real, y_real)
    gb_real     = blocks_real["gb"]
    y_perm_real = blocks_real["y_perm"]

    real_sizes = np.array([
        s for s in gb_real.sizes for _ in range(gb_real.L_by_size[s].shape[0])
    ])

    print("=== Validation report ===\n")

    syn_sizes = data["block_sizes"].astype(int)
    ks_stat, ks_p = ks_2samp(real_sizes, syn_sizes)

    K_real = float(np.mean(y_real))
    K_syns = []
    for seed in range(n_synthetic):
        _, y_syn = generate_synthetic(model_file, seed=seed)
        K_syns.append(float(np.mean(y_syn)))

    print("Block size distribution:")
    real_u, real_c = np.unique(real_sizes, return_counts=True)
    syn_u,  syn_c  = np.unique(syn_sizes,  return_counts=True)
    print(f"  Real:      {dict(zip(real_u.tolist(), real_c.tolist()))}")
    print(f"  Synthetic: {dict(zip(syn_u.tolist(),  syn_c.tolist()))}")
    print(f"  KS test:   stat={ks_stat:.4f}  p={ks_p:.4f}")

    print(f"\nMarginal prevalence:")
    print(f"  Real:      K={K_real:.4f}")
    print(f"  Model:     K={K_model:.4f}")
    print(f"  Synthetic: K={np.mean(K_syns):.4f} ± {np.std(K_syns):.4f}  "
          f"(n={n_synthetic} datasets)")

    kin_bins  = [0.1, 0.2, 0.3, 0.4, 0.6]
    real_conc = _pairwise_concordance(gb_real, y_perm_real, kin_bins)

    L_blocks, block_sizes_arr = _load_block_list(model_file)
    gb_syn = _gb_from_block_list(L_blocks, block_sizes_arr)
    _, y_syn0 = generate_synthetic(model_file, seed=0)
    syn_conc  = _pairwise_concordance(gb_syn, y_syn0, kin_bins)

    print(f"\nPairwise concordance by kinship bin (relative pairs within blocks):")
    print(f"  {'Kinship range':<20}  {'Real':<8}  {'Synthetic':<10}  N_pairs")
    for (lo, hi), rc, sc, np_ in zip(
            real_conc["bin_edges"],
            real_conc["concordance"],
            syn_conc["concordance"],
            real_conc["n_pairs"]):
        hi_str = f"{hi:.2f}" if np.isfinite(hi) else "∞"
        rc_str = f"{rc:.3f}" if not np.isnan(rc) else "   —"
        sc_str = f"{sc:.3f}" if not np.isnan(sc) else "   —"
        print(f"  [{lo:.2f}, {hi_str})              {rc_str:<8}  {sc_str:<10}  {np_}")

    print(f"\nPosterior recovery (sample_posterior on synthetic data):")
    kwargs   = sampler_kwargs or {}
    L_syn, y_syn = generate_synthetic(model_file, seed=0)
    result_syn = sample_posterior(
        L_syn, y_syn,
        progress_bar=False,
        mu_prior_scale=mu_prior_scale,
        **kwargs,
    )
    mu_syn_mean  = float(np.mean(result_syn["mu_all"]))
    mu_real_mean = float(np.mean(mu_samples))
    mu_real_90lo = float(np.percentile(mu_samples, 5))
    mu_real_90hi = float(np.percentile(mu_samples, 95))
    within_ci    = mu_real_90lo <= mu_syn_mean <= mu_real_90hi
    print(f"  Real posterior:      mean={mu_real_mean:.4f}  "
          f"90% CI=[{mu_real_90lo:.4f}, {mu_real_90hi:.4f}]")
    print(f"  Synthetic posterior: mean={mu_syn_mean:.4f}  "
          f"{'PASS' if within_ci else 'FAIL'} (within real 90% CI?)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    x = np.arange(len(real_u))
    ax.bar(x - 0.2, real_c, 0.4, label="Real",      color="steelblue", alpha=0.8)
    syn_c_aligned = np.array([
        int(np.sum(syn_c[syn_u == s])) if s in syn_u else 0 for s in real_u
    ])
    ax.bar(x + 0.2, syn_c_aligned, 0.4, label="Synthetic", color="firebrick", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(real_u)
    ax.set_xlabel("Block size")
    ax.set_ylabel("Count")
    ax.set_title("Block size distribution")
    ax.legend()

    ax = axes[1]
    mu_max = max(float(np.percentile(mu_samples, 99)),
                 float(np.percentile(result_syn["mu_all"], 99))) * 1.5
    mu_grid = np.linspace(1e-4, mu_max, 400)
    ax.plot(mu_grid, gaussian_kde(mu_samples)(mu_grid),
            color="steelblue", lw=2, label="Real data")
    ax.plot(mu_grid, gaussian_kde(result_syn["mu_all"])(mu_grid),
            color="firebrick", lw=2, ls="--", label="Synthetic data")
    ax.set_xlabel("μ (nats)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior recovery: μ")
    ax.legend()

    plt.tight_layout()
    outfig = "validate_synthetic.png"
    plt.savefig(outfig, dpi=120)
    print(f"\nFigure saved to {outfig!r}")
    try:
        from IPython.display import display as _ipy_display
        _ipy_display(fig)
    except ImportError:
        plt.show()
    plt.close(fig)


def test_privacy(
    L: np.ndarray,
    y: np.ndarray,
    model_file: str,
    n_test: int = 20,
    **sampler_kwargs,
) -> dict:
    """
    Membership inference test: compare P(y_i=1 | relatives, θ_full) vs
    P(y_i=1 | relatives, θ_loo) for n_test randomly chosen individuals
    with at least one relative (block size ≥ 2).

    For a well-regularised Bayesian model with M >> 1 individuals,
    |P_full - P_loo| should be small relative to the posterior SD of mu.

    Parameters
    ----------
    L, y           : original data
    model_file     : path to compressed_model.npz
    n_test         : number of test individuals
    **sampler_kwargs : forwarded to sample_posterior()

    Returns
    -------
    dict: diffs, max_diff, mean_diff, p95_diff, mu_posterior_sd
    """
    L = np.asarray(L, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    data          = np.load(model_file)
    mu_samples    = data["mu_samples"]
    beta0_samples = data["beta0_samples"]
    mu_prior_scale = float(data["mu_prior_scale"])

    blocks = build_blocks(L, y)
    gb     = blocks["gb"]
    y_perm = blocks["y_perm"]
    M      = gb.M

    mu_full    = float(np.mean(mu_samples))
    beta0_full = float(np.mean(beta0_samples))

    L_perm = _reconstruct_L_perm(gb)

    candidates: list = []
    for s in gb.sizes:
        if s < 2:
            continue
        idxs = gb.idx_by_size[s]
        n_s  = len(idxs) // s
        for k in range(n_s):
            for local_i in range(s):
                candidates.append((int(idxs[k * s + local_i]), s, k, local_i))

    if not candidates:
        print("No individuals with relatives (all blocks are singletons).")
        return {}

    rng     = np.random.default_rng(42)
    chosen  = rng.choice(len(candidates), min(n_test, len(candidates)), replace=False)
    test_set = [candidates[i] for i in chosen]

    print(f"Privacy test: {len(test_set)} of {len(candidates)} individuals with relatives\n")

    diffs = []
    for global_i, s, k, local_i in test_set:
        L_k        = gb.L_by_size[s][k]
        A_k        = L_k @ L_k.T
        block_idxs = gb.idx_by_size[s][k * s:(k + 1) * s]
        block_y    = y_perm[block_idxs]

        eta_full = beta0_full
        for local_j in range(s):
            if local_j == local_i:
                continue
            eta_full += A_k[local_i, local_j] * (2 * float(block_y[local_j]) - 1) * mu_full
        P_full = float(_sigmoid(np.array([eta_full]))[0])

        keep = np.ones(M, dtype=bool)
        keep[global_i] = False
        result_loo = sample_posterior(
            L_perm[np.ix_(keep, keep)],
            y_perm[keep],
            progress_bar=False,
            mu_prior_scale=mu_prior_scale,
            **sampler_kwargs,
        )
        mu_loo    = float(np.mean(result_loo["mu_all"]))
        beta0_loo = float(np.mean(
            np.concatenate([d["beta0"] for d in result_loo["chain_dicts"]])
        ))

        eta_loo = beta0_loo
        for local_j in range(s):
            if local_j == local_i:
                continue
            eta_loo += A_k[local_i, local_j] * (2 * float(block_y[local_j]) - 1) * mu_loo
        P_loo = float(_sigmoid(np.array([eta_loo]))[0])

        diff = abs(P_full - P_loo)
        diffs.append(diff)
        print(f"  i={global_i:5d}  block_size={s}  "
              f"P_full={P_full:.4f}  P_loo={P_loo:.4f}  |diff|={diff:.4f}")

    diffs  = np.array(diffs)
    mu_sd  = float(np.std(mu_samples))
    max_d  = float(np.max(diffs))
    mean_d = float(np.mean(diffs))
    p95_d  = float(np.percentile(diffs, 95))

    print(f"\nSummary over {len(diffs)} test individuals:")
    print(f"  max  |P_full - P_loo| = {max_d:.4f}")
    print(f"  mean |P_full - P_loo| = {mean_d:.4f}")
    print(f"  95th percentile       = {p95_d:.4f}")
    print(f"  Posterior SD of mu    = {mu_sd:.4f}  (reference scale)")
    threshold = 0.02
    if max_d < threshold:
        print(f"  PASS: max |diff| = {max_d:.4f} < {threshold}")
    else:
        print(f"  WARNING: max |diff| = {max_d:.4f} >= {threshold}")

    return {
        "diffs": diffs,
        "max_diff": max_d,
        "mean_diff": mean_d,
        "p95_diff": p95_d,
        "mu_posterior_sd": mu_sd,
    }
