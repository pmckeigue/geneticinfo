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

Model
-----
  y_i ~ Bernoulli(sigmoid(phi + (L z)_i))
  r_i in {-1, +1},  P(r_i=+1) = p  (discrete class indicator)
  z | r, mu ~ N(mu * r, 2*mu * I)
  p ~ Beta(K_beta * p_obs, K_beta * (1 - p_obs))
  mu ~ HalfCauchy(mu_prior_scale)

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
from scipy.stats import gaussian_kde
from scipy.sparse.csgraph import connected_components as _scipy_connected_components
from threadpoolctl import threadpool_limits


def _set_blas_threads(n: int) -> None:
    """
    Set BLAS/LAPACK thread count in the current process.

    Works for both OpenBLAS and MKL:
    1. Update environment variables — MKL re-reads these after fork when it
       recreates its thread pool.
    2. mkl.set_num_threads(n) via the mkl-service package (conda MKL envs).
    3. Direct ctypes call to openblas_set_num_threads() for OpenBLAS.
    4. threadpoolctl.threadpool_limits() as a final fallback, with
       dl_iterate_phdr warnings suppressed.
    """
    import ctypes
    # 1. Env vars — inherited by forked workers and re-read by MKL on fork
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    # 2. MKL Python API (mkl-service package, standard in conda MKL envs)
    try:
        import mkl
        mkl.set_num_threads(n)
    except Exception:
        pass
    # 3. OpenBLAS direct C API via global symbol table
    try:
        ctypes.CDLL(None).openblas_set_num_threads(ctypes.c_int(n))
    except Exception:
        pass
    # 4. threadpoolctl fallback
    _old = sys.unraisablehook
    sys.unraisablehook = lambda _args: None
    try:
        threadpool_limits(limits=n, user_api="blas")
    finally:
        sys.unraisablehook = _old


def _pool_init_blas(n_threads: int) -> None:
    """Pool initializer: set BLAS threads in each worker right after fork."""
    _set_blas_threads(n_threads)

# Set conservative defaults at import time so the parent process does not spin
# up many threads before forking.  Each worker overrides this via
# threadpool_limits() using the per-chain allocation from sample_posterior().
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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
) -> ChainConfig:
    return ChainConfig(
        n_warmup=n_warmup,
        n_samples=n_samples,
        prior_loc=0.0,
        prior_scale=mu_prior_scale,
        beta0_sd=5.0,            # not used by LRDiscreteBlockdiagPGGibbs
        slice_w=1.5,
        slice_m=20,
        slice_max_steps=250,
        inner_latent_cycles=2,
        learn_p=True,
        p_prior_conc=p_prior_conc,
        project_affects_beta0=False,
        diag_every=n_warmup + n_samples + 1,  # suppress per-step diagnostics
    )


def _halfcauchy_pdf(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 2.0 / (np.pi * scale * (1.0 + (x / scale) ** 2))


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
        phi_init=None,
        mu_init=float(np.exp(getattr(cfg, "prior_loc", 0.0))),
    )

    n_w = lr_cfg.n_warmup
    n_s = lr_cfg.n_samples
    UPDATE_EVERY = 50

    last = 0
    for i in range(n_w):
        sampler.step()
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
        mu[t] = sampler.mu
        beta0[t] = sampler.phi
        p[t] = sampler.current_p()
        ll[t] = bernoulli_loglik_logit_np(sampler.current_eta(), y_perm)
        if (t + 1) % UPDATE_EVERY == 0 or t == n_s - 1:
            delta = (t + 1) - last
            last = t + 1
            q.put(("step", chain_id, delta, sampler.mu))

    q.put(("done", chain_id))

    return {
        "chain_id": int(chain_id),
        "true_mu": float(true_mu),
        "mu": mu,
        "beta0": beta0,
        "p": p,
        "ll": ll,
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
        Number of OpenBLAS/MKL threads allocated to each chain for
        matrix multiplications, Cholesky decompositions, and
        eigendecompositions.  Defaults to ``n_cpu // n_chains`` so that
        total thread usage equals the number of available cores.  For a
        single large block (e.g. 2918×2918), increasing this (and
        reducing n_chains accordingly) gives the largest speedup: each
        Gibbs step contains four O(n³) BLAS/LAPACK calls that scale
        well with thread count.

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

    cfg = _make_chain_config(n_warmup, n_samples, mu_prior_scale, p_prior_conc)

    n_cpu = os.cpu_count() or 1
    if n_blas_threads is None:
        n_blas_threads = max(1, n_cpu // n_chains)
    print(f"  {n_chains} chain(s) × {n_blas_threads} BLAS thread(s) per chain "
          f"({n_chains * n_blas_threads} of {n_cpu} CPUs)")
    _set_blas_threads(n_blas_threads)
    from threadpoolctl import threadpool_info
    _old_hook = sys.unraisablehook
    sys.unraisablehook = lambda _args: None
    try:
        _blas_info = [x for x in threadpool_info() if x.get("user_api") == "blas"]
    finally:
        sys.unraisablehook = _old_hook
    if _blas_info:
        print(f"  [debug] BLAS: " +
              ", ".join(f"{x['internal_api']} num_threads={x['num_threads']}" for x in _blas_info), flush=True)
    else:
        print(f"  [debug] threadpoolctl found no BLAS libraries", flush=True)

    p_obs = float(np.mean(y_perm))

    ctx = mp.get_context("fork")
    if progress_bar:
        # Set the module-level queue BEFORE creating the Pool so that forked
        # workers inherit it directly (avoids pickling the Queue through the
        # pool task pipe, which can deadlock).
        global _POOL_PROGRESS_Q
        _POOL_PROGRESS_Q = ctx.Queue()
        print(f"  [debug] progress queue created", flush=True)

        jobs = [
            (cid, gb, y_perm, seed + cid, cfg, float("nan"), p_obs)
            for cid in range(n_chains)
        ]
        print(f"  [debug] jobs built ({len(jobs)} chains)", flush=True)

        bars = [
            tqdm(total=n_warmup + n_samples, position=cid,
                 desc=f"chain {cid}  warmup", leave=True, dynamic_ncols=True)
            for cid in range(n_chains)
        ]
        print(f"  [debug] tqdm bars created, spawning Pool", flush=True)

        with ctx.Pool(processes=n_chains,
                      initializer=_pool_init_blas,
                      initargs=(n_blas_threads,)) as pool:
            print(f"  [debug] Pool spawned, submitting map_async", flush=True)
            async_result = pool.map_async(_worker_with_queue, jobs)
            print(f"  [debug] map_async submitted, entering monitor loop", flush=True)
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
                        print(f"  [debug] chain {cid} done ({finished}/{n_chains})", flush=True)
                except _queue_module.Empty:
                    pass
        for bar in bars:
            bar.close()
        chain_dicts = async_result.get()
        print()  # newline after the stacked bars
    else:
        jobs = [
            (cid, gb, y_perm, seed + cid, cfg, float("nan"), p_obs, False, False)
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

    prior_d = _halfcauchy_pdf(mu_grid, scale)
    post_d = gaussian_kde(mu_all)(mu_grid)

    # Likelihood ≈ posterior / prior (importance-weighted KDE), then normalise.
    prior_at_samples = np.maximum(_halfcauchy_pdf(mu_all, scale), 1e-300)
    w = 1.0 / prior_at_samples
    w /= w.sum()
    lik_d = gaussian_kde(mu_all, weights=w)(mu_grid)
    lik_d /= np.trapezoid(lik_d, mu_grid)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(mu_grid, prior_d, color="gray", lw=1.8, ls="--",
            label=f"Prior  [half-Cauchy(scale={scale:.2g})]")
    ax.plot(mu_grid, post_d, color="steelblue", lw=2.2,
            label="Posterior")
    ax.plot(mu_grid, lik_d, color="firebrick", lw=2.0, ls="-.",
            label="Likelihood  (posterior / prior)")

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

    ax.text(
        0.97, 0.95,
        f"median = {median_mu:.3f}\n"
        f"90% CI = [{ci90_lo:.3f}, {ci90_hi:.3f}]\n"
        f"ESS = {ess:.0f}\n"
        f"$\\hat{{R}}$ = {rhat:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
    )

    ax.legend(fontsize=9)
    ax.set_xlim(0, mu_max)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if outfile is not None:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {outfile}")
    else:
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
