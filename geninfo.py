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
import numpy as np
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Suppress BLAS threading contention when running multiple chains via fork.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from polyagamma_gibbs import infer_blocks_from_L, BlockStructure, ChainConfig
from pg_gibbs_vectorized import GroupedBlocks
from pg_gibbs_clean import _worker_preloaded_lrpg


# ─── private helpers ─────────────────────────────────────────────────────────

def _build_block_structure(
    L: np.ndarray,
    y: np.ndarray,
) -> tuple[GroupedBlocks, np.ndarray, dict]:
    """
    Permute L and y into block-diagonal order and return GroupedBlocks.

    Returns
    -------
    gb : GroupedBlocks
    y_perm : np.ndarray  (y reordered to match gb)
    block_info : dict with keys M, n_blocks, sizes_summary
    """
    L_np = np.asarray(L, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)

    perm_rows, perm_cols, row_slices, col_slices = infer_blocks_from_L(L_np, tol_rel=0.0)
    L_perm = L_np[perm_rows, :][:, perm_cols]
    y_perm = y_np[perm_rows]

    bs = BlockStructure(L_perm, col_slices, row_slices)
    gb = GroupedBlocks.from_block_structure(bs)

    sizes_summary = {s: int(gb.L_by_size[s].shape[0]) for s in gb.sizes}
    block_info = {
        "M": int(gb.M),
        "n_blocks": int(bs.n_blocks),
        "sizes_summary": sizes_summary,  # {block_size: n_blocks_of_that_size}
    }
    return gb, y_perm, block_info


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


# ─── public API ──────────────────────────────────────────────────────────────

def sample_posterior(
    L: np.ndarray,
    y: np.ndarray,
    *,
    n_chains: int = 4,
    n_warmup: int = 1000,
    n_samples: int = 5000,
    mu_prior_scale: float = 1.0,
    p_prior_conc: float = 20.0,
    seed: int = 0,
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

    Returns
    -------
    dict with keys:
      "mu_chains"   : np.ndarray (n_chains, n_samples) — per-chain samples
      "mu_all"      : np.ndarray (n_chains * n_samples,) — all samples pooled
      "chain_dicts" : list of per-chain result dicts (mu, beta0, p, ll, chain_id)
      "block_info"  : dict — M, n_blocks, sizes_summary
      "cfg"         : ChainConfig used
      "mu_prior_scale" : float
    """
    L = np.asarray(L, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be a square 2-D array; got shape {L.shape}")
    M = L.shape[0]
    if y.shape != (M,):
        raise ValueError(f"y must have shape ({M},); got {y.shape}")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0s and 1s")

    gb, y_perm, block_info = _build_block_structure(L, y)
    cfg = _make_chain_config(n_warmup, n_samples, mu_prior_scale, p_prior_conc)

    p_obs = float(np.mean(y))

    # Job tuple matches _worker_preloaded_lrpg signature:
    #   (chain_id, gb, y_perm, seed, cfg, true_mu, p_obs_override, ...)
    jobs = [
        (cid, gb, y_perm, seed + cid, cfg, float("nan"), p_obs)
        for cid in range(n_chains)
    ]

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_chains) as pool:
        chain_dicts = pool.map(_worker_preloaded_lrpg, jobs)

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
