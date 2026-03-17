"""
PG-Gibbs with vectorised block operations on create_data() dataset.
4 chains run in parallel (fork), one per CPU core.

Key speedup: instead of looping over n_blocks=3237 in Python, operations
are grouped by block size and vectorised:
  - 3180 singletons: one numpy call each for Cholesky, r-update, theta-cache
  - 57 pairs:        batched 2x2 analytically (no scipy Cholesky per block)
"""
import multiprocessing as mp
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import arviz as az
from tqdm import tqdm

from polyagamma_gibbs import ChainConfig
from pg_gibbs_vectorized import _worker

TARGET   = 1.0
K        = 0.01
DATASEED = 0
NUM_CHAINS = 4

cfg = ChainConfig(
    n_warmup             = 1000,
    n_samples            = 10000,
    prior_loc            = 0.0,   # initial theta = log(mu); half-Cauchy has no loc
    prior_scale          = 1.0,   # half-Cauchy scale for mu
    beta0_sd             = 5.0,
    slice_w              = 1.5,
    slice_m              = 20,
    slice_max_steps      = 250,
    inner_latent_cycles  = 1,
    learn_p              = True,
    p_prior_conc         = 200.0,
    project_affects_beta0= True,
    diag_every           = 100,
    data_flag            = True,
)

if __name__ == "__main__":
    print(f"True mu = {TARGET:.4f}   lambda_S = {np.exp(TARGET):.4f}")
    print(f"{NUM_CHAINS} chains x {cfg.n_samples} samples  ({cfg.n_warmup} warmup)\n")

    jobs = [(cid, TARGET, 42+cid, DATASEED, K, cfg) for cid in range(NUM_CHAINS)]

    ctx  = mp.get_context("fork")
    lock = ctx.RLock()
    with ctx.Pool(processes=NUM_CHAINS,
                  initializer=tqdm.set_lock, initargs=(lock,)) as pool:
        chains_out = pool.map(_worker, jobs)

    mu_all = np.concatenate([o["mu"] for o in chains_out])
    print(f"\nTrue mu = {TARGET:.4f}")
    print(f"Posterior mu:  mean={mu_all.mean():.4f}  "
          f"median={np.median(mu_all):.4f}  sd={mu_all.std():.4f}  "
          f"90% CI=[{np.percentile(mu_all,5):.4f}, {np.percentile(mu_all,95):.4f}]")

    az_data = az.convert_to_inference_data(
        {"mu": np.stack([o["mu"] for o in chains_out])})
    print(az.summary(az_data, var_names=["mu"]).to_string())

    # extract ArviZ summary for annotation
    summ     = az.summary(az_data, var_names=["mu"])
    ess_bulk = float(summ["ess_bulk"].iloc[0])
    r_hat    = float(summ["r_hat"].iloc[0])

    # plot — prior is log-normal: theta = log(mu) ~ N(prior_loc, prior_scale)
    mu_max  = max(5.0, float(np.percentile(mu_all, 99)) * 1.5)
    mu_grid = np.linspace(1e-4, mu_max, 500)

    def halfcauchy_pdf(x, scale):
        """Half-Cauchy(scale) density at x > 0."""
        return 2.0 / (np.pi * scale * (1.0 + (x / scale) ** 2))

    prior_d = halfcauchy_pdf(mu_grid, cfg.prior_scale)
    post_d  = gaussian_kde(mu_all)(mu_grid)

    prior_at_samples = np.maximum(halfcauchy_pdf(mu_all, cfg.prior_scale), 1e-300)
    w = 1.0 / prior_at_samples
    w /= w.sum()
    lik_d  = gaussian_kde(mu_all, weights=w)(mu_grid)
    lik_d /= np.trapezoid(lik_d, mu_grid)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mu_grid, prior_d, label=f"Prior  [half-Cauchy(scale={cfg.prior_scale})]",
            color="gray", linestyle="--", lw=1.8)
    ax.plot(mu_grid, post_d,  label="Posterior",
            color="steelblue", lw=2.2)
    ax.plot(mu_grid, lik_d,   label="Likelihood  (posterior / prior, KDE)",
            color="darkorange", lw=2.2)
    ax.axvline(TARGET, color="red", linestyle=":", lw=1.5,
               label=f"True μ = {TARGET:.2f}")
    ax.set_xlabel("μ  (genetic information, nats)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"PG-Gibbs (vectorised blocks)   {NUM_CHAINS} chains × {cfg.n_samples:,} samples\n"
        f"K = {K}   target μ = {TARGET}   M = 3294   57 sib pairs",
        fontsize=10,
    )
    ax.text(0.97, 0.95,
            f"ESS_bulk = {ess_bulk:.0f}\n$\\hat{{R}}$ = {r_hat:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.legend(fontsize=10)
    ax.set_xlim(0, mu_max)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig("mu_prior_posterior_likelihood.png", dpi=150)
    print("Plot saved to mu_prior_posterior_likelihood.png")
