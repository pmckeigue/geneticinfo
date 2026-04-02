"""
Simulate a case-control dataset with simulate_casecontrol_related(),
plot genotypic-value densities in cases vs controls, then fit the
PG-Gibbs model to estimate mu.

4 chains run in parallel via fork.  Singletons are excluded from the
theta log-posterior (min_block_size=2) so that unrelated individuals
do not bias mu downward.
"""
import multiprocessing as mp
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import arviz as az
from tqdm import tqdm

from geneticinfo_functions import simulate_casecontrol_related, print_casecontrol_summary
from polyagamma_gibbs import (
    ChainConfig, infer_blocks_from_L, BlockStructure,
)
from pg_gibbs_vectorized import GroupedBlocks, _worker_preloaded

# ── Simulation parameters ───────────────────────────────────────────────────
MU_TRUE          = 1.0
K                = 0.01
SEED_DATA        = 42
NUM_CHAINS       = 4

N_FULLSIB_PAIRS  = 400_000
N_FULLSIB_TRIPS  = 200_000
N_HALFSIB_PAIRS  = 40_000
N_UNRELATED      = 4_000

cfg = ChainConfig(
    n_warmup             = 1000,
    n_samples            = 5000,
    prior_loc            = 0.0,    # initial theta; half-Cauchy has no loc
    prior_scale          = 1.0,    # half-Cauchy scale for mu
    beta0_sd             = 5.0,
    slice_w              = 1.5,
    slice_m              = 20,
    slice_max_steps      = 250,
    inner_latent_cycles  = 1,
    learn_p              = True,
    p_prior_conc         = 200.0,
    project_affects_beta0= True,
    diag_every           = 200,
    data_flag            = True,
)

if __name__ == "__main__":
    # ── 1. Simulate ─────────────────────────────────────────────────────────
    print("Simulating case-control dataset ...")
    y_sample, A_sample, L_sample, info, g_sample = simulate_casecontrol_related(
        n_fullsib_pairs = N_FULLSIB_PAIRS,
        n_fullsib_trips = N_FULLSIB_TRIPS,
        n_halfsib_pairs = N_HALFSIB_PAIRS,
        n_unrelated     = N_UNRELATED,
        K               = K,
        mu              = MU_TRUE,
        seed            = SEED_DATA,
        return_genotypic_values = True,
    )
    print_casecontrol_summary(info)
    M = info["M"]
    print(f"\nCase-control sample: M={M}  "
          f"cases={info['n_cases_sample']}  controls={info['n_controls_sample']}")

    # ── 2. Plot genotypic-value densities ────────────────────────────────────
    g_cases    = g_sample[y_sample == 1]
    g_controls = g_sample[y_sample == 0]
    sigma      = float(info["s"])   # sqrt(2*mu)

    # Theoretical class-conditional means under the mixture model
    # G | class=+ ~ N(+mu, 2*mu),  G | class=- ~ N(-mu, 2*mu)
    g_lo = min(g_sample.min(), -4 * sigma)
    g_hi = max(g_sample.max(),  4 * sigma)
    g_grid = np.linspace(g_lo, g_hi, 500)

    fig_g, ax_g = plt.subplots(figsize=(8, 4))
    ax_g.plot(g_grid, gaussian_kde(g_cases)(g_grid),
              color="firebrick", lw=2.0, label=f"Cases  (n={len(g_cases):,})")
    ax_g.plot(g_grid, gaussian_kde(g_controls)(g_grid),
              color="steelblue", lw=2.0, label=f"Controls  (n={len(g_controls):,})")
    # Theoretical mixture components
    from scipy.stats import norm as _norm
    ax_g.plot(g_grid, _norm.pdf(g_grid, +MU_TRUE, sigma),
              color="firebrick", lw=1.2, linestyle="--",
              label=f"N(+μ, 2μ)  μ={MU_TRUE}")
    ax_g.plot(g_grid, _norm.pdf(g_grid, -MU_TRUE, sigma),
              color="steelblue", lw=1.2, linestyle="--",
              label=f"N(−μ, 2μ)  μ={MU_TRUE}")
    ax_g.axvline(0, color="gray", lw=0.8, linestyle=":")
    ax_g.set_xlabel("Genotypic value  g", fontsize=12)
    ax_g.set_ylabel("Density", fontsize=12)
    ax_g.set_title(
        f"Genotypic-value distributions in cases and controls\n"
        f"K={K}   μ={MU_TRUE}   σ=√(2μ)={sigma:.3f}   M={M:,}",
        fontsize=10,
    )
    ax_g.legend(fontsize=9)
    ax_g.set_ylim(bottom=0)
    fig_g.tight_layout()
    fig_g.savefig("genotypic_value_densities.png", dpi=150)
    print("Plot saved: genotypic_value_densities.png")

    # ── 3. Build block structure (once, shared to all chains via fork) ───────
    print("\nBuilding block structure ...")
    L_np = np.asarray(L_sample, dtype=np.float64)
    y_np = np.asarray(y_sample, dtype=np.float64)

    perm_rows, perm_cols, row_slices, col_slices = infer_blocks_from_L(L_np, tol_rel=0.0)
    L_perm = L_np[perm_rows, :][:, perm_cols]
    y_perm = y_np[perm_rows]

    bs = BlockStructure(L_perm, col_slices, row_slices)
    gb = GroupedBlocks.from_block_structure(bs)
    sizes_str = "  ".join(f"size-{s}:{gb.L_by_size[s].shape[0]}" for s in gb.sizes)
    print(f"  M={gb.M}  blocks={bs.n_blocks}  {sizes_str}")
    M_related = sum(gb.L_by_size[s].shape[0] * s
                    for s in gb.sizes if s >= 2)
    print(f"  Related individuals (block size ≥ 2): {M_related}")

    # ── 4. Run 4 chains in parallel ──────────────────────────────────────────
    print(f"\nRunning {NUM_CHAINS} chains × {cfg.n_samples:,} samples "
          f"({cfg.n_warmup} warmup) ...")
    jobs = [
        (cid, gb, y_perm, 42 + cid, cfg, MU_TRUE)
        for cid in range(NUM_CHAINS)
    ]

    ctx  = mp.get_context("fork")
    lock = ctx.RLock()
    with ctx.Pool(processes=NUM_CHAINS,
                  initializer=tqdm.set_lock, initargs=(lock,)) as pool:
        chains_out = pool.map(_worker_preloaded, jobs)

    # ── 5. Summarise ─────────────────────────────────────────────────────────
    mu_all  = np.concatenate([o["mu"] for o in chains_out])
    az_data = az.convert_to_inference_data(
        {"mu": np.stack([o["mu"] for o in chains_out])})
    summ     = az.summary(az_data, var_names=["mu"])
    ess_bulk = float(summ["ess_bulk"].iloc[0])
    r_hat    = float(summ["r_hat"].iloc[0])

    print(f"\nTrue mu = {MU_TRUE:.4f}   lambda_S = {np.exp(MU_TRUE):.4f}")
    print(f"Posterior mu:  mean={mu_all.mean():.4f}  "
          f"median={np.median(mu_all):.4f}  sd={mu_all.std():.4f}  "
          f"90% CI=[{np.percentile(mu_all, 5):.4f}, {np.percentile(mu_all, 95):.4f}]")
    print(summ.to_string())

    # ── 6. Plot prior / posterior / likelihood ───────────────────────────────
    def halfcauchy_pdf(x, scale):
        return 2.0 / (np.pi * scale * (1.0 + (x / scale) ** 2))

    mu_max  = max(5.0, float(np.percentile(mu_all, 99)) * 1.5)
    mu_grid = np.linspace(1e-4, mu_max, 500)

    prior_d = halfcauchy_pdf(mu_grid, cfg.prior_scale)
    post_d  = gaussian_kde(mu_all)(mu_grid)

    prior_at_samples = np.maximum(halfcauchy_pdf(mu_all, cfg.prior_scale), 1e-300)
    w = 1.0 / prior_at_samples;  w /= w.sum()
    lik_d  = gaussian_kde(mu_all, weights=w)(mu_grid)
    lik_d /= np.trapezoid(lik_d, mu_grid)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mu_grid, prior_d,
            label=f"Prior  [half-Cauchy(scale={cfg.prior_scale})]",
            color="gray", linestyle="--", lw=1.8)
    ax.plot(mu_grid, post_d,
            label="Posterior", color="steelblue", lw=2.2)
    ax.plot(mu_grid, lik_d,
            label="Likelihood  (posterior / prior, KDE)",
            color="darkorange", lw=2.2)
    ax.axvline(MU_TRUE, color="red", linestyle=":", lw=1.5,
               label=f"True μ = {MU_TRUE:.2f}")
    ax.set_xlabel("μ  (genetic information, nats)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"PG-Gibbs   {NUM_CHAINS} chains × {cfg.n_samples:,} samples   "
        f"K={K}   M={M:,}   M_related={M_related}\n"
        f"empirical λ_S = {info['empirical_lambda_S']:.3f}   "
        f"theoretical = {info['theoretical_lambda_S']:.3f}",
        fontsize=10,
    )
    ax.text(0.97, 0.95,
            f"ESS_bulk = {ess_bulk:.0f}\n$\\hat{{R}}$ = {r_hat:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.legend(fontsize=10)
    ax.set_xlim(0, mu_max);  ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig("mu_prior_posterior_likelihood.png", dpi=150)
    print("Plot saved: mu_prior_posterior_likelihood.png")
