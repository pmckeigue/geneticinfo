"""
Run Pólya-gamma collapsed-Gibbs on ascertained sibships.

Generates data via simulate_ascertained_sibships, builds BlockStructure
directly from the tiled Cholesky factors, runs 4 chains in parallel.

The PG-Gibbs sampler uses a Normal prior on theta = log(mu):
  theta ~ N(prior_loc, prior_scale)
which corresponds to a log-normal prior on mu.  With prior_loc=0.0 the
prior is centred on mu=1.0 (the true value used for simulation).
"""
import os
import multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import halfcauchy as scipy_halfcauchy
from tqdm import tqdm
import arviz as az

from geneticinfo_functions import simulate_ascertained_sibships, print_ascertained_summary
from polyagamma_gibbs import (
    BlockStructure,
    ChainConfig,
    bernoulli_loglik_logit_np,
    build_theta_eig_cache,
    collapsed_logpost_theta_uncentered_blocks_eig,
    sample_beta0_z_gaussian_blocks_joint,
    sample_r_block_exact_collapsed,
    slice_sample_theta_collapsed_numpy,
    update_c_gibbs_cpu,
    update_indicators_gibbs_cpu,
    _chol,
)
from polyagamma import random_polyagamma


# ── Simulate ascertained dataset ──────────────────────────────────────────
print("Simulating ascertained sibships ...")
y_np, L_blocks_3_np, L_blocks_2_np, n3, n2, info = simulate_ascertained_sibships(
    n_fullsib_pairs=6_000,
    n_fullsib_trips=3_000,
    n_halfsib_pairs=600,
    K=0.01,
    mu=1.0,
)
print_ascertained_summary(info)

M = 3 * n3 + 2 * n2
print(f"\nM={M}  n3={n3} triplets  n2={n2} pairs")
print(f"True: mu={info['mu']:.4f}  s={info['s']:.4f}  K={info['K_target']:.4f}")
print(f"K_observed={info['K_observed']:.4f}")

# ── Build block-diagonal L matrix ─────────────────────────────────────────
print("Building block-diagonal L matrix ...")
L_full = np.zeros((M, M), dtype=np.float64)
offset3 = 3 * n3
for b in range(n3):
    i = b * 3
    L_full[i:i+3, i:i+3] = L_blocks_3_np[b]
for b in range(n2):
    i = offset3 + b * 2
    L_full[i:i+2, i:i+2] = L_blocks_2_np[b]

y_full = np.asarray(y_np, dtype=np.float64)

row_slices = (
    [slice(b*3, b*3+3) for b in range(n3)] +
    [slice(offset3 + b*2, offset3 + b*2 + 2) for b in range(n2)]
)


# ── Per-chain worker (runs in a subprocess) ───────────────────────────────
def _run_chain(args):
    chain_id, seed, cfg_dict, L_full, y_full, row_slices, p_obs = args

    # Reconstruct objects inside the subprocess
    cfg = ChainConfig(**cfg_dict)
    blocks = BlockStructure(L_full, col_slices=row_slices, row_slices=row_slices)
    kappa = y_full - 0.5
    M_loc = len(y_full)

    rng = np.random.default_rng(seed)
    K_conc = cfg.p_prior_conc   # use the configured concentration
    a0 = K_conc * p_obs
    b0_b = K_conc * (1.0 - p_obs)

    # theta = log(mu); prior: N(prior_loc, prior_scale) on theta
    theta = float(cfg.prior_loc)           # init at prior mean (log-scale)
    mu_pos = float(np.exp(theta))
    beta0 = float(np.log(p_obs + 1e-12) - np.log1p(p_obs + 1e-12))

    r = np.where(rng.uniform(size=M_loc) < p_obs, 1.0, -1.0).astype(np.float64)
    z_unc = mu_pos * r + np.sqrt(2.0 * mu_pos) * rng.normal(size=M_loc)
    z = (z_unc - z_unc.mean()).astype(np.float64)
    c = float(update_c_gibbs_cpu(rng, mu_pos, r))
    p_mix = float(p_obs)

    mu_post, theta_post, beta0_post, ll_post = [], [], [], []
    last_omega = None

    total_iters = cfg.n_warmup + cfg.n_samples
    pbar = tqdm(total=total_iters, desc=f"chain {chain_id}",
                position=chain_id, leave=True, dynamic_ncols=True)

    for it in range(total_iters):
        for _ in range(cfg.inner_latent_cycles):
            eta = beta0 + blocks.L_times_vec(z)
            last_omega = random_polyagamma(1.0, np.abs(eta))

            mu_pos = float(np.exp(theta))
            tau = 1.0 / (2.0 * mu_pos)
            Lc_blocks, b_like_blocks = [], []
            for b_idx, cs in enumerate(blocks.col_slices):
                rs = blocks.row_slices[b_idx]
                Lb = blocks.L_blocks[b_idx]
                w_b = last_omega[rs]
                Wb = Lb * np.sqrt(w_b)[:, None]
                A_b = Wb.T @ Wb
                A_b[np.diag_indices_from(A_b)] += tau
                Lc_b = _chol(A_b)
                Lc_blocks.append(Lc_b)
                t_b = kappa[rs] - last_omega[rs] * beta0
                b_like_blocks.append(Lb.T @ t_b)

            r_new = r.copy()
            for b_idx, cs in enumerate(blocks.col_slices):
                r_b_new = sample_r_block_exact_collapsed(
                    rng, Lc_blocks[b_idx], b_like_blocks[b_idx],
                    r_new[cs], p_mix, max_enum=12,
                )
                r_new[cs] = r_b_new
            r = r_new

            mu_pos = float(np.exp(theta))
            beta0, z = sample_beta0_z_gaussian_blocks_joint(
                rng=rng, blocks=blocks, omega=last_omega, kappa=kappa,
                r=r, mu_pos=mu_pos, beta0_sd=cfg.beta0_sd,
                project_affects_beta0=cfg.project_affects_beta0,
            )

        mu_pos = float(np.exp(theta))
        c = float(update_c_gibbs_cpu(rng, mu_pos, r))
        r = update_indicators_gibbs_cpu(rng, z, c, mu_pos, p_mix)

        if cfg.learn_p:
            n_plus = float((r > 0).sum())
            n_minus = float((r < 0).sum())
            n_total = n_plus + n_minus
            # rescale to 10 pseudo-observations (as in original PG-Gibbs)
            p_mix = float(rng.beta(a0 + 10.0 * n_plus / n_total,
                                   b0_b + 10.0 * n_minus / n_total))

        beta0, z = sample_beta0_z_gaussian_blocks_joint(
            rng=rng, blocks=blocks, omega=last_omega, kappa=kappa,
            r=r, mu_pos=mu_pos, beta0_sd=cfg.beta0_sd,
            project_affects_beta0=cfg.project_affects_beta0,
        )

        cache = build_theta_eig_cache(blocks, last_omega, beta0, kappa, r)
        f_th = lambda th: collapsed_logpost_theta_uncentered_blocks_eig(
            float(th), cache, float(cfg.prior_loc), float(cfg.prior_scale))
        theta = float(slice_sample_theta_collapsed_numpy(
            rng, theta, f_th, w=cfg.slice_w, m=cfg.slice_m,
            max_steps=cfg.slice_max_steps))

        if it + 1 > cfg.n_warmup:
            mu_val = float(np.exp(theta))
            mu_post.append(mu_val)
            theta_post.append(theta)
            beta0_post.append(beta0)
            eta_ll = beta0 + blocks.L_times_vec(z)
            ll_post.append(bernoulli_loglik_logit_np(eta_ll, y_full))

        if cfg.diag_every and ((it + 1) % cfg.diag_every == 0) and len(mu_post) > 1:
            pbar.set_postfix({"mu": f"{np.median(mu_post):.3f}", "p": f"{p_mix:.4f}"})
        pbar.update(1)

    pbar.close()
    return {
        "chain_id": chain_id,
        "mu": np.array(mu_post, dtype=np.float64),
        "theta": np.array(theta_post, dtype=np.float64),
        "beta0": np.array(beta0_post, dtype=np.float64),
        "ll": np.array(ll_post, dtype=np.float64),
    }


# ── Run 4 chains in parallel ──────────────────────────────────────────────
num_chains = 4
cfg = ChainConfig(
    n_warmup=500,
    n_samples=1000,
    prior_loc=0.0,       # N(0, prior_scale) on theta=log(mu): centred on mu=1
    prior_scale=1.5,
    beta0_sd=0.1,        # tight prior on beta0: keeps it near logit(K_target)
    slice_w=1.5,
    slice_m=20,
    slice_max_steps=250,
    inner_latent_cycles=1,
    learn_p=True,
    p_prior_conc=500.0,  # very concentrated prior on p_mix near K_target
    project_affects_beta0=True,
    diag_every=100,
    data_flag=False,
)
# Use the POPULATION prevalence (not the selected-sample case rate) as the
# anchor for the beta0 prior and p_mix prior.  In the ascertained sample
# ~42 % are cases, but the correct beta0 = logit(K_target) ≈ -4.6.
# Anchoring tightly prevents the sampler from drifting to the wrong beta0.
p_obs = float(info["K_target"])   # population K, not selected-sample K

print(f"\n{'='*60}")
print(f"PG-Gibbs  (M={M}, n3={n3}, n2={n2}, {num_chains} chains × {cfg.n_samples} samples)")
print(f"Prior on theta=log(mu): N({cfg.prior_loc}, {cfg.prior_scale})")
print(f"beta0 anchored at logit(K_target)={np.log(p_obs/(1-p_obs)):.3f}  (beta0_sd={cfg.beta0_sd})")
print(f"{'='*60}")

jobs = [
    (cid, 42 + cid, cfg.__dict__, L_full, y_full, row_slices, p_obs)
    for cid in range(num_chains)
]

if __name__ == "__main__":
    ctx = mp.get_context("fork")
    lock = ctx.RLock()
    with ctx.Pool(
        processes=num_chains,
        initializer=tqdm.set_lock,
        initargs=(lock,),
    ) as pool:
        chains_out = pool.map(_run_chain, jobs)

    # ── Summary ────────────────────────────────────────────────────────────
    mu_all = np.concatenate([o["mu"] for o in chains_out])
    print(f"\nTrue mu = {info['mu']:.4f}")
    print(f"Posterior mu:  mean={mu_all.mean():.4f}  median={np.median(mu_all):.4f}"
          f"  sd={mu_all.std():.4f}"
          f"  5%-95%=[{np.percentile(mu_all,5):.4f}, {np.percentile(mu_all,95):.4f}]")

    az_data = az.convert_to_inference_data(
        {"mu": np.stack([o["mu"] for o in chains_out])}
    )
    summary = az.summary(az_data, var_names=["mu"])
    print(summary.to_string())

    # ── Plot ───────────────────────────────────────────────────────────────
    mu_max = max(5.0, float(np.percentile(mu_all, 99)) * 1.5)
    mu_grid = np.linspace(1e-4, mu_max, 500)

    prior_density = scipy_halfcauchy.pdf(mu_grid, scale=0.5)
    posterior_kde = gaussian_kde(mu_all)
    posterior_density = posterior_kde(mu_grid)

    prior_at_samples = scipy_halfcauchy.pdf(mu_all, scale=0.5)
    weights = 1.0 / np.maximum(prior_at_samples, 1e-300)
    weights /= weights.sum()
    likelihood_kde = gaussian_kde(mu_all, weights=weights)
    likelihood_density = likelihood_kde(mu_grid)
    likelihood_density /= np.trapezoid(likelihood_density, mu_grid)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mu_grid, prior_density, label="Prior  [HalfCauchy(0.5)]",
            color="gray", linestyle="--", linewidth=1.8)
    ax.plot(mu_grid, posterior_density, label="Posterior",
            color="steelblue", linewidth=2.2)
    ax.plot(mu_grid, likelihood_density,
            label="Likelihood  (posterior/prior, KDE)",
            color="darkorange", linewidth=2.2)
    ax.axvline(info["mu"], color="red", linestyle=":", linewidth=1.5,
               label=f"True mu = {info['mu']:.2f}")
    ax.set_xlabel("mu  (genetic information, nats)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "PG-Gibbs: prior, posterior, likelihood\n"
        f"Ascertained sibships  |  n3={n3} triplets, n2={n2} pairs  |  M={M}  |  "
        f"{num_chains} chains × {cfg.n_samples} samples",
        fontsize=10,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, mu_max)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig("pg_gibbs_mu_posterior.png", dpi=150)
    print("\nPlot saved to pg_gibbs_mu_posterior.png")
