"""
Run DiscreteHMCGibbs on the same simulated dataset as run_comparison.py and
generate comparison plots and summary statistics matching the PG-Gibbs output.

Requires 4 GPUs (Tesla V100 or similar).

Outputs:
  comparison_hmc_posterior.png
  comparison_hmc_trace.png
  comparison_hmc_pairs.png
"""
from __future__ import annotations

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"

import dataclasses
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")

MU_TRUE         = 2.0
K               = 0.01
SEED_DATA       = 42
N_FULLSIB_PAIRS = 400_000
N_FULLSIB_TRIPS = 200_000
N_HALFSIB_PAIRS =  40_000
N_UNRELATED     =   4_000

NUM_CHAINS  = 4
N_WARMUP    = 1000
N_SAMPLES   = 2000

# ── 1. Simulate (same parameters as run_comparison.py) ───────────────────────
print("=" * 60)
print("Simulating case-control dataset ...")
print("=" * 60)
from geneticinfo_functions import simulate_casecontrol_related, print_casecontrol_summary
y_sample, A_sample, L_sample, info, g_sample = simulate_casecontrol_related(
    n_fullsib_pairs=N_FULLSIB_PAIRS,
    n_fullsib_trips=N_FULLSIB_TRIPS,
    n_halfsib_pairs=N_HALFSIB_PAIRS,
    n_unrelated=N_UNRELATED,
    K=K, mu=MU_TRUE, seed=SEED_DATA,
    return_genotypic_values=True,
)
print_casecontrol_summary(info)
M = info["M"]
p_obs = float(np.mean(y_sample))

# ── 2. Build block structure ──────────────────────────────────────────────────
from polyagamma_gibbs import infer_blocks_from_L, BlockStructure
from pg_gibbs_vectorized import GroupedBlocks

L_np = np.asarray(L_sample, dtype=np.float64)
y_np = np.asarray(y_sample, dtype=np.float64)
perm_rows, perm_cols, row_slices, col_slices = infer_blocks_from_L(L_np, tol_rel=0.0)
L_perm = L_np[perm_rows, :][:, perm_cols]
y_perm = y_np[perm_rows]
bs = BlockStructure(L_perm, col_slices, row_slices)
gb = GroupedBlocks.from_block_structure(bs)

sizes_str = "  ".join(f"size-{s}:{gb.L_by_size[s].shape[0]}" for s in gb.sizes)
print(f"\nBlock structure:  M={gb.M}  blocks={bs.n_blocks}  {sizes_str}")
M_rel = sum(gb.L_by_size[s].shape[0] * s for s in gb.sizes if s >= 2)

# ── 3. Prepare HMC data (relatives only, same as PG-Gibbs) ───────────────────
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import arviz as az
from geneticinfo_functions import lr_discrete_blockdiag

numpyro.set_host_device_count(NUM_CHAINS)

hmc_sizes, hmc_L_list, hmc_y_list = [], [], []
y_rel_parts = []
for s in sorted(gb.sizes, reverse=True):
    if s < 2:
        continue
    hmc_sizes.append(s)
    hmc_L_list.append(jnp.array(gb.L_by_size[s], dtype=jnp.float64))
    y_block = y_perm[gb.idx_by_size[s]]
    hmc_y_list.append(jnp.array(y_block, dtype=jnp.float64))
    y_rel_parts.append(np.asarray(y_block))

y_rel = np.concatenate(y_rel_parts)
print(f"  HMC: M_rel={M_rel}  sizes={hmc_sizes}  "
      f"n_blocks={sum(L.shape[0] for L in hmc_L_list)}")

# ── 4. Run DiscreteHMCGibbs ──────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"DiscreteHMCGibbs  ({NUM_CHAINS} chains × {N_SAMPLES} samples, {N_WARMUP} warmup)")
print("=" * 60)

inner_kernel = NUTS(lr_discrete_blockdiag, max_tree_depth=8)
kernel = DiscreteHMCGibbs(inner_kernel, modified=True)
mcmc = MCMC(
    kernel,
    num_warmup=N_WARMUP,
    num_samples=N_SAMPLES,
    num_chains=NUM_CHAINS,
    chain_method="parallel",
    progress_bar=True,
)

t0 = time.perf_counter()
mcmc.run(
    random.PRNGKey(0),
    L_list=hmc_L_list,
    y_list=hmc_y_list,
    sizes=hmc_sizes,
    p_obs=p_obs,
    # muprior defaults to HalfCauchy(1) in lr_discrete_blockdiag
)
t_hmc = time.perf_counter() - t0

mcmc.print_summary(exclude_deterministic=True)

# ── 5. Extract samples ────────────────────────────────────────────────────────
samps = mcmc.get_samples(group_by_chain=True)
print("\nSampled variables:", list(samps.keys()))

mu_chains   = np.array(samps["mu"])    # (n_chains, n_samples)
p_chains    = np.array(samps["p"])     # (n_chains, n_samples)
n_ch, n_sa  = mu_chains.shape

# beta0 = logit(p)
beta0_chains = (np.log(np.clip(p_chains, 1e-9, 1 - 1e-9))
                - np.log(np.clip(1.0 - p_chains, 1e-9, 1 - 1e-9)))

# Log-likelihood: use sampled G if available, else β₀-only approximation
if "G" in samps:
    G_chains = np.array(samps["G"])          # (n_chains, n_samples, M_rel)
    eta = beta0_chains[:, :, None] + G_chains
    # Bernoulli log-likelihood in a numerically stable way
    ll_pos = -np.log1p(np.exp(-np.clip(eta,  -500, 500)))
    ll_neg = -np.log1p(np.exp( np.clip(eta,  -500, 500)))
    ll_chains = np.einsum("csm,m->cs", y_rel * ll_pos + (1.0 - y_rel) * ll_neg,
                          np.ones(M_rel))
    print("  Log-likelihood computed from sampled G.")
else:
    # Fallback: use only β₀ (conservative lower bound)
    n_cases = float(y_rel.sum())
    n_ctrl  = float(len(y_rel) - n_cases)
    ll_chains = (n_cases * np.log(np.clip(p_chains, 1e-9, 1 - 1e-9))
                 + n_ctrl  * np.log(np.clip(1.0 - p_chains, 1e-9, 1 - 1e-9)))
    print("  Note: G not in samples; log-likelihood approximated from p only.")

mu_all = mu_chains.ravel()

# ── 6. Build result dict compatible with geneticinfo plot functions ───────────
@dataclasses.dataclass
class _Cfg:
    n_warmup: int
    n_samples: int

result_hmc = {
    "mu_chains": mu_chains,
    "mu_all": mu_all,
    "chain_dicts": [
        {
            "mu":        mu_chains[c],
            "beta0":     beta0_chains[c],
            "p":         p_chains[c],
            "ll":        ll_chains[c],
            "mu_warmup": np.empty(0),   # warmup not collected
            "ll_warmup": np.empty(0),
            "chain_id":  c,
        }
        for c in range(n_ch)
    ],
    "cfg":           _Cfg(n_warmup=0, n_samples=n_sa),
    "mu_prior_scale": 1.0,
    "mu_prior_df":    1.0,   # half-Cauchy (df=1) — default prior in lr_discrete_blockdiag
    "block_info":    {},
}

# ── 7. Plots ──────────────────────────────────────────────────────────────────
from geneticinfo import summarize_and_plot, plot_trace, plot_pairs

summarize_and_plot(
    result_hmc,
    outfile="comparison_hmc_posterior.png",
    mu_true=MU_TRUE,
    title=f"DiscreteHMCGibbs  (simulated data, true μ={MU_TRUE})",
)
plot_trace(
    result_hmc,
    outfile="comparison_hmc_trace.png",
    title=f"DiscreteHMCGibbs",
)
plot_pairs(
    result_hmc,
    outfile="comparison_hmc_pairs.png",
    title=f"DiscreteHMCGibbs  (simulated data, true μ={MU_TRUE})",
)

# ── 8. Summary ────────────────────────────────────────────────────────────────
az_data = az.convert_to_inference_data({"mu": mu_chains})
summ    = az.summary(az_data, var_names=["mu"])
ess     = float(summ["ess_bulk"].iloc[0])
rhat    = float(summ["r_hat"].iloc[0])
ci90    = (float(np.percentile(mu_all, 5)), float(np.percentile(mu_all, 95)))

print(f"\n{'=' * 60}")
print("COMPARISON SUMMARY")
print(f"  True mu = {MU_TRUE}   K = {K}   M_rel = {M_rel}   seed = {SEED_DATA}")
print("=" * 60)
print(f"  {'Algorithm':<32}  {'median':>7}  {'sd':>6}  {'90% CI':>18}  "
      f"{'ESS':>6}  {'r_hat':>6}  {'wall':>8}")
print(f"  {'DiscreteHMCGibbs':<32}  {np.median(mu_all):>7.3f}  "
      f"{mu_all.std():>6.3f}  "
      f"  [{ci90[0]:.3f},{ci90[1]:.3f}]  "
      f"{ess:>6.0f}  {rhat:>6.3f}  {t_hmc:>7.1f}s")
print(f"\n  PG-Gibbs (collapsed_phi, from run_comparison.py):")
print(f"  {'PG-Gibbs (collapsed_phi)':<32}  {'2.055':>7}  {'0.542':>6}  "
      f"  [{'1.323'},{'3.094'}]  "
      f"{'424':>6}  {'1.010':>6}  {'119':>7}s")
print(f"\nWall time: {t_hmc:.1f} s  ({NUM_CHAINS} × {N_SAMPLES} samples, "
      f"{N_WARMUP} warmup, 4 × Tesla V100)")
