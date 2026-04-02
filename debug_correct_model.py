"""Diagnostic script: trace correct_model Gibbs loop for 20 iterations."""
import numpy as np
import os
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from geneticinfo_functions import simulate_casecontrol_related
from polyagamma_gibbs import infer_blocks_from_L, BlockStructure, ChainConfig
from pg_gibbs_vectorized import (
    GroupedBlocks, sparse_matvec,
    build_Lc_and_blike, sample_r_blocks_fast, sample_beta0_z_fast,
    build_theta_eig_cache_fast, theta_logpost_fast,
    sample_logit_p_slice_corrected,
)
from polyagamma import random_polyagamma
from polyagamma_gibbs import (
    update_c_gibbs_cpu, slice_sample_theta_collapsed_numpy,
)

# Small simulation for speed
np.random.seed(0)
y_sample, A_sample, L_sample, info, g_sample = simulate_casecontrol_related(
    n_fullsib_pairs=400_000,
    n_fullsib_trips=200_000,
    n_halfsib_pairs=40_000,
    n_unrelated=4_000,
    K=0.01, mu=1.0, seed=42,
    return_genotypic_values=True,
)
y_np = np.asarray(y_sample, dtype=np.float64)
L_np = np.asarray(L_sample, dtype=np.float64)

perm_rows, perm_cols, row_slices, col_slices = infer_blocks_from_L(L_np, tol_rel=0.0)
L_perm = L_np[perm_rows, :][:, perm_cols]
y_perm = y_np[perm_rows]

bs = BlockStructure(L_perm, col_slices, row_slices)
gb = GroupedBlocks.from_block_structure(bs)

# Build relatives-only GroupedBlocks
rel_sizes = sorted([s for s in gb.sizes if s >= 2], reverse=True)
rel_idx_ordered = np.concatenate([gb.idx_by_size[s] for s in rel_sizes])
y_rel = y_perm[rel_idx_ordered]

new_L_by_size = {}
new_idx_by_size = {}
offset = 0
for s in rel_sizes:
    n_s = gb.L_by_size[s].shape[0]
    new_L_by_size[s] = gb.L_by_size[s]
    new_idx_by_size[s] = np.arange(offset, offset + n_s * s)
    offset += n_s * s

M_rel = offset
gb_rel = GroupedBlocks(M=M_rel, sizes=rel_sizes,
                       L_by_size=new_L_by_size, idx_by_size=new_idx_by_size)

print(f"M_rel={M_rel}  sizes={rel_sizes}")
print(f"y_rel: {y_rel.sum():.0f} cases, {(1-y_rel).sum():.0f} controls")

# Setup
p0 = float(np.mean(y_sample))
K_prior = 20.0
BETA0_SD = float(np.sqrt(1.0 / (0.25 * 21.0)))
rng = np.random.default_rng(42)
M = M_rel
kappa = y_rel - 0.5

# Init
p_mix = float(np.mean(y_rel))
beta0 = float(np.log(p_mix + 1e-12) - np.log1p(p_mix + 1e-12))
theta = 0.0
mu_pos = 1.0
r = np.where(rng.uniform(size=M) < p_mix, 1.0, -1.0).astype(np.float64)
z_unc = mu_pos * r + np.sqrt(2.0 * mu_pos) * rng.normal(size=M)
z = (z_unc - z_unc.mean()).astype(np.float64)

# Precompute c_vec
c_vec = np.zeros(M)
for _s in gb_rel.sizes:
    _idx = gb_rel.idx_by_size[_s]
    _Ls = gb_rel.L_by_size[_s]
    c_vec[_idx] = _Ls.sum(axis=-1).ravel()
print(f"c_vec stats: min={c_vec.min():.3f} max={c_vec.max():.3f} mean={c_vec.mean():.3f}")

print(f"\n{'it':>3}  {'theta':>7}  {'mu':>6}  {'beta0':>7}  {'p_mix':>6}  {'cc':>6}  {'n+':>4}  {'eta_max':>8}  {'omega_min':>10}")
print("-" * 80)

for it in range(200):
    # Sample ω
    eta = beta0 + sparse_matvec(gb_rel, z)
    omega = random_polyagamma(1.0, np.abs(eta))

    mu_pos = float(np.exp(theta))
    cc = 0.5 * (2.0 * p_mix - 1.0)

    # Direct r update: p(r_i | Z_mix_i, β₀, p) ∝ N(Z_mix_i; r_i·μ, 2μ)·Bern(r_i; p)
    # log-odds(r_i=+1) = β₀ + Z_mix_i.  ω does not enter the r conditional given Z_mix.
    Z_mix_indiv = z + (2.0 * p_mix - 1.0) * mu_pos
    log_odds_r  = beta0 + Z_mix_indiv
    r = np.where(
        rng.uniform(size=M_rel) < 1.0 / (1.0 + np.exp(-log_odds_r)),
        1.0, -1.0).astype(np.float64)

    # Update z (β₀ fixed)
    _, z = sample_beta0_z_fast(
        rng, gb_rel, omega, kappa, r, mu_pos, BETA0_SD,
        beta0_fixed=beta0, no_centering=True, center_correction=cc)

    # Compute G_hat = L @ Z_mix
    G = sparse_matvec(gb_rel, z)
    G_hat = G + (2.0 * p_mix - 1.0) * mu_pos * c_vec

    n_plus = float((r > 0).sum())
    n_minus = float((r < 0).sum())
    p_mix_old = p_mix

    # Update β₀
    beta0_new = sample_logit_p_slice_corrected(
        rng, n_plus, n_minus, omega, kappa, G_hat, c_vec, mu_pos,
        K_prior=K_prior, p_obs_prior=p0, beta0_init=beta0)

    p_mix = float(1.0 / (1.0 + np.exp(-beta0_new)))

    # Re-centre z
    z += (p_mix_old - p_mix) * 2.0 * mu_pos
    beta0 = beta0_new

    # Update θ: use centred b_Z (center_correction=cc) with correct normalisation
    cc_theta = float(0.5 * (2.0 * p_mix - 1.0))
    C_norm   = n_plus * (1.0 - p_mix) ** 2 + n_minus * p_mix ** 2
    cache = build_theta_eig_cache_fast(
        gb_rel, omega, beta0, kappa, r, min_block_size=1,
        center_correction=cc_theta, C_norm=C_norm)
    f_th = lambda th: theta_logpost_fast(
        float(th), cache, 0.0, 1.0)
    theta = float(slice_sample_theta_collapsed_numpy(
        rng, theta, f_th, w=1.5, m=20, max_steps=250))

    mu_pos = np.exp(theta)
    cc_new = 0.5 * (2.0 * p_mix - 1.0)
    print(f"{it:>3}  {theta:>7.3f}  {mu_pos:>6.3f}  {beta0:>7.4f}  "
          f"{p_mix:>6.4f}  {cc_new:>6.4f}  {n_plus:>4.0f}  "
          f"{np.abs(eta).max():>8.3f}  {omega.min():>10.6f}")
