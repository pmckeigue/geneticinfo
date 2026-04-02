"""Check the theta log-posterior shape with and without center_correction."""
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
)
from polyagamma import random_polyagamma
from polyagamma_gibbs import update_c_gibbs_cpu

np.random.seed(0)
y_sample, A_sample, L_sample, info, g_sample = simulate_casecontrol_related(
    n_fullsib_pairs=400_000, n_fullsib_trips=200_000,
    n_halfsib_pairs=40_000, n_unrelated=4_000,
    K=0.01, mu=1.0, seed=42, return_genotypic_values=True,
)
y_np = np.asarray(y_sample, dtype=np.float64)
L_np = np.asarray(L_sample, dtype=np.float64)

perm_rows, perm_cols, row_slices, col_slices = infer_blocks_from_L(L_np, tol_rel=0.0)
L_perm = L_np[perm_rows, :][:, perm_cols]
y_perm = y_np[perm_rows]
bs = BlockStructure(L_perm, col_slices, row_slices)
gb = GroupedBlocks.from_block_structure(bs)

rel_sizes = sorted([s for s in gb.sizes if s >= 2], reverse=True)
rel_idx_ordered = np.concatenate([gb.idx_by_size[s] for s in rel_sizes])
y_rel = y_perm[rel_idx_ordered]

new_L_by_size, new_idx_by_size = {}, {}
offset = 0
for s in rel_sizes:
    n_s = gb.L_by_size[s].shape[0]
    new_L_by_size[s] = gb.L_by_size[s]
    new_idx_by_size[s] = np.arange(offset, offset + n_s * s)
    offset += n_s * s
M_rel = offset
gb_rel = GroupedBlocks(M=M_rel, sizes=rel_sizes,
                       L_by_size=new_L_by_size, idx_by_size=new_idx_by_size)

kappa = y_rel - 0.5
rng = np.random.default_rng(42)

# Fix r to n_plus=680, n_minus=440 (roughly what we see after drift)
# to isolate the theta logpost shape
n_plus_target = 680
r = np.concatenate([np.ones(n_plus_target), -np.ones(M_rel - n_plus_target)])
rng.shuffle(r)
r = r.astype(np.float64)
n_plus = (r > 0).sum()
n_minus = (r < 0).sum()
print(f"n+={n_plus:.0f}, n-={n_minus:.0f}, M={M_rel}")

# Fix beta0 and z near correct values
p_mix = 0.63
beta0 = np.log(p_mix / (1 - p_mix))
print(f"p_mix={p_mix}, beta0={beta0:.4f}")

mu_true = 1.0
theta_true = np.log(mu_true)
z = (r * mu_true + np.sqrt(2.0 * mu_true) * rng.normal(size=M_rel))
z -= z.mean()

# Sample omega at the correct eta
eta = beta0 + sparse_matvec(gb_rel, z)
omega = random_polyagamma(1.0, np.abs(eta))
print(f"omega: mean={omega.mean():.4f}, min={omega.min():.6f}")

# Compute c_vec
c_vec = np.zeros(M_rel)
for _s in gb_rel.sizes:
    _idx = gb_rel.idx_by_size[_s]
    _Ls = gb_rel.L_by_size[_s]
    c_vec[_idx] = _Ls.sum(axis=-1).ravel()

cc = 0.5 * (2.0 * p_mix - 1.0)
print(f"cc = {cc:.4f}")

# Evaluate theta logpost at many theta values
zmix_correction = (2.0 * p_mix - 1.0) * 1.0  # (2p-1)*mu_pos at mu=1

print("\n--- theta logpost comparison ---")
print(f"{'theta':>7}  {'mu':>6}  {'lp_nocc':>10}  {'lp_withcc':>11}  {'lp_zmix':>9}  {'zmix-nocc':>10}")

theta_vals = np.linspace(-4, 2, 30)
for theta in theta_vals:
    # Without center_correction (original approach)
    cache_nocc = build_theta_eig_cache_fast(
        gb_rel, omega, beta0, kappa, r, min_block_size=1, center_correction=0.0)
    lp_nocc = theta_logpost_fast(theta, cache_nocc, 0.0, 1.0)

    # With center_correction (current buggy correct_model approach)
    cache_cc = build_theta_eig_cache_fast(
        gb_rel, omega, beta0, kappa, r, min_block_size=1, center_correction=cc)
    lp_cc = theta_logpost_fast(theta, cache_cc, 0.0, 1.0)

    # With zmix_correction (new correct approach)
    cache_zmix = build_theta_eig_cache_fast(
        gb_rel, omega, beta0, kappa, r, min_block_size=1,
        zmix_correction=zmix_correction, c_vec=c_vec)
    lp_zmix = theta_logpost_fast(theta, cache_zmix, 0.0, 1.0)

    mu = np.exp(theta)
    print(f"{theta:>7.3f}  {mu:>6.3f}  {lp_nocc:>10.3f}  {lp_cc:>11.3f}  {lp_zmix:>9.3f}  {lp_zmix - lp_nocc:>10.3f}")
