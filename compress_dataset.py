"""
compress_dataset.py — Privacy-preserving compressed model for offline testing.

Workflow
--------
On the secure platform (once, after running sample_posterior):

    from compress_dataset import compress
    compress(L, y, posterior_result, "compressed_model.npz")

Offline (after downloading compressed_model.npz):

    from compress_dataset import generate_synthetic
    L_syn, y_syn = generate_synthetic("compressed_model.npz", seed=0)

Validation (needs real data — run on secure platform):

    from compress_dataset import validate
    validate(L, y, "compressed_model.npz")

Privacy test (run on secure platform):

    from compress_dataset import test_privacy
    test_privacy(L, y, "compressed_model.npz", n_test=20)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, gaussian_kde


# ── private helpers ───────────────────────────────────────────────────────────

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
        idxs = gb.idx_by_size[s]   # length n_s * s
        n_s  = len(idxs) // s
        for k in range(n_s):
            block_idxs = idxs[k * s:(k + 1) * s]
            L_perm[np.ix_(block_idxs, block_idxs)] = gb.L_by_size[s][k]
    return L_perm


def _gb_from_block_list(L_blocks: list[np.ndarray], block_sizes: np.ndarray):
    """
    Build a GroupedBlocks from an ordered list of L sub-matrices.

    Blocks are assumed to occupy consecutive rows/columns of a block-diagonal
    matrix in the order given (block 0 → rows 0..s0-1, block 1 → s0..s0+s1-1,
    etc.).  This matches the layout used by generate_synthetic().
    """
    from pg_gibbs_vectorized import GroupedBlocks

    by_size: dict[int, list] = {}
    offset = 0
    for k, s_raw in enumerate(block_sizes):
        s = int(s_raw)
        by_size.setdefault(s, []).append((offset, L_blocks[k]))
        offset += s

    sizes = sorted(by_size.keys())
    L_by_size  = {}
    idx_by_size = {}
    for s, entries in by_size.items():
        L_by_size[s]  = np.stack([e[1] for e in entries])   # (n_s, s, s)
        idx_by_size[s] = np.concatenate(
            [np.arange(e[0], e[0] + s) for e in entries]
        )

    M = int(sum(int(s) for s in block_sizes))
    return GroupedBlocks(M=M, sizes=sizes,
                         L_by_size=L_by_size, idx_by_size=idx_by_size)


def _load_block_list(model_file: str) -> tuple[list[np.ndarray], np.ndarray]:
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


def _pairwise_concordance(
    gb,
    y_perm: np.ndarray,
    bins: list[float],
) -> dict:
    """
    Compute within-block pairwise concordance P(y_i == y_j) by kinship bin.

    Only blocks of size >= 2 contribute pairs.  Kinship is A_{ij} = (L L^T)_{ij}.

    Parameters
    ----------
    gb      : GroupedBlocks
    y_perm  : (M,) binary outcomes aligned with gb
    bins    : upper edges of kinship bins (e.g. [0.1, 0.2, 0.3, 0.4, 0.6])

    Returns
    -------
    dict with keys:
      concordance : list of concordance rates per bin (nan if no pairs)
      n_pairs     : list of pair counts per bin
      bin_edges   : list of (lo, hi) for each bin
    """
    bin_edges = list(zip([0.0] + list(bins), list(bins) + [np.inf]))

    A_vals   = []
    conc_vals = []

    for s in gb.sizes:
        if s < 2:
            continue
        Ls   = gb.L_by_size[s]       # (n_s, s, s)
        idxs = gb.idx_by_size[s]     # (n_s * s,)
        n_s  = Ls.shape[0]
        for k in range(n_s):
            L_k   = Ls[k]
            A_k   = L_k @ L_k.T
            block_y = y_perm[idxs[k * s:(k + 1) * s]]
            for i in range(s):
                for j in range(i + 1, s):
                    A_vals.append(A_k[i, j])
                    conc_vals.append(int(block_y[i] == block_y[j]))

    A_arr    = np.array(A_vals)
    conc_arr = np.array(conc_vals)

    concordance = []
    n_pairs     = []
    for lo, hi in bin_edges:
        mask = (A_arr >= lo) & (A_arr < hi)
        n = int(np.sum(mask))
        n_pairs.append(n)
        concordance.append(float(np.mean(conc_arr[mask])) if n > 0 else float("nan"))

    return {"concordance": concordance, "n_pairs": n_pairs, "bin_edges": bin_edges}


# ── public API ────────────────────────────────────────────────────────────────

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
    from geninfo import build_blocks

    L = np.asarray(L, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    blocks    = build_blocks(L, y, corr_threshold=corr_threshold)
    gb        = blocks["gb"]
    block_info = blocks["block_info"]

    # Per-block L matrices in sorted-size order
    block_sizes_list: list[int] = []
    L_flat_parts: list[np.ndarray] = []
    for s in sorted(gb.sizes):
        Ls  = gb.L_by_size[s]        # (n_s, s, s)
        n_s = Ls.shape[0]
        for k in range(n_s):
            block_sizes_list.append(s)
            L_flat_parts.append(Ls[k].ravel())

    block_sizes  = np.array(block_sizes_list, dtype=np.int32)
    block_L_flat = np.concatenate(L_flat_parts).astype(np.float64)

    # Posterior samples: concatenate across chains
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
) -> tuple[np.ndarray, np.ndarray]:
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

    rng = np.random.default_rng(seed)

    # Draw one posterior sample (index determined by seed)
    n_post = len(mu_samples)
    idx    = int(seed % n_post)
    mu     = float(mu_samples[idx])
    beta0  = float(beta0_samples[idx])
    p      = float(p_samples[idx])

    # Reconstruct per-block L matrices
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
        # Class indicators r_i ∈ {-1, +1}, P(r_i=+1) = p
        r = 2 * rng.binomial(1, p, s) - 1
        # Latent z | r, mu ~ N(r*mu, 2*mu * I)
        z = rng.normal(r * mu, np.sqrt(max(2 * mu, 1e-10)), s)
        # Linear predictor and Bernoulli draw
        eta = beta0 + L_k @ z
        prob = _sigmoid(eta)
        y_syn[pos:pos + s] = rng.binomial(1, prob)
        pos += s

    # Assemble block-diagonal L_syn
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
    sampler_kwargs : forwarded to sample_posterior() (e.g. n_warmup, n_samples)
    """
    from geninfo import build_blocks, sample_posterior

    data          = np.load(model_file)
    mu_samples    = data["mu_samples"]
    K_model       = float(data["K"])
    mu_prior_scale = float(data["mu_prior_scale"])

    L_real = np.asarray(L_real, dtype=np.float64)
    y_real = np.asarray(y_real, dtype=np.float64)

    # Block structure of real data
    blocks_real = build_blocks(L_real, y_real)
    gb_real     = blocks_real["gb"]
    y_perm_real = blocks_real["y_perm"]

    real_sizes = np.array([
        s for s in gb_real.sizes for _ in range(gb_real.L_by_size[s].shape[0])
    ])

    print("=== Validation report ===\n")

    # ── 1 & 3. Block sizes and prevalence across synthetic datasets ──────────
    syn_sizes = data["block_sizes"].astype(int)          # same structure by construction
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

    # ── 2. Pairwise concordance ───────────────────────────────────────────────
    kin_bins = [0.1, 0.2, 0.3, 0.4, 0.6]
    real_conc = _pairwise_concordance(gb_real, y_perm_real, kin_bins)

    L_blocks, block_sizes_arr = _load_block_list(model_file)
    gb_syn = _gb_from_block_list(L_blocks, block_sizes_arr)
    _, y_syn0 = generate_synthetic(model_file, seed=0)
    syn_conc = _pairwise_concordance(gb_syn, y_syn0, kin_bins)

    print(f"\nPairwise concordance by kinship bin (relative pairs within blocks):")
    print(f"  {'Kinship range':<20}  {'Real':<8}  {'Synthetic':<10}  N_pairs")
    for b, ((lo, hi), rc, sc, np_) in enumerate(zip(
            real_conc["bin_edges"],
            real_conc["concordance"],
            syn_conc["concordance"],
            real_conc["n_pairs"])):
        hi_str = f"{hi:.2f}" if np.isfinite(hi) else "∞"
        rc_str = f"{rc:.3f}" if not np.isnan(rc) else "   —"
        sc_str = f"{sc:.3f}" if not np.isnan(sc) else "   —"
        print(f"  [{lo:.2f}, {hi_str})              {rc_str:<8}  {sc_str:<10}  {np_}")

    # ── 4. Posterior recovery ─────────────────────────────────────────────────
    print(f"\nPosterior recovery (sample_posterior on synthetic data):")
    kwargs = sampler_kwargs or {}
    L_syn, y_syn = generate_synthetic(model_file, seed=0)
    result_syn = sample_posterior(
        L_syn, y_syn,
        progress_bar=False,
        mu_prior_scale=mu_prior_scale,
        **kwargs,
    )
    mu_syn_mean   = float(np.mean(result_syn["mu_all"]))
    mu_real_mean  = float(np.mean(mu_samples))
    mu_real_90lo  = float(np.percentile(mu_samples, 5))
    mu_real_90hi  = float(np.percentile(mu_samples, 95))
    within_ci     = mu_real_90lo <= mu_syn_mean <= mu_real_90hi
    print(f"  Real posterior:      mean={mu_real_mean:.4f}  "
          f"90% CI=[{mu_real_90lo:.4f}, {mu_real_90hi:.4f}]")
    print(f"  Synthetic posterior: mean={mu_syn_mean:.4f}  "
          f"{'PASS' if within_ci else 'FAIL'} (within real 90% CI?)")

    # ── Figure ────────────────────────────────────────────────────────────────
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
    plt.show()


def test_privacy(
    L: np.ndarray,
    y: np.ndarray,
    model_file: str,
    n_test: int = 20,
    **sampler_kwargs,
) -> dict:
    """
    Membership inference test: compare P(y_i=1 | relatives, θ_full) vs
    P(y_i=1 | relatives, θ_loo) for n_test randomly chosen individuals who
    have at least one relative (block size ≥ 2).

    Prediction formula (plan approximation):
        eta_i ≈ beta0 + Σ_{j≠i} A_{ij} * (2*y_j - 1) * mu
        P(y_i=1 | relatives, theta) = sigmoid(eta_i)

    LOO procedure: remove individual i, re-run sample_posterior(), compare.

    For a well-regularised Bayesian model with M >> 1, each individual
    contributes O(1/M) to the posterior, so |P_full - P_loo| should be small
    relative to the posterior SD of mu.

    Parameters
    ----------
    L, y           : original data
    model_file     : path to compressed_model.npz
    n_test         : number of test individuals
    **sampler_kwargs : forwarded to sample_posterior() (e.g. n_warmup, n_chains)

    Returns
    -------
    dict: diffs, max_diff, mean_diff, p95_diff, mu_posterior_sd
    """
    from geninfo import build_blocks, sample_posterior

    L = np.asarray(L, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    data          = np.load(model_file)
    mu_samples    = data["mu_samples"]
    beta0_samples = data["beta0_samples"]
    mu_prior_scale = float(data["mu_prior_scale"])

    # Full-data block structure and posterior means
    blocks  = build_blocks(L, y)
    gb      = blocks["gb"]
    y_perm  = blocks["y_perm"]
    M       = gb.M

    mu_full    = float(np.mean(mu_samples))
    beta0_full = float(np.mean(beta0_samples))

    # Reconstruct permuted block-diagonal L for LOO datasets
    L_perm = _reconstruct_L_perm(gb)

    # Collect individuals with relatives
    candidates: list[tuple[int, int, int, int]] = []
    for s in gb.sizes:
        if s < 2:
            continue
        idxs = gb.idx_by_size[s]    # length n_s * s
        n_s  = len(idxs) // s
        for k in range(n_s):
            for local_i in range(s):
                candidates.append((int(idxs[k * s + local_i]), s, k, local_i))

    if not candidates:
        print("No individuals with relatives (all blocks are singletons).")
        return {}

    rng    = np.random.default_rng(42)
    chosen = rng.choice(len(candidates), min(n_test, len(candidates)), replace=False)
    test_set = [candidates[i] for i in chosen]

    print(f"Privacy test: {len(test_set)} of {len(candidates)} individuals with relatives\n")

    diffs = []
    for global_i, s, k, local_i in test_set:
        L_k        = gb.L_by_size[s][k]                           # (s, s)
        A_k        = L_k @ L_k.T                                   # (s, s)
        block_idxs = gb.idx_by_size[s][k * s:(k + 1) * s]
        block_y    = y_perm[block_idxs]

        # Full-data prediction for individual i given relatives
        eta_full = beta0_full
        for local_j in range(s):
            if local_j == local_i:
                continue
            eta_full += A_k[local_i, local_j] * (2 * float(block_y[local_j]) - 1) * mu_full
        P_full = float(_sigmoid(np.array([eta_full]))[0])

        # LOO posterior: remove individual i from the permuted dataset
        keep  = np.ones(M, dtype=bool)
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

        # LOO prediction (same formula with LOO parameters)
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
