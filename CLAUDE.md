# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for Bayesian inference of **genetic information for discrimination** from binary (case/control) traits. The core statistical problem is estimating the expected log-likelihood ratio (weight of evidence) for genetic prediction of disease status, parameterized through a logistic mixed model with correlated random effects derived from genetic relationship matrices.

The key quantity being estimated is **μ** (expected log-likelihood ratio in nats), which equals `log(λ_S)` where `λ_S` is the sibling recurrence risk ratio.

## Key Statistical Model

All models share the structure: observed binary outcome `y` is Bernoulli with logits = `beta0 + G`, where `G = L @ Z` incorporates genetic correlation via the Cholesky factor `L` of the genetic relationship matrix.

The mixture model (`lr_discrete` / `lr_discrete_blockdiag`) uses a two-component mixture for the random effects:
- Discrete class indicator `r_i ∈ {−1, +1}` with mixing probability `p = sigmoid(phi)`
- Conditional distribution `Z_mix | r, mu ~ N(mu*r, 2*mu*I)`
- The constraint `mu = 0.5 * s^2` enforces that variance equals twice the mean for the log-LR distribution

## Primary Python API: `geneticinfo.py`

The main entry point is `sample_posterior()`. The **default algorithm** is PG-Gibbs with:
- **Prior**: half-Cauchy(scale=1) on μ (`mu_prior_df=1.0`, `mu_prior_scale=1.0`)
- **Algorithm**: `use_collapsed_phi=True` — integrates out Z_mix before slicing on μ and φ
- **Heavy-tail fix**: `use_cauchy_aux` is auto-enabled when `mu_prior_df < 10`, replacing the heavy-tailed marginal prior with a scale-mixture auxiliary variable (μ | v ~ HalfNormal(0, v), v ~ IG(1/2, 1/2)) so the slice sampler sees a light-tailed conditional at each step
- **ASIS**: `use_asis=False` (default) — ASIS adds negligible ESS gain when `collapsed_phi` is active

```python
from geneticinfo import sample_posterior, summarize_and_plot, plot_trace, plot_pairs

result = sample_posterior(
    L,          # (M, M) lower-triangular Cholesky factor of GRM
    y,          # (M,) binary outcomes
    n_warmup=1000, n_samples=5000, n_chains=4,
)
summarize_and_plot(result, mu_true=2.0, outfile="posterior.png")
plot_trace(result, outfile="trace.png")
plot_pairs(result, outfile="pairs.png")
```

### `sample_posterior()` key parameters

| Parameter | Default | Notes |
|---|---|---|
| `mu_prior_df` | `1.0` | df=1 → half-Cauchy; df≥10 → approximately Gaussian |
| `mu_prior_scale` | `1.0` | Scale of the half-Student-t prior on μ |
| `use_collapsed_phi` | `True` | Integrate out Z_mix for both μ and φ slice steps |
| `use_cauchy_aux` | `None` | Auto-enabled when `collapsed_phi=True` and `mu_prior_df < 10` |
| `use_asis` | `False` | ASIS NCP interweaving step (no benefit with collapsed_phi) |
| `jags_adapt` | `False` | JAGS-style adaptation of slice step widths during warmup |
| `n_warmup_phase1` | `0` | Fixed-width burn-in iterations before JAGS adaptation begins |

## Core Sampler Modules

### `pg_gibbs_clean.py` — `LRDiscreteBlockdiagPGGibbs`

The main sampler class. Each `step()` call performs:
1. Resample PG augmentation variables `omega | eta`
2. Resample `Z_mix | omega, y, phi, mu, r`
3. Resample indicators `r | Z_mix, phi` (exact Bernoulli)
4. Update `phi | rest` (slice, optionally collapsed over Z_mix)
5. Update `theta = log(mu) | rest` (slice, collapsed over Z_mix via eigendecomposition)
   - If `use_cauchy_aux`: first resample `v | mu ~ IG(1, (mu²+1)/2)` and use HalfNormal(0,v) conditional prior
6. Refresh `Z_mix` under the new μ

Key functions:
- `build_mu_collapsed_cache()` — eigendecomposition of K = L^T Ω L; shared by μ and φ collapsed updates
- `logpost_theta_mu_collapsed(theta, cache, aux_v=0.0)` — log-posterior for the collapsed μ slice step
- `step_asis()` — NCP interweaving step; uses `aux_v` when `use_cauchy_aux=True`
- `LRBlockdiagPGConfig` — sampler configuration dataclass (defaults: `mu_prior_df=1.0`, `use_collapsed_phi=True`)

### `pg_gibbs_vectorized.py` — `GroupedBlocks`

Block-by-size storage for the block-diagonal GRM Cholesky factor. Groups blocks of the same size for vectorised operations.

### `polyagamma_gibbs.py`

Lower-level utilities: `slice_sample_1d()`, `BlockStructure`, `infer_blocks_from_L()`, PG sampling helpers.

### `geneticinfo_functions.py`

Model definitions for NumPyro (`lr_discrete_blockdiag`), simulation (`simulate_casecontrol_related`), and preprocessing utilities.

## Alternative Sampler: DiscreteHMCGibbs (NumPyro / JAX)

`lr_discrete_blockdiag` in `geneticinfo_functions.py` can be fitted with NumPyro's `DiscreteHMCGibbs` (NUTS inner kernel). This requires 4 GPUs and achieves ~9× lower ESS/s than PG-Gibbs on the same dataset and prior. See `run_hmc_comparison.py`.

## Comparison Scripts

- **`run_comparison.py`** — Compares PG-Gibbs with and without ASIS on simulated data (same prior and block structure). Outputs posterior/trace/pairs plots per condition.
- **`run_hmc_comparison.py`** — Head-to-head comparison of PG-Gibbs (collapsed_phi, half-Cauchy) vs DiscreteHMCGibbs with identical prior. Requires 4 GPUs for the HMC section.

## Performance (simulated data, M_rel=1374, true μ=2.0, half-Cauchy prior)

```
Algorithm                  ESS   ESS/s   R-hat   wall     hardware
PG-Gibbs (collapsed_phi)   454    3.50   1.000    130s    CPU, 4 cores
DiscreteHMCGibbs           261    0.37   1.010    707s    GPU, 4× V100
```

## Environment Notes

- Python environment: `/home/pmckeigue/venv/bin/python3`
- JAX 64-bit mode is enabled: `jax.config.update("jax_enable_x64", True)`
- GPU memory fraction: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.15`
- Multi-chain parallelism: PG-Gibbs uses `multiprocessing.Pool` (fork); HMC uses `numpyro.set_host_device_count(4)` with `chain_method="parallel"`
- The `infohsq.R` script plots expected information (bits) vs heritability on the liability scale
