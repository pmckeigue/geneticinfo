# geneticinfo

Bayesian inference of **genetic information for discrimination** from binary (case/control) traits.

The key quantity being estimated is Λ (lambda) — the expected log-likelihood ratio (in nats) favouring case over non-case status, given an individual's genetic risk (McKeigue, 2019). This gives the maximum expected information for discrimination that could be obtained from a polygenic risk score.  If the class-conditional distribution of the log-likelihood ratio favouring case over control status in controls is Gaussian with mean -Λ, the distribution in cases is also Gaussian with mean Λ, and both distributions have variance 2Λ. If the disease is rare (risk < 1%) and Λ is not very large (< 1 natural log unit), Λ = log(λ<sub>S</sub>) where λ<sub>S</sub> is the sibling recurrence risk ratio (Clayton, 2009).

## Models

All models share the structure: observed binary outcome y is Bernoulli with logits = β₀ + G, where G = L @ Z incorporates genetic correlation via the Cholesky factor L of the genetic relationship matrix.

- **`logistic_mvnorm`** — Gaussian random effects: z ~ Normal(0,1), G = s (L @ z). This model would be appropriate where sampling is based on a total population, rather than a case-control sample.
- **`logistic_mix2_mvnorm`** — Two-component mixture random effects using `MixtureSameFamily`, with the constraint μ = 0.5 s².
- **`lr_discrete`** — Same mixture model with explicit discrete latent class indicators, for use with `DiscreteHMCGibbs`.
- **`lr_discrete_blockdiag`** — Block-diagonal variant of `lr_discrete` that operates on batched per-family Cholesky factors after removing singletons, for efficient MCMC on large samples.

---

## Sampling algorithm 1: DiscreteHMCGibbs (NumPyro / JAX)

### Overview

The `lr_discrete_blockdiag` model is fitted with NumPyro's `DiscreteHMCGibbs` (modified=True, NUTS inner kernel). The class indicators r_i ∈ {−1, +1} are discrete latent variables; `DiscreteHMCGibbs` alternates between enumerating over these exactly and running NUTS for the continuous parameters (μ, s, β₀).

**Key preprocessing — reduction to relatives only.**  Unrelated individuals (singletons in the genetic relationship matrix) contribute no information about λ_S and are discarded before fitting. The functions `find_blocks()` and `reduce_to_relatives()` extract connected components, reducing the data from M to M_rel ≪ M.

### Running the example

```bash
python test_discrete_gibbs_large.py
```

#### Simulated dataset

A population of 884,000 individuals is generated containing 400,000 full-sib pairs (genetic correlation 0.5), 40,000 half-sib pairs (correlation 0.25), and 4,000 unrelated individuals. Disease status follows a logistic model calibrated to prevalence K = 0.01 with genetic information μ = 1.0 nat. The case-control sample retains all cases and an equal number of controls.

```
=== Population ===
  Total individuals:          884,000
    Full-sib pairs:           400,000
    Half-sib pairs:            40,000
    Unrelated:                  4,000
  Cases in population:          8,963
  Observed prevalence:   0.010139  (target: 0.0100)
  Concordant-affected full-sib pairs: 77

=== Sibling recurrence risk ratio (λ_S) ===
  Theoretical exp(μ):   2.7183
  Empirical (full sibs): 1.8694
  Ratio empirical/theoretical: 0.6877

=== Case-control sample ===
  Sample size M:          17926
    Cases:                 8963
    Controls:              8963
  Full-sib pairs (both sampled):  199
    of which both affected:       77
  Half-sib pairs (both sampled):  15

=== True parameters ===
  μ       = 1.0000  (expected log-LR, nats)
  s        = 1.4142
  α        = -5.5465  (intercept)
  K        = 0.0100
```

#### Block-diagonal reduction

```
=== Block decomposition ===
  Original M:       17926
  Reduced M:        428
  Number of blocks: 214
  Block size:       2
  Reduction ratio:  0.024
```

The 17,926 × 17,926 relationship matrix is reduced to 214 independent 2×2 blocks (428 individuals in relative pairs). The per-block Cholesky factors are stored as a `(214, 2, 2)` array and the matmul is done via `jnp.einsum`, replacing the original O(M²) dense matmul.

#### Posterior summary (4 chains, 2000 warmup + 2000 samples)

```
                     mean       std    median      5.0%     95.0%     n_eff     r_hat
             mu      2.25      1.58      1.75      0.37      4.54    207.30      1.00
              p      0.63      0.04      0.63      0.57      0.70    939.44      1.01
              s      2.01      0.67      1.87      1.00      3.09    204.63      1.00
```

True values: μ = 1.00, s = 1.41.  The true μ is within the 90% credible interval [0.37, 4.54].  Wall time: ~2.5 minutes on 4 × Tesla V100-SXM2-32GB GPUs.

### Usage

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from jax import random
from geneticinfo import (
    simulate_casecontrol_related, print_casecontrol_summary,
    lr_discrete_blockdiag, reduce_to_relatives,
)

y, A, L, info = simulate_casecontrol_related(
    n_fullsib_pairs=400_000, n_halfsib_pairs=40_000, n_unrelated=4000,
)
y_red, A_red, L_red, kept_idx, block_sizes = reduce_to_relatives(A, y)
block_size = block_sizes[0]
n_blocks = len(block_sizes)

import numpy as np
L_blocks = np.zeros((n_blocks, block_size, block_size))
offset = 0
for i in range(n_blocks):
    L_blocks[i] = L_red[offset:offset + block_size, offset:offset + block_size]
    offset += block_size

numpyro.set_host_device_count(4)
inner_kernel = NUTS(lr_discrete_blockdiag, max_tree_depth=8)
kernel = DiscreteHMCGibbs(inner_kernel, modified=True)
mcmc = MCMC(kernel, num_warmup=2000, num_samples=2000,
            num_chains=4, chain_method="parallel")
mcmc.run(random.PRNGKey(0),
         L_blocks=jnp.array(L_blocks), y_flat=jnp.array(y_red, dtype=jnp.float64),
         n_blocks=n_blocks, block_size=block_size, p_obs=float(y.mean()))
mcmc.print_summary()
```

---

## Sampling algorithm 2: Pólya-gamma Gibbs sampler (NumPy / CPU)

### Overview

An alternative Gibbs sampler is implemented in `pg_gibbs_vectorized.py` using the Pólya-gamma (PG) data-augmentation scheme of Polson, Scott and Windle (2013).  Introducing per-individual auxiliary variables ω_i ~ PG(1, |η_i|) renders the logistic likelihood conditionally Gaussian, enabling exact block Gibbs updates for all continuous latents.

Unlike `DiscreteHMCGibbs`, the PG sampler:

- **Operates on the full case-control sample** (M individuals, including singletons), keeping singletons for the β₀ and z updates while excluding them from the μ likelihood.
- **Collapses z out analytically** when sampling μ: the precision matrix of each family block is diagonalised via an eigenvalue cache, and the slice sampler for θ = log μ evaluates the collapsed log-posterior in O(M_rel) time.
- **Handles mixed block sizes** (pairs, triplets, …) with vectorised routines grouped by size; singletons are batched in a single NumPy pass.
- **Runs on CPU** via `multiprocessing.Pool` with fork, with no JAX or GPU dependency.

#### Singleton exclusion from the μ likelihood

The collapsed log-posterior for θ contains terms −¼ M μ − ½ M log(2μ) arising from the z-prior normalisation. For singleton blocks these terms carry no information about λ_S but impose a strong downward pull when M ≫ M_rel. The sampler therefore restricts this sum to blocks of size ≥ 2, using only the M_rel ≪ M related individuals for the μ update.

#### Performance

A key implementation detail: `np.dot` on arrays of length ~14,000 triggered OpenBLAS multi-thread wakeup (~50 ms per call) because the thread pool slept between Gibbs steps. Replacing three `np.dot` calls with element-wise `.sum()` operations (which bypass BLAS) reduced `sample_beta0_z_fast` from 54 ms to 1.1 ms per call, yielding a 38× overall speedup (3.8 → 145 iterations/second for M = 14,736).

### Running the example

```bash
python run_pg_gibbs_simdata.py
```

This simulates a large case-control dataset with full-sib pairs, full-sib triplets, and half-sib pairs; plots genotypic-value densities in cases vs controls; builds the block structure; and runs 4 chains in parallel.

#### Simulated dataset (× 2 population)

```
=== Population ===
  Total individuals:        1,484,000
    Full-sib pairs:           400,000
    Full-sib triplets:        200,000
    Half-sib pairs:            40,000
    Unrelated:                  4,000

=== Case-control sample ===
  Sample size M:          29,496
    Cases:                14,748
    Controls:             14,748
  Related individuals (block size >= 2):  1,120
  Blocks: 557 pairs + 2 triplets
```

#### Posterior summary (4 chains, 1000 warmup + 5000 samples, CPU)

```
     mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
mu  0.632  0.219   0.225    1.044      0.007    0.004    1027.0    1916.0    1.0
```

True value: μ = 1.00.  90% CI: [0.291, 1.006].  Wall time: ~1 min 43 s on CPU (4 cores).

---

## Comparison on the same dataset

Both algorithms were run on the same case-control sample (seed = 42) from the population described above (1,484,000 individuals; 400k full-sib pairs, 200k full-sib triplets, 40k half-sib pairs, 4k unrelated; K = 0.01, true μ = 1.0).  The script `run_comparison.py` simulates once and passes the same block structure to both samplers.

**Data summary**

```
M = 29,496  (14,748 cases, 14,748 controls)
Related individuals (block size ≥ 2):  1,120
  Blocks: 557 pairs + 2 triplets
```

DiscreteHMCGibbs operates on the 557 size-2 blocks (M_rel = 1,114); the 2 size-3 triplet blocks (6 individuals, 0.5% of M_rel) are excluded because `lr_discrete_blockdiag` requires uniform block size.  PG-Gibbs uses all 557 pairs and both triplets for the μ likelihood, while retaining all M = 29,496 individuals for the β₀ and z updates.

**Results (true μ = 1.0)**

```
Algorithm             median     sd     90% CI          ESS    r_hat   wall time
PG-Gibbs              0.616   0.227  [0.272, 1.026]    879    1.010      107 s  (CPU, 4 cores)
DiscreteHMCGibbs      1.139   0.487  [0.594, 2.112]    311    1.000      331 s  (GPU, 4×Tesla V100)
```

Both 90% credible intervals contain the true value μ = 1.0.

![Prior, posterior and likelihood for both algorithms](comparison_prior_posterior_likelihood.png)

**Interpretation.**  DiscreteHMCGibbs produces a wider posterior centred closer to the true value; its posterior median (1.139) is slightly above the truth.  PG-Gibbs produces a narrower posterior whose median (0.616) is below the truth, with the true value lying at the 95th percentile of the posterior.  Several factors contribute to this difference:

- *Intercept handling.*  In `lr_discrete_blockdiag`, β₀ is a deterministic function of the mixing proportion p (β₀ = logit(p)), with p ~ Beta(10, 10) centred at the full-sample case proportion 0.5.  This prevents β₀ from freely absorbing the ascertainment-induced case enrichment among relatives (~67% cases in the relatives-only subset).  In PG-Gibbs, β₀ is sampled jointly with z via a Schur complement and is constrained indirectly by the singletons (28,376 individuals with 50% case rate), which anchor β₀ near zero through the full-sample β₀ posterior.

- *Mixing.*  The PG-Gibbs IAT ≈ 23 (20,000 samples / ESS 879) is lower than DiscreteHMCGibbs IAT ≈ 26 (8,000 samples / ESS 311), reflecting faster mixing from the collapsed μ update.

- *Posterior concentration.*  With K = 0.01 and 1:1 case-control matching, most relative pairs in the sample are discordant (one case, one control), which carry less information about λ_S than concordant-affected pairs.  The half-Cauchy(1) prior, which has substantial mass at small μ, therefore has considerable influence on both posteriors.

## Algorithm comparison

| Feature | DiscreteHMCGibbs | PG-Gibbs |
|---|---|---|
| Framework | NumPyro / JAX | NumPy (pure CPU) |
| Hardware | GPU (4 × Tesla V100) | CPU (4 cores) |
| Individuals used | M_rel only (singletons discarded) | All M (singletons excluded from μ update only) |
| Mixed block sizes | No (same-size blocks required) | Yes (pairs, triplets, … grouped by size) |
| Discrete latents r_i | Gibbs via DiscreteHMCGibbs | Collapsed for μ; exact enumeration for z |
| Continuous update | NUTS (joint μ, s, β₀) | Slice sampler on θ = log μ; Gaussian draw for β₀, z |
| Prior on μ | Half-Cauchy(1) | Half-Cauchy(1) |
| μ posterior median (true = 1.0) | 1.139 | 0.616 |
| μ 90% CI | [0.594, 2.112] | [0.272, 1.026] |
| ESS (bulk) | 311 from 8,000 samples | 879 from 20,000 samples |
| IAT | ~26 | ~23 |
| Wall time (same dataset) | 331 s | 107 s |

---

## Files

| File | Description |
|---|---|
| `geneticinfo.py` | Model definitions, block-diagonal preprocessing, SVI fitting, and simulation |
| `pg_gibbs_vectorized.py` | Vectorised PG-Gibbs sampler: grouped block operations, collapsed θ update |
| `polyagamma_gibbs.py` | Lower-level PG-Gibbs utilities: slice sampler, block structure, PG helpers |
| `run_pg_gibbs_simdata.py` | PG-Gibbs on `simulate_casecontrol_related()` dataset; plots genotypic densities |
| `run_pg_gibbs_create_data.py` | PG-Gibbs on `create_data()` small dataset |
| `run_comparison.py` | Run both algorithms on the same simulated dataset; print side-by-side summary |
| `test_discrete_gibbs_large.py` | End-to-end DiscreteHMCGibbs: simulate, reduce, fit, plot |
| `pg_gibbs_report.md` / `.pdf` | Detailed technical report with model description and results |

## References

- Clayton DG (2009). Prediction and interaction in complex disease genetics: experience in type 1 diabetes. *PLoS Genetics*, 5(7), e1000540.
- McKeigue PM (2019). Quantifying performance of a diagnostic test as the expected information for discrimination: relation to the C-statistic. *Statistical Methods in Medical Research*, 28(6), 1841–1851.
- Polson NG, Scott JG, Windle J (2013). Bayesian inference for logistic models using Pólya-gamma latent variables. *Journal of the American Statistical Association*, 108(504), 1339–1349.

## License

See [LICENSE](LICENSE).
