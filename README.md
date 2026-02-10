# geneticinfo

Bayesian inference of **genetic information for discrimination** from binary (case/control) traits.

The key quantity being estimated is Lambda — the expected log-likelihood ratio (in nats) for genetic prediction of disease status. Under the logistic polygenic model, Lambda = log(lambda\_S) where lambda\_S is the sibling recurrence risk ratio, and Lambda = 0.5 * s^2 where s is the scale of the class-conditional log-likelihood ratio distributions.

## Models

All models share the structure: observed binary outcome y is Bernoulli with logits = beta0 + G, where G = L @ Z incorporates genetic correlation via the Cholesky factor L of the genetic relationship matrix.

- **`logistic_mvnorm`** — Gaussian random effects: z ~ Normal(0,1), G = s * (L @ z). Default model for SVI fitting.
- **`logistic_mix2_mvnorm`** — Two-component mixture random effects using `MixtureSameFamily`, with the constraint mu = 0.5 * s^2.
- **`lr_discrete`** — Same mixture model with explicit discrete latent class indicators, for use with `DiscreteHMCGibbs`.

## Quick start

### Requirements

- JAX (with GPU support recommended)
- NumPyro
- NumPy, SciPy

### Running the example

```bash
python run_example.py
```

This will:

1. Simulate a case-control dataset from a population with full-sib pairs (correlation 0.5), half-sib pairs (correlation 0.25), and unrelated individuals. Default parameters: prevalence K=0.01, genetic information mu=1.0 nat.
2. Fit `logistic_mvnorm` via SVI with an `AutoLowRankMultivariateNormal` variational guide (rank 20, 15000 steps).
3. Print a posterior summary comparing estimated and true values of beta0, s, and mu.

### Using the module directly

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from geneticinfo import simulate_casecontrol_related, print_casecontrol_summary, fit_svi_lowrank

# Simulate data
y, A, L, info = simulate_casecontrol_related(K=0.01, mu=1.0)
print_casecontrol_summary(info)

# Fit model
M = info["M"]
result = fit_svi_lowrank(M, jnp.array(L), jnp.array(y, dtype=jnp.float64))

# Posterior samples are in result["s"], result["mu"], result["beta0"]
```

## Files

| File | Description |
|------|-------------|
| `geneticinfo.py` | Model definitions, SVI fitting, and simulation functions |
| `run_example.py` | End-to-end example script |
| `utils_mcmc.py` | Multi-page PDF diagnostic report for NUTS MCMC runs |

## License

See [LICENSE](LICENSE).
