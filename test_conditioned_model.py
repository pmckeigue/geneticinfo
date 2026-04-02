"""Test logistic_mix2_conditioned with NUTS on simulated case-control data."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.infer import MCMC, NUTS
import numpy as np

from geneticinfo_functions import (
    simulate_casecontrol_related,
    print_casecontrol_summary,
    logistic_mix2_conditioned,
    logistic_mvnorm,
)

# --- Simulate data ---
y_np, A_np, L_np, info = simulate_casecontrol_related(
    n_fullsib_pairs=10_000, n_halfsib_pairs=1_000, n_unrelated=10_000,
)
print_casecontrol_summary(info)

M = info["M"]
y = jnp.array(y_np, dtype=jnp.float64)
L = jnp.array(L_np, dtype=jnp.float64)

# --- Run NUTS on logistic_mix2_conditioned ---
print("\n" + "=" * 60)
print("Fitting logistic_mix2_conditioned with NUTS")
print("=" * 60)

kernel = NUTS(logistic_mix2_conditioned, max_tree_depth=10)
mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=1)
mcmc.run(random.PRNGKey(0), M=M, L=L, y=y)
mcmc.print_summary(exclude_deterministic=False)

samples = mcmc.get_samples()
mu_samples = samples["mu"]
s_samples = samples["s"]
print(f"\nTrue mu = {info['mu']:.4f}")
print(f"Posterior mu: mean={float(mu_samples.mean()):.4f}, "
      f"median={float(jnp.median(mu_samples)):.4f}, "
      f"sd={float(mu_samples.std()):.4f}")
print(f"Posterior s:  mean={float(s_samples.mean()):.4f}, "
      f"median={float(jnp.median(s_samples)):.4f}, "
      f"sd={float(s_samples.std()):.4f}")
