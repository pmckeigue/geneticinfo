"""Compare logistic_mvnorm (NUTS) and logistic_mix2_mvnorm (SVI)
on simulated case-control data."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, RenyiELBO, Predictive
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from numpyro.optim import Adam
import numpy as np

from geneticinfo_functions import (
    simulate_casecontrol_related,
    print_casecontrol_summary,
    logistic_mvnorm,
    logistic_mix2_mvnorm,
)

# --- Simulate data (small for speed) ---
y_np, A_np, L_np, info = simulate_casecontrol_related(
    n_fullsib_pairs=10_000, n_halfsib_pairs=1_000, n_unrelated=10_000,
)
print_casecontrol_summary(info)

M = info["M"]
y = jnp.array(y_np, dtype=jnp.float64)
L = jnp.array(L_np, dtype=jnp.float64)

# ====================================================================
# 1. logistic_mvnorm with NUTS
# ====================================================================
print("\n" + "=" * 60)
print("1. logistic_mvnorm with NUTS")
print("=" * 60)

kernel = NUTS(logistic_mvnorm, max_tree_depth=10)
mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=1)
mcmc.run(random.PRNGKey(0), M=M, L=L, y=y)

samples_nuts = mcmc.get_samples()
s_nuts = samples_nuts["s"]
mu_nuts = 0.5 * s_nuts**2

print(f"\n  True mu = {info['mu']:.4f}, true s = {info['s']:.4f}")
print(f"  NUTS  s:  mean={float(s_nuts.mean()):.4f}, "
      f"median={float(jnp.median(s_nuts)):.4f}, "
      f"sd={float(s_nuts.std()):.4f}")
print(f"  NUTS  mu: mean={float(mu_nuts.mean()):.4f}, "
      f"median={float(jnp.median(mu_nuts)):.4f}, "
      f"sd={float(mu_nuts.std()):.4f}")

# n_eff and r_hat for s
mcmc.print_summary(prob=0.9, exclude_deterministic=False)

# ====================================================================
# 2. logistic_mix2_mvnorm with SVI
# ====================================================================
print("\n" + "=" * 60)
print("2. logistic_mix2_mvnorm with SVI (AutoLowRankMVN)")
print("=" * 60)

guide = AutoLowRankMultivariateNormal(logistic_mix2_mvnorm, rank=20)
optimizer = Adam(step_size=0.001)
elbo = RenyiELBO(alpha=0, num_particles=2)
svi = SVI(logistic_mix2_mvnorm, guide, optimizer, loss=elbo)

svi_result = svi.run(
    random.key(42), 15000, M=M, L=L, y=y, progress_bar=True
)

predictive = Predictive(guide, params=svi_result.params, num_samples=2000)
posterior_svi = predictive(random.PRNGKey(43))
s_svi = posterior_svi["s"]
mu_svi = 0.5 * s_svi**2

print(f"\n  True mu = {info['mu']:.4f}, true s = {info['s']:.4f}")
print(f"  SVI   s:  mean={float(s_svi.mean()):.4f}, "
      f"median={float(jnp.median(s_svi)):.4f}, "
      f"sd={float(s_svi.std()):.4f}")
print(f"  SVI   mu: mean={float(mu_svi.mean()):.4f}, "
      f"median={float(jnp.median(mu_svi)):.4f}, "
      f"sd={float(mu_svi.std()):.4f}")

# Final ELBO
print(f"  Final ELBO loss: {float(svi_result.losses[-1]):.2f}")
