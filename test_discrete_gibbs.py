"""Test lr_discrete with DiscreteHMCGibbs on simulated case-control data."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import numpy as np

from geneticinfo_functions import simulate_casecontrol_related, print_casecontrol_summary, lr_discrete

# --- Use the smaller simulation first for speed ---
y_np, A_np, L_np, info = simulate_casecontrol_related(
    n_fullsib_pairs=10_000, n_halfsib_pairs=1_000, n_unrelated=10_000,
)
print_casecontrol_summary(info)

M = info["M"]
y = jnp.array(y_np, dtype=jnp.float64)
L = jnp.array(L_np, dtype=jnp.float64)

print(f"\n{'='*60}")
print(f"lr_discrete with DiscreteHMCGibbs  (M={M})")
print(f"{'='*60}")

inner_kernel = NUTS(lr_discrete, max_tree_depth=8)
kernel = DiscreteHMCGibbs(inner_kernel, modified=True)
mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=1)
mcmc.run(random.PRNGKey(0), M=M, L=L, y=y)

samples = mcmc.get_samples()
mu_samples = samples["mu"]
s_samples = samples["s"]
p_samples = samples["p"]

print(f"\n  True: mu={info['mu']:.4f}, s={info['s']:.4f}")
print(f"  mu:  mean={float(mu_samples.mean()):.4f}, "
      f"median={float(jnp.median(mu_samples)):.4f}, "
      f"sd={float(mu_samples.std()):.4f}")
print(f"  s:   mean={float(s_samples.mean()):.4f}, "
      f"median={float(jnp.median(s_samples)):.4f}, "
      f"sd={float(s_samples.std()):.4f}")
print(f"  p:   mean={float(p_samples.mean()):.4f}, "
      f"median={float(jnp.median(p_samples)):.4f}")

# Print summary for global parameters only
import io, sys
buf = io.StringIO()
old = sys.stdout
sys.stdout = buf
mcmc.print_summary(prob=0.9, exclude_deterministic=False)
sys.stdout = old
for line in buf.getvalue().splitlines():
    if any(k in line for k in ["mu ", " s ", " p ", "beta", "n_eff", "r_hat", "diverge"]):
        print("  " + line)
