"""Simulate case-control data with related individuals and fit with SVI.

Example usage:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 python run_example.py
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geneticinfo import (
    simulate_casecontrol_related,
    print_casecontrol_summary,
    fit_svi_lowrank,
)

print("JAX devices:", jax.devices())
print()

# --- Simulate dataset ---
print("Simulating case-control dataset (mu=1.0, K=0.01) ...")
y_np, A_np, L_np, info = simulate_casecontrol_related()
print_casecontrol_summary(info)

M = info["M"]
y = jnp.array(y_np, dtype=jnp.float64)
L = jnp.array(L_np, dtype=jnp.float64)

# --- Fit with SVI ---
print(f"\nFitting logistic_mvnorm with SVI (M={M}) ...")
result = fit_svi_lowrank(M, L, y)

# --- Print posterior summary ---
print("\n=== SVI Posterior Summary ===")
print(f"  {'Param':>6s}  {'mean':>8s}  {'sd':>8s}  {'3%':>8s}  {'97%':>8s}  {'true':>8s}")
print(f"  {'-'*50}")
true_vals = {"beta0": info["alpha"], "s": info["s"], "mu": info["mu"]}
for name in ["beta0", "s", "mu"]:
    samples = result[name]
    m = float(jnp.mean(samples))
    sd = float(jnp.std(samples))
    q03 = float(jnp.percentile(samples, 3))
    q97 = float(jnp.percentile(samples, 97))
    tv = true_vals[name]
    print(f"  {name:>6s}  {m:8.4f}  {sd:8.4f}  {q03:8.4f}  {q97:8.4f}  {tv:8.4f}")

# --- ELBO convergence ---
losses = result["losses"]
print(f"\nELBO loss: step 1000 = {float(losses[999]):.1f}, "
      f"step {len(losses)} = {float(losses[-1]):.1f}")

# --- Plot ELBO trace ---
losses_np = np.asarray(losses)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(losses_np, linewidth=0.5)
ax.set_xlabel("SVI step")
ax.set_ylabel("ELBO loss")
ax.set_title("ELBO loss trace")
fig.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__) or ".", "elbo_trace.png")
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\nELBO trace plot saved to {plot_path}")
