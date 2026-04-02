"""Plot distributions of genotypic values in cases vs controls."""
import numpy as np
import matplotlib.pyplot as plt

from geneticinfo_functions import simulate_casecontrol_related, print_casecontrol_summary

# Run simulation with default settings
y, A, L, info, g = simulate_casecontrol_related(return_genotypic_values=True)
print_casecontrol_summary(info)

g_cases = g[y == 1]
g_controls = g[y == 0]

print(f"\nGenotypic values — cases:    mean={g_cases.mean():.3f}, sd={g_cases.std():.3f}")
print(f"Genotypic values — controls: mean={g_controls.mean():.3f}, sd={g_controls.std():.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(g.min(), g.max(), 60)
ax.hist(g_controls, bins=bins, alpha=0.6, density=True, label="Controls")
ax.hist(g_cases, bins=bins, alpha=0.6, density=True, label="Cases")
ax.set_xlabel("Genotypic value")
ax.set_ylabel("Density")
ax.set_title(f"Distribution of genotypic values (mu={info['mu']:.1f}, K={info['K_target']})")
ax.legend()
fig.tight_layout()
fig.savefig("genotypic_values.png", dpi=150)
print("\nPlot saved to genotypic_values.png")
