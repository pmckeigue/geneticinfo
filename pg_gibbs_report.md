---
title: "Bayesian Inference of Genetic Information for Disease Discrimination"
subtitle: "Pólya-gamma Gibbs Sampler Targeting the lr_discrete_blockdiag Model"
date: "March 2026"
geometry: margin=2.5cm
fontsize: 11pt
numbersections: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{microtype}
  - \usepackage{graphicx}
  - \renewcommand{\arraystretch}{1.25}
---

# Introduction

A central quantity in genetic epidemiology is the expected log-likelihood ratio (weight of evidence, in nats) for predicting an individual's disease status from their genome, known here as **genetic information** $\mu$.  Under a Gaussian liability model this equals $0.5 s^2$, where $s$ is the scale of the class-conditional log-likelihood ratio distribution.  Equivalently, $\lambda_S = e^\mu$ is the sibling recurrence-risk ratio, the standard measure of familial aggregation.

The goal is to infer $\mu$ from a case-control dataset in which some sampled individuals are biological relatives, without access to individual genotypes.  Instead, the genetic relationship matrix (GRM) $\Phi$, computed from genome-wide SNP data, is used to describe the covariance structure.

# Statistical Model

## Generative model

Let $\mathbf{y} \in \{0,1\}^M$ denote case-control status for $M$ individuals and let $\Phi = LL^\top$ be the Cholesky decomposition of the GRM.  The model is a logistic regression with a two-component mixture random effect:

$$
\mu \;\sim\; \mathrm{HalfCauchy}(2), \qquad \phi = \mathrm{logit}(p) \;\sim\; \mathcal{N}(0,\, \sigma_\phi^2), \qquad p \;\sim\; \mathrm{Beta}(20\,p_{\mathrm{obs}},\; 20\,(1-p_{\mathrm{obs}}))
$$

For each individual $i = 1, \ldots, M$:

$$
r_i \mid p \;\sim\; 2\,\mathrm{Bernoulli}(p) - 1 \;\in\; \{-1,+1\},
$$
$$
Z_{\mathrm{mix},i} \mid r_i,\mu \;\sim\; \mathcal{N}(r_i\,\mu,\; 2\mu),
$$
$$
\eta_i \;=\; \phi + (L\mathbf{Z}_{\mathrm{mix}})_i - \bar\Delta\,(L\mathbf{1})_i, \qquad \bar\Delta = (2p-1)\,\mu,
$$
$$
y_i \mid \eta_i \;\sim\; \mathrm{Bernoulli}\!\left(\sigma(\eta_i)\right), \quad \sigma(x) = (1+e^{-x})^{-1}.
$$

The binary indicator $r_i$ determines the class ($+1 =$ genetic risk carrier, $-1 =$ non-carrier); the mixing proportion $p$ estimates the population prevalence $K$.  The Cholesky factor $L$ is block-diagonal, one block per connected family cluster.  The shift $\bar\Delta(L\mathbf{1})_i$ centres the logit on the marginal mean of $Z_{\mathrm{mix}}$.

The theoretical constraint $\mu = \tfrac{1}{2}s^2$ (variance equals twice the mean for the genotypic value) is enforced through the parameterisation $Z_{\mathrm{mix},i} \mid r_i, \mu \sim \mathcal{N}(r_i\mu, 2\mu)$, so the marginal genotypic-value distribution is

$$
(L\mathbf{Z}_{\mathrm{mix}})_i \;\sim\; \tfrac{1}{2}\,\mathcal{N}(\mu,\, 2\mu) \;+\; \tfrac{1}{2}\,\mathcal{N}(-\mu,\, 2\mu).
$$

## Reduction to relatives

Unrelated individuals (singletons in the GRM) contribute no information about $\lambda_S$: their $Z_{\mathrm{mix}}$ terms do not covary with any other individual, so after marginalising $Z_{\mathrm{mix}}$ the likelihood factors for singletons are flat in $\mu$.  All singletons are therefore discarded before fitting, retaining only the $M_{\mathrm{rel}} \ll M$ individuals in family blocks of size $\geq 2$.

# Pólya-gamma Gibbs Sampler

## Pólya-gamma augmentation

Inference is performed by Gibbs sampling under the Pólya-gamma (PG) augmentation of Polson, Scott and Windle (2013).  Introducing auxiliary variables $\omega_i \sim \mathrm{PG}(1, |\eta_i|)$, the logistic likelihood becomes conditionally Gaussian:

$$
\kappa_i \mid \omega_i,\eta_i \;\sim\; \mathcal{N}\!\left(\omega_i\,\eta_i,\; \omega_i^{-1}\right), \qquad \kappa_i = y_i - \tfrac{1}{2}.
$$

Given $\boldsymbol\omega$, the full conditional of $\mathbf{Z}_{\mathrm{mix}}$ is Gaussian with precision matrix $L^\top\operatorname{diag}(\boldsymbol\omega)L + \tau I$ (where $\tau = 1/(2\mu)$ from the Gaussian prior) and a linear right-hand side.

## Full Gibbs sweep

The `LRDiscreteBlockdiagPGGibbs` sampler performs the following steps each iteration:

1. **Sample $\boldsymbol\omega$:** $\omega_i \mid \eta_i \sim \mathrm{PG}(1, |\eta_i|)$, implemented via the `polyagamma` C extension.

2. **Sample $\mathbf{Z}_{\mathrm{mix}}$:** joint Gaussian draw for each family block $b$,
$$
\mathbf{Z}_{\mathrm{mix},b} \mid \boldsymbol\omega_b, \kappa_b, \phi, \mu, \mathbf{r}_b \;\sim\; \mathcal{N}(A_b^{-1}\mathbf{b}_b,\; A_b^{-1}),
$$
where $A_b = L_b^\top\operatorname{diag}(\boldsymbol\omega_b)L_b + \tau I$ and $\mathbf{b}_b = L_b^\top(\boldsymbol\kappa_b - \boldsymbol\omega_b c_b) + \tfrac{1}{2}\mathbf{r}_b$, with $c_b = \phi - \bar\Delta v_b$ the per-individual offset and $v_b = (L_b\mathbf{1})_b$.  For size-2 blocks the Cholesky solve is fully vectorised without LAPACK calls.

3. **Sample $\mathbf{r}$:** exact Bernoulli draw per individual,
$$
p(r_i = +1 \mid Z_{\mathrm{mix},i}, \phi) = \sigma(Z_{\mathrm{mix},i} + \phi).
$$

4. **Sample $\phi$:** univariate slice sampler on the PG-augmented log-posterior,
$$
\log p(\phi \mid \boldsymbol\omega, \mathbf{Z}_{\mathrm{mix}}, \mathbf{r}, \mathbf{y}) \;\propto\; \boldsymbol\kappa^\top\boldsymbol\eta(\phi) - \tfrac{1}{2}\boldsymbol\omega^\top\boldsymbol\eta(\phi)^2 + (a-1)\log p + (b-1)\log(1-p) + n_+\log p + n_-\log(1-p),
$$
where $n_+, n_-$ are the current counts of $r_i = +1$ and $r_i = -1$, and $a = 20\,p_{\mathrm{obs}}$, $b = 20(1-p_{\mathrm{obs}})$ are the Beta prior parameters.

5. **Sample $\mu$ (collapsed):** slice sampler on the log-posterior $p(\theta \mid \boldsymbol\omega, \mathbf{r}, \phi, \mathbf{y})$ with $\theta = \log\mu$, marginalising $\mathbf{Z}_{\mathrm{mix}}$ analytically.  For each family block $b$ the eigendecomposition $A_b = U_b\operatorname{diag}(\boldsymbol\lambda_b)U_b^\top$ yields
$$
\log p(\theta \mid \cdots) \;\propto\; \log p(\theta) - \tfrac{1}{2}M_{\mathrm{rel}}\log(2\mu) - \tfrac{1}{4}M_{\mathrm{rel}}\mu + C(\boldsymbol\omega,\phi,\mu) + \tfrac{1}{2}\sum_b\sum_j \frac{(p_{0,bj} + \mu\,\bar m\,p_{1,bj})^2}{\lambda_{bj}+\tau} - \tfrac{1}{2}\sum_b\sum_j\log(\lambda_{bj}+\tau),
$$
where $\bar m = \tanh(\phi/2)$, $\tau = 1/(2\mu)$, $p_{0,bj} = (U_b^\top\mathbf{b}_{0,b})_j$, $p_{1,bj} = (U_b^\top\mathbf{b}_{1,b})_j$ are projections of the $\mu$-independent and $\mu$-dependent parts of the right-hand side onto the eigenvectors, and $C(\boldsymbol\omega,\phi,\mu)$ collects the offset-only quadratic terms.

6. **Refresh $\mathbf{Z}_{\mathrm{mix}}$:** repeat step 2 under the updated $\mu$ to improve mixing.

## Prior on $\mu$

The half-Cauchy prior $\mu \sim \mathrm{HalfCauchy}(2)$ is used throughout.  In the $\theta = \log\mu$ parameterisation the log-prior (including the Jacobian) is

$$
\log p(\theta) = \theta - \log\!\left[1 + \left(e^\theta/2\right)^2\right] + \mathrm{const},
$$

which is flat around $\theta = 0$ ($\mu = 1$) and provides regularisation against extreme values.

# Simulation Study

## Simulated population and case-control sample

The generative model was simulated with true $\mu = 2.0$ ($\lambda_S = e^2 \approx 7.39$) and disease prevalence $K = 0.01$.  The population comprised 400,000 full-sib pairs (genetic correlation 0.5), 200,000 full-sib triplets (genetic correlation 0.5), 40,000 half-sib pairs (genetic correlation 0.25), and 4,000 unrelated individuals (1,484,000 total).  A 1:1 case-control sample retained all 14,761 cases and an equal number of controls.

\begin{table}[H]
\centering
\caption{Case-control sample characteristics (seed = 42, true $\mu = 2.0$, $K = 0.01$).}
\begin{tabular}{lr}
\toprule
Quantity & Count \\
\midrule
Total sample $M$ & 29,522 \\
Cases / Controls & 14,761 / 14,761 \\
Full-sib pairs (both sampled) & 283 \\
\quad of which concordant-affected & 166 \\
Triplet sib-pairs (both sampled) & 400 \\
\quad concordant case-case & 241 \\
\quad discordant & 107 \\
\quad concordant ctrl-ctrl & 52 \\
Half-sib pairs (both sampled) & 19 \\
\midrule
$M_{\mathrm{rel}}$ (block size $\geq 2$) & 1,374 \\
\quad Size-2 blocks (pairs) & 672 \\
\quad Size-3 blocks (triplets) & 10 \\
\bottomrule
\end{tabular}
\end{table}

## Comparison with DiscreteHMCGibbs

The `LRDiscreteBlockdiagPGGibbs` sampler was compared with NumPyro's `DiscreteHMCGibbs` (NUTS inner kernel, `modified=True`) on the same data.  Both algorithms operated on the same $M_{\mathrm{rel}} = 1{,}374$ relatives.  Four chains were run for each algorithm.

\begin{table}[H]
\centering
\caption{Posterior summaries for $\mu$ (true value: 2.0).  ESS: effective sample size (bulk); $\hat{R}$: Gelman-Rubin statistic.}
\begin{tabular}{lrrrrrrrr}
\toprule
Algorithm & Warmup & Samples & Median & SD & 90\% CI & ESS & $\hat{R}$ & Wall time \\
\midrule
PG-Gibbs & 1,000 & 5,000 & 2.167 & 0.993 & [1.253,\;4.230] & 189 & 1.030 & 87 s \\
DiscreteHMCGibbs & 2,000 & 2,000 & 2.134 & 0.889 & [1.259,\;3.937] & 250 & 1.020 & 416 s \\
\bottomrule
\end{tabular}
\end{table}

The posterior medians agree closely (2.167 vs 2.134) and both 90% credible intervals contain the true value $\mu = 2.0$.  The PG-Gibbs sampler is approximately 5× faster in wall time (87 s vs 416 s) on CPU versus GPU respectively, at a cost of lower ESS per sample due to slower mixing.

![Prior, posterior and likelihood for both algorithms on the same dataset.](comparison_prior_posterior_likelihood.png)

## Discussion

**Agreement between samplers.**  The close agreement in posterior medians (difference $< 2\%$) and overlapping credible intervals confirm that `LRDiscreteBlockdiagPGGibbs` is targeting the same posterior as `DiscreteHMCGibbs`.  The wider credible interval from PG-Gibbs (width 2.977 vs 2.678) reflects somewhat slower mixing (IAT $\approx 106$ vs $\approx 32$).

**Information content.**  With $K = 0.01$ and 1:1 case-control matching, most relative pairs in the sample are discordant, carrying less information about $\lambda_S$ than concordant-affected pairs.  The half-Cauchy(2) prior retains considerable influence with $M_{\mathrm{rel}} = 1{,}374$.

**Singleton exclusion.**  Only the $M_{\mathrm{rel}}$ relatives contribute to the collapsed $\mu$ log-posterior.  Including singletons (whose $Z_{\mathrm{mix}}$ terms do not covary with any other individual) would introduce a large spurious downward bias via the $-\tfrac{1}{4}M\mu$ term.

**Mixing.**  The PG-Gibbs sampler mixes more slowly than NUTS because the $\mu$ update conditions on $\boldsymbol\omega$ (which is sampled at the current $\mu$), introducing autocorrelation.  The collapsed update (integrating out $\mathbf{Z}_{\mathrm{mix}}$ analytically) reduces but does not eliminate this dependence.

# References

- Clayton DG (2009). Prediction and interaction in complex disease genetics: experience in type 1 diabetes. *PLoS Genetics*, 5(7), e1000540.
- McKeigue PM (2019). Quantifying performance of a diagnostic test as the expected information for discrimination: relation to the C-statistic. *Statistical Methods in Medical Research*, 28(6), 1841–1851.
- Polson NG, Scott JG, Windle J (2013). Bayesian inference for logistic models using Pólya-gamma latent variables. *Journal of the American Statistical Association*, 108(504), 1339–1349.
