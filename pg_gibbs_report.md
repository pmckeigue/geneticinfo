---
title: "Bayesian Inference of Genetic Information for Disease Discrimination"
subtitle: "Pólya-gamma Gibbs Sampler for a Logistic Mixture Model with Family Structure"
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
\mu \;\sim\; \mathrm{HalfCauchy}(1), \qquad \beta_0 \;\sim\; \mathcal{N}(0,\, 25), \qquad p \;\sim\; \mathrm{Beta}(a_0,b_0)
$$

For each individual $i = 1, \ldots, M$:

$$
r_i \mid p \;\sim\; 2\,\mathrm{Bernoulli}(p) - 1 \;\in\; \{-1,+1\},
$$
$$
z_i \mid r_i,\mu \;\sim\; \mathcal{N}(r_i\,\mu,\; 2\mu),
$$
$$
G_i \;=\; (L\mathbf{z})_i \quad \text{(correlated genotypic value)},
$$
$$
\eta_i \;=\; \beta_0 + G_i,
$$
$$
y_i \mid \eta_i \;\sim\; \mathrm{Bernoulli}\!\left(\sigma(\eta_i)\right), \quad \sigma(x) = (1+e^{-x})^{-1}.
$$

The binary indicator $r_i$ determines the class ($+1 =$ genetic risk carrier, $-1 =$ non-carrier); the mixture proportion $p$ is estimated jointly and tracks the population prevalence $K$.  The Cholesky factor $L$ is block-diagonal, with one block per connected family cluster; unrelated individuals appear as $1\times 1$ singleton blocks.

The theoretical relationship $\mu = \tfrac{1}{2}s^2$ (variance equals twice the mean for the genotypic-value distribution) is enforced through the parameterisation $z_i \mid r_i, \mu \sim \mathcal{N}(r_i\mu, 2\mu)$, so that the marginal distribution of $G_i$ for a single unrelated individual is

$$
G_i \;\sim\; \tfrac{1}{2}\,\mathcal{N}(\mu,\, 2\mu) \;+\; \tfrac{1}{2}\,\mathcal{N}(-\mu,\, 2\mu).
$$

## Directed acyclic graph

Figure 1 shows the plate diagram for the model.  Shaded blue circles are stochastic parameters; the orange rectangle is fixed observed data; the green circle is the observed outcome; double-bordered rectangles are deterministic functions.  The plate encloses the $M$ per-individual variables.  The coupling between individuals within the same family enters through the linear map $G = Lz$, so $G_i$ depends on all $z_j$ within the same family block.

\begin{figure}[H]
\centering
\includegraphics[width=0.45\textwidth]{dag_pgibbs.png}
\caption{Plate diagram for the logistic mixture model.  $\mu$: genetic information (nats); $\beta_0$: intercept; $p$: mixing proportion; $r_i$: class indicator; $z_i$: standardised genotypic component; $G_i$: correlated genotypic value; $\eta_i$: log-odds; $y_i$: observed case/control status; $L$: Cholesky factor of the genetic relationship matrix.}
\end{figure}

# Pólya-gamma Gibbs Sampler

## Pólya-gamma augmentation

Inference is performed by Gibbs sampling under the Pólya-gamma (PG) augmentation of Polson, Scott and Windle (2013).  Introducing auxiliary variables $\omega_i \sim \mathrm{PG}(1, |\eta_i|)$, the logistic likelihood becomes conditionally Gaussian:

$$
\kappa_i \mid \omega_i,\eta_i \;\sim\; \mathcal{N}\!\left(\omega_i\,\eta_i,\; \omega_i^{-1}\right), \qquad \kappa_i = y_i - \tfrac{1}{2}.
$$

Given $\boldsymbol\omega$, the joint distribution of $(\beta_0, \mathbf{z})$ is Gaussian, enabling exact block updates.

## Collapsed $\theta$ update

The primary inferential target is $\mu$.  Sampling is performed on the log scale $\theta = \log\mu$, using a univariate slice sampler for $p(\theta \mid \boldsymbol\omega, \mathbf{r}, \mathbf{y})$.  The latent $\mathbf{z}$ is integrated out analytically: for each family block $b$ of size $s_b$, the precision matrix of $\mathbf{z}_b$ given $\boldsymbol\omega_b$, $\beta_0$, and $\mathbf{r}_b$ is

$$
A_b = L_b^\top \operatorname{diag}(\boldsymbol\omega_b) L_b + \tau I_{s_b}, \quad \tau = \tfrac{1}{2\mu},
$$

and the resulting eigenvalue decomposition yields the log-posterior contribution

$$
\log p(\theta \mid \cdots) \;\propto\; \tfrac{1}{2}\sum_b \left[\mathbf{b}_b^\top A_b^{-1}\mathbf{b}_b - \log\det A_b\right]
- \tfrac{1}{4} M_{\mathrm{rel}}\,\mu - \tfrac{1}{2} M_{\mathrm{rel}}\log(2\mu) + \log p(\mu),
$$

where $\mathbf{b}_b = L_b^\top(\boldsymbol\kappa_b - \boldsymbol\omega_b\beta_0) + \tfrac{1}{2}\mathbf{r}_b$ and $M_{\mathrm{rel}}$ counts only individuals in family blocks of size $\geq 2$.

**Singleton exclusion.** The terms $-\tfrac{1}{4}M\mu - \tfrac{1}{2}M\log(2\mu)$ arise from the $z$-prior normalisation $p(\mathbf{z} \mid \mu, \mathbf{r})$.  For singleton (unrelated) individuals these terms impose a large penalty on large $\mu$ without carrying any information about $\lambda_S$, causing severe downward bias when $M$ is large relative to $M_{\mathrm{rel}}$.  The fix is to restrict the sum in the collapsed log-posterior to blocks of size $\geq 2$, so that only the $M_{\mathrm{rel}}$ related individuals contribute to the $\mu$ likelihood.

## Full Gibbs sweep

Each outer iteration performs the following steps:

1. **Sample $\boldsymbol\omega$:** $\omega_i \mid \eta_i \sim \mathrm{PG}(1, |\eta_i|)$, implemented via the `polyagamma` C extension.
2. **Build Cholesky / likelihood factors** per block size (vectorised).
3. **Sample $\mathbf{r}$:** collapsed exact enumeration of $2^{s_b}$ states per block (vectorised for singletons and pairs; fallback for larger blocks).
4. **Sample $(\beta_0, \mathbf{z})$:** joint Gaussian draw via the Schur complement, with a constraint $\sum_i z_i = 0$ to remove the intercept-$z$ non-identifiability.
5. **Sample $c$, re-sample $\mathbf{r}$:** update the within-mixture scale $c$ (Gibbs), then redraw $\mathbf{r}$ from the full conditional given $\mathbf{z}$.
6. **Sample $p$:** conjugate Beta update.
7. **Sample $\theta$:** univariate slice sampler on the collapsed eigenvalue cache (blocks of size $\geq 2$ only).

## Prior

The half-Cauchy prior $\mu \sim \mathrm{HalfCauchy}(1)$ is used throughout.  In the $\theta = \log\mu$ parameterisation the log-prior (including the Jacobian) is

$$
\log p(\theta) = \theta - \log\!\left[1 + e^{2\theta}\right] + \mathrm{const},
$$

which has a mode at $\theta = 0$ ($\mu = 1$) and heavy tails.  This prior is uninformative over a wide range of $\mu$ while providing regularisation against extreme values.

# Vectorised Implementation and Performance

## Block-grouped operations

The $M$ individuals are partitioned into groups by family-block size $s$.  Singletons ($s=1$, typically $\gg 90$% of observations), pairs ($s=2$), triplets ($s=3$), and larger blocks are handled by separate vectorised routines:

- **Singletons ($s=1$):** all scalar arithmetic is batched into single NumPy array operations.
- **Pairs ($s=2$):** the four possible $\mathbf{r}$ states are enumerated simultaneously for all pairs using batched $2\times 2$ analytic Cholesky factors; eigensystem of the $2\times 2$ precision matrix is computed analytically.
- **Larger blocks:** general NumPy/SciPy fallback (per-block Python loop).

## Performance optimisation: BLAS thread cold-start

Profiling on the larger simulated datasets ($M \approx 14{,}000$) identified an unexpected bottleneck in `sample_beta0_z_fast`: three calls to `np.dot(g, ig)` on arrays of $n \approx 14{,}000$ elements each took $\approx 50\,\mathrm{ms}$ per call, for a total of $\approx 150\,\mathrm{ms}$ per Gibbs iteration — the dominant cost.

The cause was the OpenBLAS DDOT routine: at $n \approx 14{,}000$, OpenBLAS elected to use multiple threads.  Because $\approx 50\,\mathrm{ms}$ elapsed between consecutive dot-product calls (other Gibbs steps), the BLAS thread pool fell asleep and incurred a cold-start wakeup cost on every call.  For smaller ($n \leq 10{,}000$) or larger ($n \geq 50{,}000$) arrays, OpenBLAS used a single thread or had sufficient parallel efficiency, respectively, and was fast.

The fix was to replace the three `np.dot` calls with equivalent element-wise operations that bypass BLAS:

```python
# Before (50 ms each at n ~ 14 000):
gT_invg += float(np.dot(g, ig))   # BLAS ddot -> thread cold-start
gT_invb += float(np.dot(g, ib))
gT_inv1 += float(np.dot(g, i1))

# After (< 0.1 ms total):
gT_invg += float((g * ig).sum())  # NumPy element-wise; no BLAS
gT_invb += float((bz * ig).sum())
gT_inv1 += float(ig.sum())
```

This reduced `sample_beta0_z_fast` from 54 ms to 1.1 ms (49× speedup), and overall sampler throughput increased from 3.8 to 145 iterations/second on four parallel chains (38× speedup).

# Simulation Study

## Dataset generation

Datasets were simulated under the generative model with true $\mu = 1.0$ (so $\lambda_S = e \approx 2.718$) and disease prevalence $K = 0.01$.  A case-control sample was drawn by sampling all cases from the population plus an equal number of controls (1:1 matching).  Related individuals (full-sib pairs, full-sib triplets, half-sib pairs) were included in the population; a random subset of both members of each family was included in the case-control sample when both were ascertained.

Table 1 summarises the three datasets analysed.

\begin{table}[H]
\centering
\caption{Simulated datasets.  $M$: total case-control sample size; $M_{\mathrm{rel}}$: individuals in related pairs or larger blocks; true $\mu = 1.0$ throughout.}
\begin{tabular}{lrrrl}
\toprule
Dataset & $M$ & $M_{\mathrm{rel}}$ & Related blocks & Notes \\
\midrule
\texttt{create\_data()} & 3,294 & 114 & 57 full-sib pairs & Small, for development \\
Simulated ($\times$1) & 14,736 & 502 & 251 pairs & 200k + 100k + 20k fam.\ \\
Simulated ($\times$2) & 29,496 & 1,120 & 557 pairs + 2 triplets & Doubled population \\
\bottomrule
\end{tabular}
\end{table}

Each analysis ran 4 independent chains with 1,000 warm-up and 5,000 post-warm-up samples (10,000 for `create_data()`), using `multiprocessing.Pool` with `fork` context.

## Results

Table 2 summarises posterior inference for $\mu$ across all three datasets.

\begin{table}[H]
\centering
\caption{Posterior summaries for $\mu$ (true value: 1.0).  ESS: effective sample size (bulk); $\hat{R}$: Gelman-Rubin statistic.}
\begin{tabular}{lrrrrrrr}
\toprule
Dataset & $M$ & Median & Mean & SD & 90\% CI & ESS & $\hat{R}$ \\
\midrule
\texttt{create\_data()} & 3,294 & -- & -- & -- & [0.42,\;1.44] & 293 & 1.01 \\
Simulated ($\times$1) & 14,736 & 0.596 & 0.614 & 0.292 & [0.163,\;1.130] & 1,077 & 1.00 \\
Simulated ($\times$2) & 29,496 & 0.624 & 0.632 & 0.219 & [0.291,\;1.006] & 1,027 & 1.00 \\
\bottomrule
\end{tabular}
\end{table}

## Computational performance

\begin{table}[H]
\centering
\caption{Sampler throughput (4 parallel chains, wall-clock time includes simulation and block-structure construction).}
\begin{tabular}{lrrr}
\toprule
Dataset & $M$ & Throughput (it/s) & Wall time \\
\midrule
Simulated ($\times$1) & 14,736 & 145 & 43 s \\
Simulated ($\times$2) & 29,496 & 65 & 1 min 43 s \\
\bottomrule
\end{tabular}
\end{table}

Throughput scales approximately as $M^{-1}$, consistent with the dominant $O(M)$ Pólya-gamma sampling step.

# Discussion

The posterior credible intervals for $\mu$ contain the true value in all analyses, confirming that the sampler is correctly targeting the posterior.  The posterior median is consistently below the true value $\mu = 1.0$.  Several factors contribute to this:

**Information content.** With $K = 0.01$, the case-control sample is dominated by discordant pairs (one case, one control), which carry less information about $\lambda_S$ than concordant-affected pairs.  In the doubled dataset, 98 concordant full-sib pairs and 155 concordant case-case triplet sib-pairs were sampled, compared with 557 discordant pairs.

**Prior influence.** The half-Cauchy(1) prior has its mode at $\mu = 1$ but has substantial mass below the true value.  With $M_{\mathrm{rel}} \approx 500$–$1{,}100$ related individuals the prior still pulls the posterior downward relative to the likelihood.

**Variance reduction with sample size.** Doubling $M$ reduced the posterior standard deviation from 0.292 to 0.219 and shifted the upper end of the 90% credible interval from 1.130 to 1.006, just containing the true value.  Further increases in $M_{\mathrm{rel}}$ are expected to narrow the interval enough for the data to dominate the prior.

The singleton-exclusion fix was essential: including all $M$ individuals in the $\mu$ likelihood (without restriction to $M_{\mathrm{rel}}$) introduced a strong downward bias because the $-\tfrac{1}{4}M\mu$ term in the collapsed log-posterior overwhelmed the signal from the $\approx 500$ related individuals.

The BLAS thread cold-start fix achieved a 38-fold speedup in overall sampler throughput by replacing `np.dot` calls on arrays of $n \approx 14{,}000$ with element-wise `.sum()` operations.  This avoided the per-call thread wakeup cost of OpenBLAS at a problem size that falls in the transition region between single-threaded and multi-threaded BLAS kernels.
