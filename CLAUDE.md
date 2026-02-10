# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for Bayesian inference of **genetic information for discrimination** from binary (case/control) traits. The core statistical problem is estimating the expected log-likelihood ratio (weight of evidence) for genetic prediction of disease status, parameterized through a logistic mixed model with correlated random effects derived from genetic relationship matrices.

The key quantity being estimated is **Lambda** (expected information for discrimination in nats), which equals `0.5 * s^2` where `s` is the scale factor of the class-conditional log-likelihood ratio distribution. Under the logistic polygenic model, Lambda equals `log(lambda_S)` where `lambda_S` is the sibling recurrence risk ratio.

## Key Statistical Models

All models share the structure: observed binary outcome `y` is Bernoulli with logits = `beta0 + G`, where `G = L @ Z` incorporates genetic correlation via the Cholesky factor `L` of the genetic relationship matrix.

- **`logistic_mvnorm`** — Gaussian random effects: `z ~ Normal(0,1)`, `G = s * (L @ z)`. Simpler but does not model the mixture structure.
- **`logistic_mix2_mvnorm`** — Two-component mixture random effects: `W ~ MixtureSameFamily` with class-conditional means `[-mu, mu]` and equal scales `s`, where `mu = 0.5 * s^2`. The deterministic relationship `mu = 0.5 * s^2` enforces the theoretical constraint that variance equals twice the mean for the log-likelihood ratio distribution.
- **`lr_discrete`** — Same mixture model but with an explicit discrete latent variable `z` (Categorical) for class membership, using `config_enumerate` for marginalization and `LocScaleReparam` for the continuous component. Sampled with `DiscreteHMCGibbs`.

## Python Modules

- **`geneticinfo.py`** — Contains the `logistic_mix2_mvnorm` model definition and a simulated sib-pair dataset generator (builds block-diagonal genetic correlation matrix with 0.5 for sibling pairs).
- **`utils_mcmc.py`** — `nuts_pdf_report()` generates a multi-page PDF diagnostic report from ArviZ `InferenceData`: trace plots, ESS, energy, autocorrelation, rank plots, pair plots with divergences, tree depth histograms.

## Notebooks

- **`geneticinfo.ipynb`** — Main notebook: loads real UK Biobank data (`A_matrix`, `y_binary`) via `rdata`, fits `logistic_mvnorm` with both NUTS MCMC and SVI (AutoNormal guide), includes NeuTraReparam approach.
- **`GeneticInferenceBinaryTrait.ipynb`** — Fits `logistic_mix2_mvnorm` on simulated sib-pair data using normalizing flow VI (`MaskedCouplingRQSpline` from flowMC) and NUTS MCMC (with and without NeuTraReparam).
- **`geneticinfo_svi.ipynb`** — SVI with `AutoNormal` guide and `RenyiELBO` on real UK Biobank data. Derives posterior quantities: `info = 0.5 * s^2 / log(2)` (bits), `lambda_S = exp(0.5 * s^2)`, prevalence `K`.
- **`mlmc_multi.ipynb`** — Microcanonical Langevin Monte Carlo (MCLMC) via BlackJAX `adjusted_mclmc_dynamic`. Includes multi-chain tuning with metric pooling, pmap across GPUs, and `accepted_only` filtering. Also has `DiscreteHMCGibbs` with `lr_discrete`.
- **`discrete_gibbs_hmc.ipynb`** — `DiscreteHMCGibbs` + NUTS for the `lr_discrete` model with enumerated discrete latent variables.
- **`geneticinfo_pymc.ipynb`** — PyMC port of `logistic_mvnorm` (incomplete, hits Cholesky errors from non-positive-definite matrices).
- **`grouphevo_pymc.ipynb`** — Separate model (`mrhevo`): Mendelian randomization with horseshoe-like sparsity prior in PyMC. Not directly related to the genetic info models.

## Technology Stack

- **Primary framework**: NumPyro (JAX-based) for model definition and inference
- **Alternative frameworks**: PyMC (pytensor-based) in some notebooks, BlackJAX for MCLMC
- **Inference methods**: NUTS MCMC, SVI (AutoNormal, AutoBNAFNormal, normalizing flows), DiscreteHMCGibbs, MCLMC
- **Diagnostics**: ArviZ for posterior analysis, summary statistics, and plotting
- **Data**: UK Biobank genetic relationship matrices loaded from `.RData` files via `rdata` package
- **Hardware**: Multi-GPU (Tesla V100-SXM2-32GB), uses `jax.pmap` and `numpyro.set_host_device_count(4)`

## Environment Notes

- JAX 64-bit mode is enabled: `jax.config.update("jax_enable_x64", True)`
- GPU memory fraction is set via `XLA_PYTHON_CLIENT_MEM_FRACTION`
- Multi-GPU parallelism: MCMC chains run in parallel via `chain_method="parallel"` (NumPyro) or `jax.pmap` (BlackJAX)
- The genetic relationship matrix and its Cholesky factor are large dense matrices (~17000 x 17000 for real data), so memory management matters
- The `infohsq.R` script plots the relationship between expected information (bits) and heritability on the liability scale for different prevalences
